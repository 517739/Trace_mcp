#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
正常时段数据获取工具 (Baseline Data Fetcher) - 最终版
- 支持 --window-hours 自定义时间窗
- 支持 --file-name 自定义文件名后缀 (防止覆盖)
- 包含悬浮节点/断链严格检查
"""

import os
import sys
import json
import csv
import time
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import config

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ================= 🔧 配置区域 =================
# 1. 指标定义
TARGET_METRICS = [
    "aggregate_node_net_receive_packages_errors_per_minute",
    "aggregate_node_tcp_inuse_total_num",
    "aggregate_node_tcp_alloc_total_num",
    "aggregate_node_cpu_usage",
    "aggregate_node_memory_usage",
    "aggregate_node_disk_io_usage"
]

# 2. SLS 配置
PROJECT_NAME = config.SLS_PROJECT_NAME
LOGSTORE_NAME = config.SLS_LOGSTORE_NAME
REGION = config.SLS_REGION

# 3. 鉴权配置
os.environ.setdefault("ALIBABA_CLOUD_ROLE_SESSION_NAME", "normal-data-fetcher")

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入工具库
try:
    from tools.paas_entity_tools import umodel_get_entities
    from tools.paas_data_tools import umodel_get_golden_metrics
    from tools.common import create_cms_client, execute_cms_query
    from aliyun.log import LogClient, GetLogsRequest
    from alibabacloud_sts20150401.client import Client as StsClient
    from alibabacloud_sts20150401 import models as sts_models
    from alibabacloud_tea_openapi import models as open_api_models
    from tools.constants import REGION_ID, WORKSPACE_NAME
except ImportError as e:
    print(f"❌ 依赖缺失: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === 悬浮节点检查 ===
def check_orphan_root(spans: list) -> bool:
    """检查 Trace 是否存在断链 (允许最多 1 个悬浮根节点)"""
    if not spans: return False
    
    span_ids = set()
    for s in spans:
        sid = str(s.get('SpanId', '')).strip()
        if sid: span_ids.add(sid)
    
    roots = {"", "nan", "None", "null", "-1", "0"}
    dangling_count = 0
    
    for s in spans:
        pid = str(s.get('ParentID', '')).strip()
        if pid not in span_ids and pid not in roots:
            dangling_count += 1
            
    return dangling_count <= 1

class NormalDataFetcher:
    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.cms_client = create_cms_client(REGION_ID)
        self.sls_client = self._init_sls_client()

    def _init_sls_client(self):
        """初始化 SLS 客户端 (带 STS)"""
        config = open_api_models.Config(
            access_key_id=os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
            access_key_secret=os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
            endpoint=f'sts.{REGION}.aliyuncs.com'
        )
        sts_client = StsClient(config)
        resp = sts_client.assume_role(sts_models.AssumeRoleRequest(
            role_arn=os.environ["ALIBABA_CLOUD_ROLE_ARN"],
            role_session_name="normal-fetcher",
            duration_seconds=3600
        ))
        creds = resp.body.credentials
        return LogClient(
            endpoint=f"{REGION}.log.aliyuncs.com",
            accessKeyId=creds.access_key_id,
            accessKey=creds.access_key_secret,
            securityToken=creds.security_token
        )

    def determine_time_window(self):
        """步骤 1: 确定正常时间段"""
        logger.info(f"📅 正在扫描 {self.args.csv} 计算基准时间...")
        min_ts = float('inf')
        
        if os.path.exists(self.args.csv):
            with open(self.args.csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = int(datetime.strptime(row['start_time'], '%Y-%m-%d %H:%M:%S').timestamp())
                        if ts < min_ts: min_ts = ts
                    except: continue
        else:
            logger.warning(f"⚠️ 未找到 {self.args.csv}，使用当前时间作为基准")
            min_ts = int(time.time())
        
        # 定义：最早故障前 window_seconds小时 ~ 前 1小时
        end_time = min_ts - 60 * 60
        window_seconds = int(self.args.window_hours * 3600)
        start_time = end_time - window_seconds
        
        logger.info(f"✅ 选定正常时段: {datetime.fromtimestamp(start_time)} ~ {datetime.fromtimestamp(end_time)}")
        logger.info(f"   (窗口: {self.args.window_hours}h, 基准故障前缓冲: 1h)")
        return start_time, end_time

    def fetch_traces(self, start_ts, end_ts):
        """步骤 1: 获取 Trace 并提取活跃节点 ID"""
        logger.info("🚀 [Trace] 开始获取正常时段的 Trace...")
        
        query = "* | where try_cast(statusCode as bigint) <= 1"
        limit = self.args.trace_limit
        candidate_trace_ids = set()
        offset = 0
        target_candidates = int(limit * 2.0)
        
        # 1. 扫描 Trace ID
        while len(candidate_trace_ids) < target_candidates:
            req = GetLogsRequest(PROJECT_NAME, LOGSTORE_NAME, query=query, fromTime=start_ts, toTime=end_ts, line=100, offset=offset)
            try:
                res = self.sls_client.get_logs(req)
                if not res or not res.get_logs(): break
                logs = res.get_logs()
                for log in logs:
                    candidate_trace_ids.add(log.get_contents().get('traceId'))
                offset += len(logs)
                print(f"   已扫描 {offset} 条日志，发现 {len(candidate_trace_ids)} 个候选 TraceID...", end='\r')
                if len(logs) < 100: break
            except Exception as e:
                break
        
        # 2. 拉取 Trace 详情并提取 NodeName
        filename = f"normal_traces{self.args.file_name}.csv"
        csv_path = os.path.join(self.output_dir, filename)
        csv_headers = [
            'TraceID', 'SpanId', 'ParentID', 'ServiceName', 'NodeName', 'PodName', 
            'URL', 'SpanKind', 'StartTimeMs', 'EndTimeMs', 'DurationMs',
            'StatusCode', 'HttpStatusCode', 'fault_type', 'fault_instance', 'problem_id'
        ]
        
        batch_list = list(candidate_trace_ids)
        valid_trace_count = 0
        total_spans = 0
        
        # [新增] 用于收集 Trace 中出现的所有节点 ID
        active_nodes_in_trace = set()

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            
            for i in range(0, len(batch_list), 20):
                if valid_trace_count >= limit: break
                batch = batch_list[i:i+20]
                or_query = " OR ".join([f'traceId: "{tid}"' for tid in batch])
                trace_buffer = {tid: [] for tid in batch} 
                sub_offset = 0
                
                while True:
                    req = GetLogsRequest(PROJECT_NAME, LOGSTORE_NAME, query=or_query, fromTime=start_ts, toTime=end_ts, line=100, offset=sub_offset)
                    try:
                        res = self.sls_client.get_logs(req)
                        if not res or not res.get_logs(): break
                        logs = res.get_logs()
                        for log in logs:
                            d = log.get_contents()
                            tid = d.get('traceId')
                            if tid in trace_buffer:
                                try: res_obj = json.loads(d.get('resources', '{}'))
                                except: res_obj = {}
                                
                                # [关键] 提取 NodeName (假设它是 instance_id，如 i-xyz)
                                node_name = res_obj.get('k8s.node.name', '').strip()
                                
                                try:
                                    s_ms = int(d.get('startTime', 0)) / 1e6
                                    d_ms = int(d.get('duration', 0)) / 1e6
                                except: s_ms, d_ms = 0, 0
                                
                                span_obj = {
                                    'TraceID': tid,
                                    'SpanId': d.get('spanId'),
                                    'ParentID': d.get('parentSpanId'),
                                    'ServiceName': d.get('serviceName'),
                                    'NodeName': node_name,
                                    'PodName': res_obj.get('k8s.pod.name'),
                                    'URL': d.get('spanName'),
                                    'SpanKind': d.get('kind'),
                                    'StartTimeMs': f"{s_ms:.3f}",
                                    'EndTimeMs': f"{s_ms + d_ms:.3f}",
                                    'DurationMs': f"{d_ms:.3f}",
                                    'StatusCode': d.get('statusCode'),
                                    'HttpStatusCode': "",
                                    'fault_type': 'normal',
                                    'fault_instance': 'unknown',
                                    'problem_id': 'normal_000'
                                }
                                trace_buffer[tid].append(span_obj)
                        sub_offset += len(logs)
                        if len(logs) < 100: break
                    except: break
                
                rows_to_save = []
                for tid, spans in trace_buffer.items():
                    if not spans or len(spans) < 2: continue
                    is_error = False
                    for span in spans:
                        try:
                            sc = int(span['StatusCode']) if span['StatusCode'] and span['StatusCode'].isdigit() else 0
                            if sc > 1: is_error = True; break
                        except: pass
                    if is_error: continue
                    if not check_orphan_root(spans): continue
                    
                    # [新增] 收集活跃节点
                    for span in spans:
                        nm = span['NodeName']
                        if nm and nm != 'nan':
                            active_nodes_in_trace.add(nm)

                    rows_to_save.extend(spans)
                    valid_trace_count += 1
                
                if rows_to_save:
                    writer.writerows(rows_to_save)
                    total_spans += len(rows_to_save)
                print(f"   进度: 已获取 {valid_trace_count}/{limit} 条纯净 Trace...", end='\r')

        logger.info(f"\n✅ [Trace] 已保存 {valid_trace_count} 条 Trace。")
        logger.info(f"🔍 [Analysis] Trace 中共发现了 {len(active_nodes_in_trace)} 个独立节点 (Instance IDs)。")
        return active_nodes_in_trace

    def fetch_metrics(self, start_ts, end_ts, target_nodes=None):
        """步骤 2: 定向获取指标 (Targeted Metric Fetching)"""
        logger.info(f"🚀 [Metric] 开始获取节点指标 (目标节点数: {len(target_nodes) if target_nodes else 'ALL'})...")
        
        # 1. 获取 Entity 信息 (为了拿到 entity_id)
        # 即使指定了 target_nodes，我们也要去 umodel 查一下它们的 entity_id
        entity_query = {
            "domain": "acs",
            "entity_set_name": "acs.ecs.instance",
            "from_time": start_ts,
            "to_time": end_ts,
            "limit": 200 # 如果节点超过200，可能需要分批，但目前够用
        }
        nodes_res = umodel_get_entities.invoke(entity_query)
        
        active_entities = []
        found_node_names = set()
        
        if nodes_res and nodes_res.data:
            for node in nodes_res.data:
                instance_id = node.get('instance_id')
                # 如果指定了目标，只处理目标列表里的
                if target_nodes and instance_id not in target_nodes:
                    continue
                active_entities.append(node)
                found_node_names.add(instance_id)
        
        # [关键] 检查幽灵节点
        if target_nodes:
            ghost_nodes = target_nodes - found_node_names
            if ghost_nodes:
                logger.warning(f"⚠️  发现 {len(ghost_nodes)} 个幽灵节点 (Trace中有, 但云监控未发现): {list(ghost_nodes)[:5]}...")
                logger.warning("    -> 这些节点可能已释放或未安装插件，将无法获取 Golden Metrics。")
        
        logger.info(f"   将尝试拉取 {len(active_entities)} 个有效节点的指标...")
        
        filename = f"normal_metrics_{self.args.file_name}.csv"
        csv_path = os.path.join(self.output_dir, filename)
        headers = ['problem_id', 'fault_type', 'instance_id', 'timestamp'] + sorted(TARGET_METRICS)
        
        rows_to_write = []
        CHUNK_SIZE = 1800

        for i, node in enumerate(active_entities):
            instance_id = node.get('instance_id')
            entity_id = node.get('__entity_id__')
            if not entity_id: continue

            node_data = {} 
            current_chunk_start = start_ts
            
            while current_chunk_start < end_ts:
                current_chunk_end = min(current_chunk_start + CHUNK_SIZE, end_ts)
                chunk_found_metrics = set()

                # --- 策略 A: Golden Metrics ---
                try:
                    gm_res = umodel_get_golden_metrics.invoke({
                        "domain": "acs",
                        "entity_set_name": "acs.ecs.instance",
                        "entity_ids": [entity_id],
                        "from_time": current_chunk_start,
                        "to_time": current_chunk_end
                    })
                    if gm_res and gm_res.data:
                        for item in gm_res.data:
                            m_name = item.get('metric')
                            if m_name in TARGET_METRICS:
                                chunk_found_metrics.add(m_name)
                                import ast
                                vals = ast.literal_eval(item.get('__value__', '[]'))
                                tss = ast.literal_eval(item.get('__ts__', '[]'))
                                for v, t in zip(vals, tss):
                                    t_int = int(t)
                                    t_ns = t_int * 1000000 if t_int < 1e14 else t_int
                                    if t_ns not in node_data: node_data[t_ns] = {}
                                    node_data[t_ns][m_name] = v
                except: pass

                # --- 策略 B: CMS 补缺 ---
                missing = [m for m in TARGET_METRICS if m not in chunk_found_metrics]
                if missing:
                    for m in missing:
                        # CMS 查询依赖于 entity (可能查不到幽灵节点)
                        query = f".entity_set with(domain='acs', name='acs.ecs.instance', ids=['{entity_id}']) | entity-call get_metric('{m}')"
                        try:
                            res = execute_cms_query(self.cms_client, WORKSPACE_NAME, query, current_chunk_start, current_chunk_end)
                            if res and res.data:
                                for r in res.data:
                                    v = r.get('value') or r.get(m)
                                    t = r.get('timestamp') or r.get('ts')
                                    if v is not None and t is not None:
                                        t_int = int(t)
                                        t_ns = t_int * 1000000 if t_int < 1e14 else t_int
                                        if t_ns not in node_data: node_data[t_ns] = {}
                                        node_data[t_ns][m] = v
                        except: pass
                
                current_chunk_start = current_chunk_end
            
            # --- 重采样与填充 ---
            if self.args.interval and self.args.interval > 0 and node_data:
                try:
                    df = pd.DataFrame.from_dict(node_data, orient='index')
                    df.index = pd.to_datetime(df.index, unit='ns')
                    for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df_resampled = df.resample(f'{self.args.interval}s').mean().ffill().fillna(0.0)
                    
                    new_timestamps = df_resampled.index.astype(np.int64).tolist()
                    for idx, ts_val in enumerate(new_timestamps):
                        row_vals = df_resampled.iloc[idx].to_dict()
                        row = {
                            'problem_id': 'normal_000',
                            'fault_type': 'normal',
                            'instance_id': instance_id,
                            'timestamp': ts_val
                        }
                        for m in TARGET_METRICS: row[m] = row_vals.get(m, 0.0)
                        rows_to_write.append(row)
                except Exception as e:
                    logger.error(f"   [Node {instance_id}] 处理失败: {e}")

            if (i+1) % 5 == 0: print(f"   已处理 {i+1}/{len(active_entities)} 个节点...", end='\r')

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows_to_write)
        
        logger.info(f"\n✅ [Metric] 已保存 {len(rows_to_write)} 条指标数据至 {csv_path}")

    def run(self):
        s_ts, e_ts = self.determine_time_window()
        
        # [修改] 1. 先获取 Trace，并拿到节点名单
        active_nodes = self.fetch_traces(s_ts, e_ts)
        
        # [修改] 2. 带着名单去抓指标 (扩宽一点时间窗口以免边缘丢失)
        self.fetch_metrics(s_ts - 180, e_ts, target_nodes=active_nodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset/b_gt.csv", help="故障列表路径")
    parser.add_argument("--output-dir", default="data/NormalData", help="输出目录")
    parser.add_argument("--trace-limit", type=int, default=2000, help="获取多少条正常 Trace")
    parser.add_argument("--interval", type=int, default=30, help="指标重采样间隔(秒)")
    
    # [新增] 参数
    parser.add_argument("--window-hours", type=float, default=12.0, help="获取故障前多少小时的数据")
    parser.add_argument("--file-name", type=str, default="1e5_30s_demo", help="输出文件名后缀 (例如 '_v1')")
    
    args = parser.parse_args()

    fetcher = NormalDataFetcher(args)
    fetcher.run()