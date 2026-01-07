#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
正常时段数据获取工具 (Baseline Data Fetcher) - 修复版

逻辑：
1. 读取 b_gt.csv，找到最早的故障时间 (Min Start Time)。
2. 定义正常时间窗：[最早故障时间 - 2小时, 最早故障时间 - 1小时]。
3. Metric: 获取该时段所有活跃 ECS 的性能指标 (支持 --interval 重采样)。
4. Trace: 获取该时段 try_cast(statusCode as bigint) <= 1 的正常 Trace。
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

# # 2. SLS 配置
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
    from tools.constants import REGION_ID, WORKSPACE_NAME
    from aliyun.log import LogClient, GetLogsRequest
    from alibabacloud_sts20150401.client import Client as StsClient
    from alibabacloud_sts20150401 import models as sts_models
    from alibabacloud_tea_openapi import models as open_api_models
except ImportError as e:
    print(f"❌ 依赖缺失: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        with open(self.args.csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = int(datetime.strptime(row['start_time'], '%Y-%m-%d %H:%M:%S').timestamp())
                    if ts < min_ts: min_ts = ts
                except: continue
        
        # 定义：最早故障前 2小时 ~ 前 1小时
        end_time = min_ts - 3600
        start_time = end_time - 3600
        
        logger.info(f"✅ 选定正常时段: {datetime.fromtimestamp(start_time)} ~ {datetime.fromtimestamp(end_time)}")
        return start_time, end_time

    def fetch_metrics(self, start_ts, end_ts):
        """步骤 2: 获取 Metric 数据 (支持重采样)"""
        logger.info("🚀 [Metric] 开始获取正常时段的节点指标...")
        
        entity_query = {
            "domain": "acs",
            "entity_set_name": "acs.ecs.instance",
            "from_time": start_ts,
            "to_time": end_ts,
            "limit": 200
        }
        nodes_res = umodel_get_entities.invoke(entity_query)
        if not nodes_res or not nodes_res.data:
            logger.warning("   ⚠️ 未发现活跃节点")
            return

        nodes = nodes_res.data
        logger.info(f"   发现 {len(nodes)} 个活跃节点，正在拉取指标...")
        if self.args.interval:
            logger.info(f"   ⏱️ 已启用重采样: 每 {self.args.interval} 秒聚合一条数据")
        
        csv_path = os.path.join(self.output_dir, "normal_metrics.csv")
        headers = ['problem_id', 'fault_type', 'instance_id', 'timestamp'] + sorted(TARGET_METRICS)
        
        rows_to_write = []
        
        for i, node in enumerate(nodes):
            instance_id = node.get('instance_id')
            entity_id = node.get('__entity_id__')
            if not entity_id: continue

            gm_res = umodel_get_golden_metrics.invoke({
                "domain": "acs",
                "entity_set_name": "acs.ecs.instance",
                "entity_ids": [entity_id],
                "from_time": start_ts,
                "to_time": end_ts
            })

            # 收集原始数据: {timestamp(ns): {metric: val}}
            node_data = {} 
            
            if gm_res and gm_res.data:
                for item in gm_res.data:
                    m_name = item.get('metric')
                    if m_name in TARGET_METRICS:
                        import ast
                        vals = ast.literal_eval(item.get('__value__', '[]'))
                        tss = ast.literal_eval(item.get('__ts__', '[]'))
                        for v, t in zip(vals, tss):
                            if t not in node_data: node_data[t] = {}
                            node_data[t][m_name] = v
            
            # 🔥 重采样逻辑
            if self.args.interval and self.args.interval > 0 and node_data:
                try:
                    # 1. 转 DataFrame
                    df = pd.DataFrame.from_dict(node_data, orient='index')
                    # 2. 处理时间索引 (纳秒转 datetime)
                    df.index = pd.to_datetime(df.index, unit='ns')
                    # 3. 重采样 (均值)
                    df_resampled = df.resample(f'{self.args.interval}s').mean()
                    
                    # 4. 回填数据
                    node_data_resampled = {}
                    # 将时间戳转回纳秒 int64
                    new_timestamps = df_resampled.index.astype(np.int64).tolist()
                    
                    for idx, ts_val in enumerate(new_timestamps):
                        row_vals = df_resampled.iloc[idx].to_dict()
                        # 过滤掉全空的行 (可选，这里保留以维持时间连续性，但填充空缺)
                        node_data_resampled[ts_val] = {k: v for k, v in row_vals.items() if pd.notnull(v)}
                    
                    # 替换原始数据
                    node_data = node_data_resampled
                except Exception as e:
                    logger.error(f"   [Node {instance_id}] 重采样失败: {e}")

            # 整理为 CSV 行
            for ts, metrics in node_data.items():
                if not metrics: continue # 跳过空行
                row = {
                    'problem_id': 'normal_000',
                    'fault_type': 'normal',
                    'instance_id': instance_id,
                    'timestamp': ts
                }
                for m in TARGET_METRICS:
                    row[m] = metrics.get(m, "")
                rows_to_write.append(row)
                
            if (i+1) % 5 == 0: print(f"   已处理 {i+1}/{len(nodes)} 个节点...", end='\r')

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows_to_write)
        
        logger.info(f"\n✅ [Metric] 已保存 {len(rows_to_write)} 条指标数据至 {csv_path}")

    def fetch_traces(self, start_ts, end_ts):
        """步骤 3: 获取 Trace 数据 (严格过滤版 - 逻辑对齐 build_trace_dataset.py)"""
        logger.info("🚀 [Trace] 开始获取正常时段的 Trace...")
        
        # 1. 初筛: 获取包含至少一个正常Span的候选TraceID
        # (这里还是用宽泛查询，因为我们会在本地做二次严格检查)
        query = "* | where try_cast(statusCode as bigint) <= 1"
        limit = self.args.trace_limit
        
        candidate_trace_ids = set()
        offset = 0
        
        # 批量获取候选 ID
        while len(candidate_trace_ids) < limit * 1.5: # 多获取一点，因为本地过滤会丢弃一部分
            req = GetLogsRequest(PROJECT_NAME, LOGSTORE_NAME, query=query, fromTime=start_ts, toTime=end_ts, line=100, offset=offset)
            try:
                res = self.sls_client.get_logs(req)
                if not res or not res.get_logs(): break
                logs = res.get_logs()
                for log in logs:
                    candidate_trace_ids.add(log.get_contents().get('traceId'))
                offset += len(logs)
                if len(logs) < 100: break
            except Exception as e:
                logger.error(f"SLS Query Error: {e}")
                break
        
        logger.info(f"   已获取 {len(candidate_trace_ids)} 个候选 TraceID，正在进行严格过滤和拉取...")
        
        csv_path = os.path.join(self.output_dir, "normal_traces.csv")
        csv_headers = [
            'TraceID', 'SpanId', 'ParentID', 'ServiceName', 'NodeName', 'PodName', 
            'URL', 'SpanKind', 'StartTimeMs', 'EndTimeMs', 'DurationMs',
            'StatusCode', 'HttpStatusCode', 'fault_type', 'fault_instance', 'problem_id'
        ]
        
        batch_list = list(candidate_trace_ids)
        valid_trace_count = 0
        total_spans = 0
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            
            # 每次处理 20 个 TraceID
            for i in range(0, len(batch_list), 20):
                if valid_trace_count >= limit: break
                
                batch = batch_list[i:i+20]
                or_query = " OR ".join([f'traceId: "{tid}"' for tid in batch])
                
                # === 内存聚合: 将这20个Trace的所有Span先存起来 ===
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
                                # 解析 Span 数据
                                try: res_obj = json.loads(d.get('resources', '{}'))
                                except: res_obj = {}
                                try:
                                    s_ms = int(d.get('startTime', 0)) / 1e6
                                    d_ms = int(d.get('duration', 0)) / 1e6
                                except: s_ms, d_ms = 0, 0
                                
                                # 暂存原始数据对象
                                span_obj = {
                                    'TraceID': tid,
                                    'SpanId': d.get('spanId'),
                                    'ParentID': d.get('parentSpanId'),
                                    'ServiceName': d.get('serviceName'),
                                    'NodeName': res_obj.get('host.id') or res_obj.get('k8s.node.name'),
                                    'PodName': res_obj.get('k8s.pod.name'),
                                    'URL': d.get('spanName'),
                                    'SpanKind': d.get('kind'),
                                    'StartTimeMs': f"{s_ms:.3f}",
                                    'EndTimeMs': f"{s_ms + d_ms:.3f}",
                                    'DurationMs': f"{d_ms:.3f}",
                                    'StatusCode': d.get('statusCode'), # 原始状态码
                                    'HttpStatusCode': "",
                                    'fault_type': 'normal',
                                    'fault_instance': 'unknown',
                                    'problem_id': 'normal_000'
                                }
                                trace_buffer[tid].append(span_obj)
                        
                        sub_offset += len(logs)
                        if len(logs) < 100: break
                    except: break
                
                # === 严格过滤: 检查每个Trace是否真正“纯净” ===
                rows_to_save = []
                for tid, spans in trace_buffer.items():
                    if not spans: continue
                    
                    # 1. 过滤掉包含异常Span的Trace (Status > 1)
                    is_dirty = False
                    for span in spans:
                        try:
                            # 兼容处理：有些statusCode可能是空或非数字，视作0
                            sc = int(span['StatusCode']) if span['StatusCode'] and span['StatusCode'].isdigit() else 0
                            if sc > 1:
                                is_dirty = True
                                break
                        except: pass
                    
                    if is_dirty: continue # 丢弃整条 Trace
                    
                    # 2. 过滤掉过短的 Trace (可选，参考 build_trace_dataset 逻辑)
                    if len(spans) < 2: continue

                    # 3. 通过检查，加入保存队列
                    rows_to_save.extend(spans)
                    valid_trace_count += 1
                
                # 写入文件
                if rows_to_save:
                    writer.writerows(rows_to_save)
                    total_spans += len(rows_to_save)
                
                print(f"   进度: 已获取 {valid_trace_count}/{limit} 条纯净 Trace...", end='\r')

        logger.info(f"\n✅ [Trace] 已保存 {valid_trace_count} 条纯净 Trace ({total_spans} Spans) 至 {csv_path}")

    def run(self):
        s_ts, e_ts = self.determine_time_window()
        self.fetch_metrics(s_ts, e_ts)
        self.fetch_traces(s_ts, e_ts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset/b_gt.csv", help="故障列表路径")
    parser.add_argument("--output-dir", default="data/NormalData", help="输出目录")
    parser.add_argument("--trace-limit", type=int, default=70000, help="获取多少条正常 Trace")
    
    # 🔥 新增参数：默认不传则保留原始精度(约10s)，传 60 则聚合为 1分钟
    parser.add_argument("--interval", type=int, default=10, help="指标重采样间隔(秒)，例如 60")
    
    args = parser.parse_args()

    fetcher = NormalDataFetcher(args)
    fetcher.run()