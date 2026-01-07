# -*- coding: utf-8 -*-
"""
节点级故障 Trace 提取工具 (二次校验版)
核心逻辑：
1. 初筛：利用 SQL Like 语句从 SLS 拉取潜在 Trace。
2. 二次校验：在本地检查每条 Trace 的所有 Span。
   - 规则：Trace 中至少有一个 Span 的 NodeName 包含目标 IP 或等于 ECS ID。
   - 动作：不满足则丢弃。
3. 统计：实时输出丢弃数量。
"""

import os
import sys
import json
import csv
import time
import argparse
import logging
from collections import defaultdict
from datetime import datetime, timedelta

import config

# ================= 🔧 鉴权配置 =================
os.environ["ALIBABA_CLOUD_ROLE_SESSION_NAME"] = "node-fault-verifier"

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from aliyun.log import LogClient, GetLogsRequest
    from alibabacloud_sts20150401.client import Client as StsClient
    from alibabacloud_sts20150401 import models as sts_models
    from alibabacloud_tea_openapi import models as open_api_models
    from tools.paas_entity_tools import umodel_get_entities
except ImportError as e:
    print(f"❌ 依赖导入失败: {e}")
    sys.exit(1)

# ================= 🔧 基础配置 =================
PROJECT_NAME = config.SLS_PROJECT_NAME
LOGSTORE_NAME = config.SLS_LOGSTORE_NAME
REGION = config.SLS_REGION
OUTPUT_FILENAME = "all_fault_traces.csv"

CSV_HEADERS = [
    'TraceID', 'SpanId', 'ParentID', 
    'ServiceName', 'NodeName', 'PodName', 
    'URL', 'SpanKind', 
    'StartTimeMs', 'EndTimeMs', 'DurationMs',
    'StatusCode', 'HttpStatusCode', 
    'fault_type', 'fault_instance', 'instance_type', 'problem_id'
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= 🛠️ 工具类定义 =================

class ECSInfoProvider:
    def get_instance_ips(self, instance_id, start_ts, end_ts):
        query_start = start_ts - 3600
        query_end = end_ts
        query = {
            "domain": "acs",
            "entity_set_name": "acs.ecs.instance",
            "from_time": query_start,
            "to_time": query_end,
            "limit": 500
        }
        ips = set()
        try:
            res = umodel_get_entities.invoke(query)
            if not res or not res.data: return []
            for node in res.data:
                if node.get('instance_id') == instance_id:
                    raw_ip = node.get('instance_ip') or node.get('privateIpAddress')
                    if isinstance(raw_ip, list):
                        for ip in raw_ip: ips.add(ip)
                    elif isinstance(raw_ip, str):
                        for ip in raw_ip.split(','):
                            if ip.strip(): ips.add(ip.strip())
        except Exception as e:
            logger.error(f"   ❌ ECS Query Error: {e}")
        return list(ips)

class AutoRefreshSLSClient:
    def __init__(self, region=REGION):
        self.region = region
        self.client = None
        self._refresh_client()
    
    def _refresh_client(self):
        config = open_api_models.Config(
            access_key_id=os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
            access_key_secret=os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
            endpoint=f'sts.{self.region}.aliyuncs.com'
        )
        sts_client = StsClient(config)
        resp = sts_client.assume_role(sts_models.AssumeRoleRequest(
            role_arn=os.environ["ALIBABA_CLOUD_ROLE_ARN"],
            role_session_name=os.environ["ALIBABA_CLOUD_ROLE_SESSION_NAME"],
            duration_seconds=3600
        ))
        creds = resp.body.credentials
        self.client = LogClient(
            endpoint=f"{self.region}.log.aliyuncs.com",
            accessKeyId=creds.access_key_id,
            accessKey=creds.access_key_secret,
            securityToken=creds.security_token
        )
    
    def get_logs(self, req):
        try:
            return self.client.get_logs(req)
        except Exception:
            self._refresh_client()
            return self.client.get_logs(req)

def safe_json_load(text):
    if not text: return {}
    try: return json.loads(text)
    except: return {}

class TraceExtractor:
    def __init__(self):
        self.client = AutoRefreshSLSClient()

    def find_trace_ids(self, query, start_ts, end_ts, limit):
        logger.info(f"      🔍 Query: {query}")
        trace_ids = set()
        offset = 0
        while len(trace_ids) < limit:
            req = GetLogsRequest(PROJECT_NAME, LOGSTORE_NAME, query=query, fromTime=start_ts, toTime=end_ts, line=100, offset=offset)
            res = self.client.get_logs(req)
            if not res or not res.get_logs(): break
            
            logs = res.get_logs()
            for log in logs:
                tid = log.get_contents().get('traceId')
                if tid: trace_ids.add(tid)
            
            offset += len(logs)
            if len(logs) < 100: break
            
        # logger.info(f"      ✅ Found IDs: {len(trace_ids)}") # 移到外层统一打印
        return list(trace_ids)[:limit]

    def fetch_and_verify_traces(self, trace_ids, start_ts, end_ts, meta, writer, existing_ids, target_ips):
        """
        拉取详情 -> 本地校验 -> 写入 CSV
        返回: (saved_count, discarded_count)
        """
        # 注意：这里传入的 trace_ids 已经是剔除过已存在 ID 的列表了（在 process_single_row 里处理）
        new_ids = trace_ids 
        if not new_ids: return 0, 0
        
        target_instance = meta['fault_instance']
        
        saved_count = 0
        discarded_count = 0
        batch_size = 20
        
        for i in range(0, len(new_ids), batch_size):
            batch = new_ids[i:i+batch_size]
            or_query = " OR ".join([f'traceId: "{tid}"' for tid in batch])
            
            batch_buffer = defaultdict(list)
            
            offset = 0
            while True:
                req = GetLogsRequest(PROJECT_NAME, LOGSTORE_NAME, query=or_query, fromTime=start_ts, toTime=end_ts, line=100, offset=offset)
                res = self.client.get_logs(req)
                if not res or not res.get_logs(): break
                
                logs = res.get_logs()
                for log in logs:
                    d = log.get_contents()
                    tid = d.get('traceId')
                    if not tid: continue

                    res_obj = safe_json_load(d.get('resources', '{}'))
                    attr_obj = safe_json_load(d.get('attributes', '{}'))
                    
                    node_name = res_obj.get('host.id') or res_obj.get('k8s.node.name', '')
                    
                    try:
                        s_ms = int(d.get('startTime', 0)) / 1e6
                        d_ms = int(d.get('duration', 0)) / 1e6
                    except: s_ms, d_ms = 0, 0

                    row = {
                        'TraceID': tid,
                        'SpanId': d.get('spanId', ''),
                        'ParentID': d.get('parentSpanId', ''),
                        'ServiceName': d.get('serviceName', ''),
                        'NodeName': node_name,  
                        'PodName': res_obj.get('k8s.pod.name', ''),
                        'URL': d.get('spanName', ''),
                        'SpanKind': d.get('kind', ''),
                        'StartTimeMs': f"{s_ms:.3f}",
                        'EndTimeMs': f"{s_ms + d_ms:.3f}",
                        'DurationMs': f"{d_ms:.3f}",
                        'StatusCode': d.get('statusCode', ''),
                        'HttpStatusCode': str(attr_obj.get('http.status_code') or attr_obj.get('rpc.grpc.status_code', '')),
                        'fault_type': meta['fault_type'],
                        'fault_instance': meta['fault_instance'],
                        'instance_type': meta['instance_type'],
                        'problem_id': meta['problem_id']
                    }
                    batch_buffer[tid].append(row)
                
                offset += len(logs)
                if len(logs) < 100: break
            
            # 本地二次校验
            for tid in batch:
                spans = batch_buffer.get(tid, [])
                if not spans: continue

                is_valid = False
                for span in spans:
                    n_name = span['NodeName']
                    if not n_name: continue
                    
                    if n_name == target_instance:
                        is_valid = True
                        break
                    
                    for ip in target_ips:
                        if ip in n_name:
                            is_valid = True
                            break
                    if is_valid: break
                
                if is_valid:
                    writer.writerows(spans)
                    existing_ids.add(tid)
                    saved_count += 1
                else:
                    discarded_count += 1
            
            print(f"      ⏳ Batch Processed: {min(i+batch_size, len(new_ids))}/{len(new_ids)} (Discarded so far: {discarded_count})...", end='\r')
            
        print("") 
        return saved_count, discarded_count

class NodeFaultProcessor:
    def __init__(self, args):
        self.args = args
        self.extractor = TraceExtractor()
        self.ecs_provider = ECSInfoProvider()
        self.existing_ids = set()
        self.out_path = os.path.join(args.output_dir, OUTPUT_FILENAME)
        
        if os.path.exists(self.out_path):
            with open(self.out_path, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row.get('TraceID'): self.existing_ids.add(row['TraceID'])
            logger.info(f"📚 Loaded {len(self.existing_ids)} existing TraceIDs from local file")

    def process(self):
        file_exists = os.path.exists(self.out_path)
        with open(self.out_path, 'a', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=CSV_HEADERS)
            if not file_exists:
                writer.writeheader()
            
            with open(self.args.csv, 'r', encoding='utf-8') as f_in:
                rows = list(csv.DictReader(f_in))
                if self.args.range:
                    s, e = map(int, self.args.range.split(','))
                    rows = [r for r in rows if s <= int(r['problem_id']) <= e]
                
                total_discarded = 0
                for row in rows:
                    total_discarded += self._process_single_row(row, writer)
                
                logger.info(f"\n🛑 All Done. Total Traces Discarded: {total_discarded}")

    def _process_single_row(self, row, writer):
        pid = row['problem_id']
        i_type = row['instance_type']
        instance = row['instance']
        
        if i_type != 'node': return 0

        logger.info(f"\n🚀 [Problem {pid}] Node Fault | Instance: {instance}")
        try:
            s_dt = datetime.strptime(row['start_time'], "%Y-%m-%d %H:%M:%S")
            e_dt = datetime.strptime(row['end_time'], "%Y-%m-%d %H:%M:%S")
            s_ts = int((s_dt + timedelta(seconds=self.args.buffer)).timestamp())
            e_ts = int((e_dt - timedelta(seconds=self.args.buffer)).timestamp())
            fetch_s_ts = int(s_dt.timestamp())
            fetch_e_ts = int(e_dt.timestamp())
        except:
            logger.error("   ❌ Time Format Error")
            return 0

        # 1. 获取 IPs
        logger.info(f"   🔎 Fetching IPs for validation...")
        ips = self.ecs_provider.get_instance_ips(instance, s_ts, e_ts)
        
        # 2. 构造查询
        conditions = [f"resources like '%{instance}%'"]
        if ips:
            logger.info(f"   ✅ Valid IPs for Check: {ips}")
            for ip in ips:
                node_target = f"cn-qingdao.{ip}"
                kv_compact = f'\\"k8s.node.name\\":\\"{node_target}\\"'
                conditions.append(f"resources like '%{kv_compact}%'")
                conditions.append(f"resources like '%{node_target}%'")
        else:
            logger.warning(f"   ⚠️ No IPs found. Strict check will only use InstanceID.")

        query = "* | where " + " OR ".join(conditions)

        # 3. 获取 TraceID List
        tids = self.extractor.find_trace_ids(query, s_ts, e_ts, self.args.limit)
        
        # ==========================================
        # 🔥 新增：清晰的统计日志 🔥
        # ==========================================
        num_found = len(tids)
        # 计算哪些是新的
        new_tids = [t for t in tids if t not in self.existing_ids]
        num_new = len(new_tids)
        num_existing = num_found - num_new
        
        logger.info(f"   📊 统计: 云端命中 {num_found} 条 | 本地已存 {num_existing} 条 | ⬇️ 待下载 {num_new} 条")
        # ==========================================

        meta = {
            'fault_type': row['fault_type'],
            'fault_instance': instance,
            'instance_type': i_type,
            'problem_id': pid
        }
        
        if new_tids:
            # 直接传入 new_tids，避免函数内再次重复计算
            saved, discarded = self.extractor.fetch_and_verify_traces(
                new_tids, fetch_s_ts, fetch_e_ts, meta, writer, self.existing_ids, ips
            )
            logger.info(f"      📉 本批次结果: 入库 {saved} 条, 校验丢弃 {discarded} 条")
            return discarded
        else:
            logger.info(f"      ✨ 所有数据已存在，跳过下载")
            return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset/b_gt.csv", help="b_gt.csv path")
    parser.add_argument("--output-dir", default="data/NodeFault", help="output directory")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--buffer", type=int, default=60)
    parser.add_argument("--range", help="Problem ID range (e.g. 1,100)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    NodeFaultProcessor(args).process()

if __name__ == "__main__":
    main()