# -*- coding: utf-8 -*-
"""
批量 Trace 提取脚本 (服务级故障专用) - 修正鉴权 & 增加统计版
功能：读取 b_gt.csv -> 过滤服务故障 -> 时间缓冲 -> 统计命中数 -> 提取 TraceID -> 拉取全量链路 -> 保存 CSV
"""

import os
import json
import csv
import time
import argparse
from datetime import datetime, timedelta
from aliyun.log import LogClient, GetLogsRequest
from alibabacloud_sts20150401.client import Client as StsClient
from alibabacloud_sts20150401 import models as sts_models
from alibabacloud_tea_openapi import models as open_api_models
from Tea.exceptions import TeaException

import config

# ================= 🔧 基础配置 =================
PROJECT_NAME = config.SLS_PROJECT_NAME
LOGSTORE_NAME = config.SLS_LOGSTORE_NAME
REGION = config.SLS_REGION

# 输出 CSV 表头
CSV_HEADERS = [
    'TraceID', 'SpanId', 'ParentID', 
    'ServiceName', 'NodeName', 'PodName', 
    'URL', 'SpanKind', 
    'StartTimeMs', 'EndTimeMs', 'DurationMs',
    'StatusCode', 'HttpStatusCode', 
    'fault_type', 'fault_instance'
]

# ===============================================

def get_sts_credentials(region: str = "cn-qingdao"):
    """获取 STS 临时凭证"""
    access_key_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
    access_key_secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
    role_arn = os.getenv('ALIBABA_CLOUD_ROLE_ARN')
    session_name = 'batch-trace-extractor'
    
    config = open_api_models.Config(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        endpoint=f'sts.{region}.aliyuncs.com'
    )
    sts_client = StsClient(config)
    
    try:
        response = sts_client.assume_role(sts_models.AssumeRoleRequest(
            role_arn=role_arn,
            role_session_name=session_name,
            duration_seconds=3600
        ))
        return response.body.credentials
    except TeaException as e:
        print(f"❌ 获取 STS 凭证失败: {e.message}")
        raise

class AutoRefreshSLSClient:
    """自动刷新 Token 的 SLS 客户端"""
    def __init__(self, region: str = "cn-qingdao"):
        self.region = region
        self.sls_endpoint = f"{region}.log.aliyuncs.com"
        self.client = None
        self._refresh_client()
    
    def _refresh_client(self):
        creds = get_sts_credentials(self.region)
        self.client = LogClient(
            endpoint=self.sls_endpoint,
            accessKeyId=creds.access_key_id,
            accessKey=creds.access_key_secret,
            securityToken=creds.security_token
        )
    
    def get_logs(self, request):
        try:
            return self.client.get_logs(request)
        except Exception as e:
            if "Unauthorized" in str(e) or "expired" in str(e).lower():
                print(f"⚠️ Token 过期，正在刷新...")
                self._refresh_client()
                return self.client.get_logs(request)
            raise e

def safe_json_load(text):
    if not text: return {}
    try: return json.loads(text)
    except: return {}

class TraceExtractor:
    def __init__(self):
        self.client = AutoRefreshSLSClient(REGION)

    def _count_total_traces(self, query, start_ts, end_ts):
        """辅助方法：统计总数"""
        # 构造聚合查询
        count_query = f"{query} | select count(distinct traceId) as total"
        try:
            req = GetLogsRequest(
                project=PROJECT_NAME, 
                logstore=LOGSTORE_NAME, 
                query=count_query, 
                fromTime=start_ts, 
                toTime=end_ts
            )
            res = self.client.get_logs(req)
            if res and res.get_logs():
                return int(res.get_logs()[0].get_contents().get('total', 0))
            return 0
        except Exception as e:
            print(f"      ⚠️ 统计总数失败: {e}")
            return -1

    def find_trace_ids(self, query, start_ts, end_ts, limit):
        """阶段一：查找符合条件的 TraceID"""
        print(f"   🔍 检索 TraceID...")
        print(f"      查询语句: {query}")
        
        # 1. 先统计总量
        total_count = self._count_total_traces(query, start_ts, end_ts)
        print(f"      📊 该时段符合条件的 Trace 总数: {total_count}")
        
        if total_count == 0:
            return []

        # 2. 拉取 ID (受 Limit 限制)
        print(f"      🔄 正在提取 TraceID (设定上限: {limit})...")
        trace_ids = set()
        offset = 0
        
        while len(trace_ids) < limit:
            req = GetLogsRequest(
                project=PROJECT_NAME, 
                logstore=LOGSTORE_NAME, 
                query=query, 
                fromTime=start_ts, 
                toTime=end_ts, 
                line=100, 
                offset=offset
            )
            res = self.client.get_logs(req)
            
            if not res or not res.get_logs():
                break
                
            logs = res.get_logs()
            for log in logs:
                tid = log.get_contents().get('traceId')
                if tid:
                    trace_ids.add(tid)
            
            offset += len(logs)
            # 如果单次获取不足 100 条，说明翻页结束
            if len(logs) < 100: 
                break
        
        final_ids = list(trace_ids)[:limit]
        print(f"      ✅ 实际提取 TraceID: {len(final_ids)} (Coverage: {len(final_ids)}/{total_count})")
        return final_ids

    def fetch_and_save_traces(self, trace_ids, start_ts, end_ts, output_path, fault_info):
        """阶段二：拉取全量链路并保存"""
        if not trace_ids:
            print("      ⚠️ 无 TraceID，跳过导出")
            return

        print(f"   📦 拉取全量 Span 并保存至: {output_path}")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        total_spans = 0
        batch_size = 20 # 每次处理 20 个 TraceID，防止 Query 过长
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            
            for i in range(0, len(trace_ids), batch_size):
                batch = trace_ids[i : i + batch_size]
                # 构造 OR 查询反查所有 Span
                # 注意：这里我们只通过 traceId 过滤，不加任何其他条件，确保拿回完整的 Trace
                or_query = " OR ".join([f'traceId: "{tid}"' for tid in batch])
                
                offset = 0
                while True:
                    req = GetLogsRequest(
                        project=PROJECT_NAME, 
                        logstore=LOGSTORE_NAME, 
                        query=or_query, 
                        fromTime=start_ts, 
                        toTime=end_ts, 
                        line=100, 
                        offset=offset
                    )
                    res = self.client.get_logs(req)
                    
                    if not res or not res.get_logs():
                        break
                    
                    logs = res.get_logs()
                    rows = []
                    
                    for log in logs:
                        data = log.get_contents()
                        res_obj = safe_json_load(data.get('resources', '{}'))
                        attr_obj = safe_json_load(data.get('attributes', '{}'))
                        
                        # 时间计算
                        try:
                            s_ns = int(data.get('startTime', 0))
                            d_ns = int(data.get('duration', 0))
                            s_ms = s_ns / 1e6
                            d_ms = d_ns / 1e6
                            e_ms = s_ms + d_ms
                        except:
                            s_ms, d_ms, e_ms = 0, 0, 0

                        # 填充行数据
                        row = {
                            'TraceID': data.get('traceId', ''),
                            'SpanId': data.get('spanId', ''),
                            'ParentID': data.get('parentSpanId', ''),
                            'ServiceName': data.get('serviceName', ''),
                            'NodeName': res_obj.get('host.id') or res_obj.get('k8s.node.name', ''),
                            'PodName': res_obj.get('k8s.pod.name', ''),
                            'URL': data.get('spanName', ''),
                            'SpanKind': data.get('kind', ''),
                            'StartTimeMs': f"{s_ms:.3f}",
                            'EndTimeMs': f"{e_ms:.3f}",
                            'DurationMs': f"{d_ms:.3f}",
                            'StatusCode': data.get('statusCode', ''),
                            'HttpStatusCode': str(attr_obj.get('http.status_code') or attr_obj.get('rpc.grpc.status_code', '')),
                            'fault_type': fault_info['fault_type'],       
                            'fault_instance': fault_info['fault_instance'] 
                        }
                        rows.append(row)
                    
                    writer.writerows(rows)
                    count = len(logs)
                    total_spans += count
                    offset += count
                    if count < 100: break
                
                print(f"      进度: {min(i+batch_size, len(trace_ids))}/{len(trace_ids)} Traces...", end='\r')
        
        print(f"\n      ✅ 完成. 共导出 {total_spans} Spans.")

class BatchProcessor:
    def __init__(self, args):
        self.args = args
        self.extractor = TraceExtractor()

    def process_row(self, row):
        pid = row['problem_id']
        p_type = row['instance_type']
        instance = row['instance']
        
        # 1. 只处理服务级故障
        if p_type != 'service':
            # print(f"⏭️  [Problem {pid}] 跳过 (类型: {p_type}, 非 Service)")
            return

        print(f"\n🚀 [Problem {pid}] 处理中... (服务: {instance}, 故障: {row['fault_type']})")
        
        # 2. 计算时间窗口 (带缓冲)
        try:
            start_dt = datetime.strptime(row['start_time'], "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(row['end_time'], "%Y-%m-%d %H:%M:%S")
            
            # 缓冲逻辑：首尾各去掉 buffer 秒
            search_start = start_dt + timedelta(seconds=self.args.buffer)
            search_end = end_dt - timedelta(seconds=self.args.buffer)
            
            if search_end <= search_start:
                print("   ⚠️ 缓冲后时间窗口无效，使用原始时间")
                search_start, search_end = start_dt, end_dt
            
            print(f"   🕒 原始时间: {row['start_time']} ~ {row['end_time']}")
            print(f"   🕒 缓冲查询: {search_start} ~ {search_end} (-{self.args.buffer}s)")
            
            s_ts = int(search_start.timestamp())
            e_ts = int(search_end.timestamp())
            
            # 为了反查完整链路，Phase 2 的拉取需要覆盖稍大的范围
            fetch_start_ts = int(start_dt.timestamp())
            fetch_end_ts = int(end_dt.timestamp())

        except Exception as e:
            print(f"   ❌ 时间解析错误: {e}")
            return

        # 3. 构造查询 & 提取
        # 逻辑：serviceName == 故障实例名
        query = f'serviceName: "{instance}"'
        
        # 阶段一：找 TraceID
        trace_ids = self.extractor.find_trace_ids(query, s_ts, e_ts, self.args.limit)
        
        if not trace_ids:
            print("   ⚠️ 未找到相关 Trace")
            return

        # 阶段二：保存 CSV
        output_file = os.path.join(self.args.output_dir, f"problem_{pid}", "trace_fusion.csv")
        
        fault_info = {
            'fault_type': row['fault_type'],
            'fault_instance': instance
        }
        
        self.extractor.fetch_and_save_traces(trace_ids, fetch_start_ts, fetch_end_ts, output_file, fault_info)

    def run(self):
        if not os.path.exists(self.args.csv):
            print(f"❌ 找不到 CSV 文件: {self.args.csv}")
            return

        with open(self.args.csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            print(f"📋 加载任务列表: {len(rows)} 条")
            
            target_rows = []
            
            # 模式 1: 指定单个 ID
            if self.args.problem_id:
                target_rows = [r for r in rows if r['problem_id'] == self.args.problem_id]
                
            # 模式 2: 指定 ID 范围 (例如 002-005)
            elif self.args.range:
                try:
                    start_id, end_id = map(int, self.args.range.split(','))
                    target_rows = [r for r in rows if start_id <= int(r['problem_id']) <= end_id]
                except:
                    print("❌ 范围格式错误，请使用: start_id,end_id (例如: 2,5)")
                    return
            
            # 模式 3: 全部
            else:
                print("⚠️ 未指定 --problem-id 或 --range，将处理 CSV 中所有服务级故障...")
                target_rows = rows

            print(f"🎯 命中任务数: {len(target_rows)}")
            
            for row in target_rows:
                self.process_row(row)
                time.sleep(1) # 避免请求过快

def main():
    parser = argparse.ArgumentParser(description="批量 Trace 提取工具")
    parser.add_argument("--csv", default="dataset/b_gt.csv", help="b_gt.csv 路径")
    parser.add_argument("--output-dir", default="output_datasets", help="数据保存根目录")
    
    # 筛选模式
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--problem-id", help="指定单个 Problem ID (如: 002)")
    group.add_argument("--range", help="指定 ID 范围 (如: 2,5)")
    
    # 参数微调
    parser.add_argument("--limit", type=int, default=2000, help="每个故障提取 TraceID 上限")
    parser.add_argument("--buffer", type=int, default=60, help="时间窗口首尾切除秒数 (默认 60s)")
    
    args = parser.parse_args()
    
    processor = BatchProcessor(args)
    processor.run()

if __name__ == "__main__":
    main()