# -*- coding: utf-8 -*-
"""
全局 Trace 提取工具 (Single File Edition)
功能：
1. 读取 b_gt.csv，筛选所有 Service 级故障。
2. 将所有 Trace 数据汇聚写入同一个 CSV 文件。
3. 自动去重：启动时加载已有 TraceID。
4. 统计输出：按 fault_type 统计样本数量。
5. 日志记录：同时输出到控制台和日志文件。
"""

import os
import json
import csv
import time
import argparse
import logging
import collections
from datetime import datetime, timedelta
from aliyun.log import LogClient, GetLogsRequest
from alibabacloud_sts20150401.client import Client as StsClient
from alibabacloud_sts20150401 import models as sts_models
from alibabacloud_tea_openapi import models as open_api_models
from Tea.exceptions import TeaException

import config

# ================= 🔧 配置区域 =================
# 1. SLS 配置
PROJECT_NAME = config.SLS_PROJECT_NAME
LOGSTORE_NAME = config.SLS_LOGSTORE_NAME
REGION = config.SLS_REGION

# 2. 输出文件名
OUTPUT_FILENAME = "all_fault_traces.csv"
LOG_FILENAME = "trace_extraction.log"

# 3. CSV 表头
CSV_HEADERS = [
    'TraceID', 'SpanId', 'ParentID', 
    'ServiceName', 'NodeName', 'PodName', 
    'URL', 'SpanKind', 
    'StartTimeMs', 'EndTimeMs', 'DurationMs',
    'StatusCode', 'HttpStatusCode', 
    'fault_type', 'fault_instance', 'problem_id' # 新增 problem_id 方便回溯
]

# ================= 🔧 鉴权配置 =================
os.environ["ALIBABA_CLOUD_ROLE_SESSION_NAME"] = "service-fault-verifier"

# ===============================================

# 配置日志
def setup_logging():
    # 创建 Logger
    logger = logging.getLogger("TraceExtractor")
    logger.setLevel(logging.INFO)
    
    # 清除已有的 Handler 防止重复打印
    if logger.handlers:
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 1. File Handler (写入文件)
    file_handler = logging.FileHandler(LOG_FILENAME, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Console Handler (输出到屏幕)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

def get_sts_credentials(region: str = "cn-qingdao"):
    access_key_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
    access_key_secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
    role_arn = os.getenv('ALIBABA_CLOUD_ROLE_ARN')
    session_name = 'single-csv-extractor'
    
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
        logger.error(f"STS 鉴权失败: {e.message}")
        raise

class AutoRefreshSLSClient:
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
                logger.warning("Token 过期，正在自动刷新...")
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
        """统计预估总数"""
        count_query = f"{query} | select count(distinct traceId) as total"
        try:
            req = GetLogsRequest(PROJECT_NAME, LOGSTORE_NAME, query=count_query, fromTime=start_ts, toTime=end_ts)
            res = self.client.get_logs(req)
            if res and res.get_logs():
                return int(res.get_logs()[0].get_contents().get('total', 0))
            return 0
        except Exception as e:
            logger.warning(f"统计总数失败: {e}")
            return -1

    def find_trace_ids(self, query, start_ts, end_ts, limit):
        """阶段一：查找 TraceID"""
        logger.info(f"   🔍 正在检索 TraceID (Query: {query})...")
        
        # 1. 统计
        total = self._count_total_traces(query, start_ts, end_ts)
        logger.info(f"      📊 SLS 中符合条件的 Trace 总数: {total}")
        
        if total == 0:
            return []

        # 2. 拉取 ID
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
        
        final_ids = list(trace_ids)[:limit]
        logger.info(f"      ✅ 提取 TraceID 成功: {len(final_ids)} 个")
        return final_ids

    def fetch_full_traces(self, trace_ids, start_ts, end_ts, meta_info, csv_writer, existing_ids):
        """阶段二：拉取全量 Span 并写入 CSV"""
        if not trace_ids: return 0

        # 过滤掉已经存在于 CSV 中的 TraceID
        new_ids = [tid for tid in trace_ids if tid not in existing_ids]
        skipped_count = len(trace_ids) - len(new_ids)
        
        if skipped_count > 0:
            logger.info(f"      ⏭️  跳过 {skipped_count} 个已存在的 TraceID，剩余需下载: {len(new_ids)}")
        
        if not new_ids:
            return 0

        total_spans = 0
        batch_size = 20
        
        # 分批处理
        for i in range(0, len(new_ids), batch_size):
            batch = new_ids[i : i + batch_size]
            or_query = " OR ".join([f'traceId: "{tid}"' for tid in batch])
            
            offset = 0
            while True:
                req = GetLogsRequest(PROJECT_NAME, LOGSTORE_NAME, query=or_query, fromTime=start_ts, toTime=end_ts, line=100, offset=offset)
                res = self.client.get_logs(req)
                if not res or not res.get_logs(): break
                
                logs = res.get_logs()
                rows = []
                for log in logs:
                    data = log.get_contents()
                    res_obj = safe_json_load(data.get('resources', '{}'))
                    attr_obj = safe_json_load(data.get('attributes', '{}'))
                    
                    try:
                        s_ns = int(data.get('startTime', 0))
                        d_ns = int(data.get('duration', 0))
                        s_ms = s_ns / 1e6
                        e_ms = (s_ns + d_ms) / 1e6
                        d_ms = d_ns / 1e6
                    except: s_ms, e_ms, d_ms = 0, 0, 0

                    rows.append({
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
                        'fault_type': meta_info['fault_type'],
                        'fault_instance': meta_info['fault_instance'],
                        'problem_id': meta_info['problem_id']
                    })
                
                # 写入文件
                if rows:
                    csv_writer.writerows(rows)
                    total_spans += len(rows)
                
                offset += len(logs)
                if len(logs) < 100: break
            
            # 更新全局 ID 集合
            for tid in batch:
                existing_ids.add(tid)
                
            print(f"      ⏳ 进度: {min(i+batch_size, len(new_ids))}/{len(new_ids)} Traces 已处理...", end='\r')
        
        print("") # 换行
        logger.info(f"      📦 已保存 {len(new_ids)} 条 Trace，共 {total_spans} 个 Span。")
        return len(new_ids)

class UnifiedProcessor:
    def __init__(self, args):
        self.args = args
        self.extractor = TraceExtractor()
        self.existing_trace_ids = set()
        self.stats = collections.defaultdict(int) # 统计字典 {fault_type: trace_count}
        self.output_path = os.path.join(self.args.output_dir, OUTPUT_FILENAME)

    def _load_existing_data(self):
        """预加载已有的 TraceID 用于去重"""
        if not os.path.exists(self.output_path):
            logger.info(f"🆕 输出文件不存在，将创建新文件: {self.output_path}")
            return False # 文件不存在

        logger.info(f"📥 正在读取已有数据进行去重: {self.output_path} ...")
        try:
            with open(self.output_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    tid = row.get('TraceID')
                    if tid:
                        self.existing_trace_ids.add(tid)
                        count += 1
            logger.info(f"✅ 已加载 {len(self.existing_trace_ids)} 个历史 TraceID (共 {count} 行 Span)")
            return True # 文件存在
        except Exception as e:
            logger.error(f"读取历史文件失败: {e}")
            return False

    def process_all(self):
        if not os.path.exists(self.args.csv):
            logger.error(f"CSV 文件不存在: {self.args.csv}")
            return

        # 1. 准备文件句柄
        file_exists = self._load_existing_data()
        
        # 使用 'a' 模式追加
        with open(self.output_path, 'a', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=CSV_HEADERS)
            
            # 如果是新文件，写入表头
            if not file_exists:
                writer.writeheader()

            # 2. 读取任务列表
            with open(self.args.csv, 'r', encoding='utf-8') as f_in:
                reader = csv.DictReader(f_in)
                rows = list(reader)
            
            # 过滤范围
            target_rows = []
            if self.args.range:
                s_id, e_id = map(int, self.args.range.split(','))
                target_rows = [r for r in rows if s_id <= int(r['problem_id']) <= e_id]
            else:
                target_rows = rows # 默认跑全部

            logger.info(f"🎯 待处理任务数: {len(target_rows)}")

            # 3. 循环处理
            for row in target_rows:
                try:
                    self._process_single_row(row, writer)
                except Exception as e:
                    logger.error(f"处理 Problem {row.get('problem_id')} 时发生异常: {e}")
                
                # 间隔休息
                time.sleep(0.5)

        # 4. 最终统计
        self._print_final_stats()

    def _process_single_row(self, row, writer):
        pid = row['problem_id']
        p_type = row['instance_type']
        instance = row['instance']
        fault_type = row['fault_type']

        # 只处理 Service
        if p_type != 'service':
            return

        logger.info(f"\n🚀 [Problem {pid}] 开始处理 | Service: {instance} | Fault: {fault_type}")

        # 时间计算
        try:
            start_dt = datetime.strptime(row['start_time'], "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(row['end_time'], "%Y-%m-%d %H:%M:%S")
            
            # 缓冲
            s_ts = int((start_dt + timedelta(seconds=self.args.buffer)).timestamp())
            e_ts = int((end_dt - timedelta(seconds=self.args.buffer)).timestamp())
            
            # 全量拉取范围
            fetch_s_ts = int(start_dt.timestamp())
            fetch_e_ts = int(end_dt.timestamp())
            
            if e_ts <= s_ts:
                s_ts, e_ts = fetch_s_ts, fetch_e_ts # 缓冲无效则回退

        except Exception as e:
            logger.error(f"时间解析错误: {e}")
            return

        # 提取 ID
        query = f'serviceName: "{instance}"'
        trace_ids = self.extractor.find_trace_ids(query, s_ts, e_ts, self.args.limit)
        
        if not trace_ids:
            logger.warning("   ⚠️ 未找到相关 Trace")
            return

        # 提取全量数据并写入
        meta = {
            'fault_type': fault_type,
            'fault_instance': instance,
            'problem_id': pid
        }
        
        count = self.extractor.fetch_full_traces(trace_ids, fetch_s_ts, fetch_e_ts, meta, writer, self.existing_trace_ids)
        
        # 更新统计 (按 fault_type)
        self.stats[fault_type] += count

    def _print_final_stats(self):
        logger.info("\n" + "="*40)
        logger.info("📊 执行完成！Fault Type 样本统计如下：")
        logger.info("="*40)
        
        total_traces = 0
        if not self.stats:
            logger.info("   (无新增数据)")
        else:
            # 按数量倒序排列
            sorted_stats = sorted(self.stats.items(), key=lambda x: x[1], reverse=True)
            for f_type, count in sorted_stats:
                logger.info(f"   🔹 {f_type:<20}: {count} Traces")
                total_traces += count
            
            logger.info("-" * 40)
            logger.info(f"   ∑ 总计新增          : {total_traces} Traces")
        
        logger.info(f"   💾 数据已保存至      : {self.output_path}")
        logger.info(f"   📝 详细日志          : {LOG_FILENAME}")
        logger.info("="*40)

def main():
    parser = argparse.ArgumentParser(description="全局 Trace 提取工具")
    parser.add_argument("--csv", default="dataset/b_gt.csv", help="b_gt.csv 路径")
    parser.add_argument("--output-dir", default="data/ServiceFault", help="输出目录")
    parser.add_argument("--limit", type=int, default=3000, help="单故障提取上限")
    parser.add_argument("--buffer", type=int, default=60, help="缓冲时间(秒)")
    parser.add_argument("--range", help="指定 Problem ID 范围 (如 2,10)")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    processor = UnifiedProcessor(args)
    processor.process_all()

if __name__ == "__main__":
    main()