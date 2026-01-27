#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1:1 还原 Debug 脚本
完全复用 2_get_normalData.py 的逻辑，仅锁定特定节点并打印中间数据。
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# === 必须复用原脚本的配置 ===
import config

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入原脚本的类（确保文件名是 2_get_normalData.py 且在同级目录）
try:
    # from get_normalData import NormalDataFetcher # 假设原文件名为 normal_data_fetcher.py，如果不是请改名
    # 如果文件名是 2_get_normalData.py，Python import 不支持数字开头，
    # 请临时将 2_get_normalData.py 重命名为 baseline_fetcher.py，或者使用下面的动态导入方式：
    import importlib.util
    spec = importlib.util.spec_from_file_location("baseline_fetcher", "scripts/2_get_normalData.py")
    baseline_fetcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline_fetcher)
    NormalDataFetcher = baseline_fetcher.NormalDataFetcher
    umodel_get_golden_metrics = baseline_fetcher.umodel_get_golden_metrics
    execute_cms_query = baseline_fetcher.execute_cms_query
    TARGET_METRICS = baseline_fetcher.TARGET_METRICS
    REGION_ID = baseline_fetcher.REGION_ID
    WORKSPACE_NAME = baseline_fetcher.WORKSPACE_NAME
except ImportError as e:
    print(f"❌ 导入原脚本失败: {e}")
    print("请确保 '2_get_normalData.py' 在当前目录下。")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DebugFetcher(NormalDataFetcher):
    def fetch_metrics(self, start_ts, end_ts):
        print(f"\n🚀 [Debug模式] 开始针对单节点进行 1:1 逻辑复现")
        print(f"   时间窗: {start_ts} -> {end_ts}")
        
        # =======================================================
        # 强制锁定目标节点 (跳过 umodel_get_entities，直接给 ID)
        # =======================================================
        target_instance_id = "i-m5ec00yjg8kxv34hyr0n"
        # 这是你日志里查出来的 entity_id，我们直接硬编码，排除 lookup 错误
        target_entity_id = "34016cd1d03b562e39370299e1e83610" 
        
        nodes = [{
            'instance_id': target_instance_id,
            '__entity_id__': target_entity_id
        }]
        
        print(f"   🔒 已锁定节点: {target_instance_id} (EntityID: {target_entity_id})")

        # =======================================================
        # 下面完全复制原脚本的逻辑，只增加了 print
        # =======================================================
        CHUNK_SIZE = 1800

        for node in nodes:
            instance_id = node.get('instance_id')
            entity_id = node.get('__entity_id__')
            
            # node_data: { timestamp_ns: { metric_name: value } }
            node_data = {} 

            current_chunk_start = start_ts
            while current_chunk_start < end_ts:
                current_chunk_end = min(current_chunk_start + CHUNK_SIZE, end_ts)
                print(f"   🔍 正在扫描分片: {current_chunk_start} ~ {current_chunk_end} ...")
                
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
                        print(f"      ✅ [GM] 接口返回了数据对象")
                        for item in gm_res.data:
                            m_name = item.get('metric')
                            if m_name in TARGET_METRICS:
                                import ast
                                vals = ast.literal_eval(item.get('__value__', '[]'))
                                tss = ast.literal_eval(item.get('__ts__', '[]'))
                                if vals:
                                    print(f"         Found {m_name}: {len(vals)} points")
                                    chunk_found_metrics.add(m_name)
                                    for v, t in zip(vals, tss):
                                        t_int = int(t)
                                        t_ns = t_int * 1000000 if t_int < 1e14 else t_int
                                        if t_ns not in node_data: node_data[t_ns] = {}
                                        node_data[t_ns][m_name] = v
                    else:
                        print(f"      ❌ [GM] 接口返回空 (或 .data 为空)")
                except Exception as e:
                    print(f"      ❌ [GM] 报错: {e}")

                # --- 策略 B: CMS 原始接口补缺 ---
                missing = [m for m in TARGET_METRICS if m not in chunk_found_metrics]
                if missing:
                    print(f"      ⚠️ [CMS] 尝试补全缺失指标: {len(missing)} 个")
                    for m in missing:
                        # 原汁原味的查询语句
                        query = f".entity_set with(domain='acs', name='acs.ecs.instance', ids=['{entity_id}']) | entity-call get_metric('{m}')"
                        try:
                            # 直接复用父类的 client
                            res = execute_cms_query(self.cms_client, WORKSPACE_NAME, query, current_chunk_start, current_chunk_end)
                            if res and res.data:
                                print(f"         ✅ [CMS] {m} 获取到 {len(res.data)} 条数据")
                                for r in res.data:
                                    v = r.get('value') or r.get(m)
                                    t = r.get('timestamp') or r.get('ts')
                                    if v is not None and t is not None:
                                        t_int = int(t)
                                        t_ns = t_int * 1000000 if t_int < 1e14 else t_int
                                        if t_ns not in node_data: node_data[t_ns] = {}
                                        node_data[t_ns][m] = v
                            else:
                                # 这里很关键：如果这里也空，那就是真没数据
                                pass 
                                # print(f"         ❌ [CMS] {m} 返回空")
                        except Exception as e:
                            print(f"         ❌ [CMS] 查询报错: {e}")
                
                current_chunk_start = current_chunk_end

            # === 打印最终抓到的原始数据摘要 ===
            print(f"\n📊 === 最终数据摘要 (Instance: {instance_id}) ===")
            if not node_data:
                print("🔴 结果: 字典为空。该节点没有任何有效指标数据。")
            else:
                print(f"🟢 结果: 捕获到了 {len(node_data)} 个时间点的数据。")
                sorted_ts = sorted(node_data.keys())
                
                # 打印前 3 条看样子
                print("   [前 3 条原始数据样例]:")
                for ts in sorted_ts[:3]:
                    readable_time = datetime.fromtimestamp(ts / 1e9).strftime('%H:%M:%S')
                    print(f"   Time: {readable_time} ({ts})")
                    for k, v in node_data[ts].items():
                        print(f"      - {k}: {v}")

if __name__ == "__main__":
    # 使用你指定的参数
    # Start: 2025-09-16 16:00:00 -> TS: 1758009600
    # End:   2025-09-16 22:00:00 -> TS: 1758031200
    
    parser = argparse.ArgumentParser()
    # 只要这几个参数就够了，反正我们在这个脚本里不真正写文件
    parser.add_argument("--output-dir", default="data/DebugOutput")
    parser.add_argument("--file-name", default="debug")
    parser.add_argument("--window-hours", type=float, default=6.0) 
    
    args = parser.parse_args()
    
    # 你的 Start/End 时间
    s_ts = 1758009600 
    e_ts = 1758031200

    fetcher = DebugFetcher(args)
    fetcher.fetch_metrics(s_ts, e_ts)