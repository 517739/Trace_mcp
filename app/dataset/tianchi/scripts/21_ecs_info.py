#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 21_ecs_info.py 改造的 ECS 映射提取器
功能：
1. 获取指定时间段内的所有 ECS 实例信息
2. 统计实例数量和 ID 数量
3. 构建 IP/Name 到 Instance ID 的映射并保存
"""

import os
import sys
import json
import time
from datetime import datetime
import config

# ================= 🔧 鉴权配置 (完全复用您的原脚本) =================
os.environ["ALIBABA_CLOUD_ROLE_SESSION_NAME"] = "my-sls-access"

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入工具 (复用原脚本依赖)
try:
    from tools.paas_entity_tools import umodel_get_entities
except ImportError:
    print("❌ 无法导入 tools，请检查项目路径")
    sys.exit(1)

def generate_ecs_mapping(start_time_str=None, end_time_str=None):
    print(f"🚀 [ECS映射模式] 开始全量扫描 ECS 节点...")

    # 1. 确定时间范围 (默认过去1小时，覆盖当前状态)
    if not start_time_str:
        now = int(time.time())
        end_timestamp = now
        start_timestamp = now - 3600 * 2 # 往前查2小时，确保不漏
    else:
        start_timestamp = int(datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S").timestamp())
        end_timestamp = int(datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S").timestamp())

    print(f"   🕒 查询时间窗口: {start_timestamp} ~ {end_timestamp}")

    # 2. 构造查询 (复用原脚本的 acs.ecs.instance 逻辑)
    query = {
        "domain": "acs",
        "entity_set_name": "acs.ecs.instance",
        "from_time": start_timestamp,
        "to_time": end_timestamp,
        "limit": 500  # 拉取限制，够大即可
    }

    try:
        # 调用接口
        res = umodel_get_entities.invoke(query)
        if not res or not res.data:
            print("   ⚠️ 未获取到 ECS 实体数据 (Result Empty)")
            return

        nodes = res.data
        print(f"   📥 原始数据拉取成功: 共 {len(nodes)} 条记录")

        # 3. 提取与映射
        unique_instance_ids = set()
        ip_map = {}   # IP -> ID
        name_map = {} # Name -> ID
        
        # 调试用：打印第一条数据看看 IP 字段长什么样
        if len(nodes) > 0:
            sample = nodes[0]
            # print(f"   🐛 [DEBUG] Sample Keys: {list(sample.keys())}")
            # print(f"   🐛 [DEBUG] Sample IP Raw: {sample.get('instance_ip', 'N/A')}")

        for node in nodes:
            # 获取核心字段
            instance_id = node.get('instance_id')
            instance_name = node.get('instance_name')
            # 兼容：原脚本中使用了 instance_ip 字段
            instance_ip_raw = node.get('instance_ip', '') 
            # 也可以尝试 privateIpAddress
            if not instance_ip_raw:
                instance_ip_raw = node.get('privateIpAddress', '')

            if not instance_id:
                continue

            # 统计 ID
            unique_instance_ids.add(instance_id)

            # 构建 Name -> ID 映射
            if instance_name:
                name_map[instance_name] = instance_id

            # 构建 IP -> ID 映射 (IP 可能是逗号分隔的字符串或列表)
            ips = []
            if isinstance(instance_ip_raw, list):
                ips = instance_ip_raw
            elif isinstance(instance_ip_raw, str):
                # 处理 "10.0.0.1,10.0.0.2" 这种格式
                ips = [ip.strip() for ip in instance_ip_raw.split(',') if ip.strip()]
            
            for ip in ips:
                ip_map[ip] = instance_id

        # 4. 输出统计结果
        print("\n" + "="*40)
        print("📊 统计结果报告")
        print("="*40)
        print(f"   🔹 查找到 ECS 实体记录数 : {len(nodes)}")
        print(f"   🔹 唯一 Instance ID 数量 : {len(unique_instance_ids)}")
        print(f"   🔹 建立 Name 映射关系数  : {len(name_map)}")
        print(f"   🔹 建立 IP 映射关系数    : {len(ip_map)}")
        print("="*40)

        # 5. 保存映射文件
        mapping_data = {
            "meta": {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_unique_ids": len(unique_instance_ids)
            },
            "ip_to_id": ip_map,
            "name_to_id": name_map,
            "all_instance_ids": list(unique_instance_ids)
        }

        # 存到 data 目录
        output_dir = os.path.join(project_root, "data")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "ecs_mapping_index.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)

        print(f"💾 映射关系已保存至: {output_file}")
        print("   (后续脚本可加载此文件，通过 IP 或 Name 查找 ID)")

    except Exception as e:
        print(f"❌ 运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

# if __name__ == "__main__":
#     # 请替换为你要分析的 Problem ID 和时间
#     # 例如 Problem 003 (假设是网络故障)
#     problem_id = "071" 
#     # 务必去 B榜题目.jsonl 确认 003 的真实时间！这里只是示例！
#     # 如果你不想查，可以用宽一点的时间范围来测试
#     start_time = "2025-09-21 15:04:00" 
#     end_time = "2025-09-21 15:26:00"
    
#     # fetch_valid_ecs_metrics(problem_id, start_time, end_time)
#     # 您可以手动传入特定故障的时间段，或者直接运行默认查最近2小时
#     generate_ecs_mapping(start_time, end_time)
#     # generate_ecs_mapping()

if __name__ == "__main__":
    # 请替换为你要分析的 Problem ID 和时间
    # 例如 Problem 003 (假设是网络故障)
    problem_id = "003" 
    # 务必去 B榜题目.jsonl 确认 003 的真实时间！这里只是示例！
    # 如果你不想查，可以用宽一点的时间范围来测试
    start_time = "2025-09-16 23:35:00" 
    end_time = "2025-09-16 23:45:00"

    generate_ecs_mapping(start_time, end_time)
    

