import json
import csv
import os

def generate_csv(b_gt_path, questions_path, output_csv_path):
    # 1. 读取 B榜题目.jsonl，建立 problem_id 到时间范围的映射
    problem_time_map = {}
    print(f"正在读取题目文件: {questions_path} ...")
    try:
        with open(questions_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    problem_id = data.get('problem_id')
                    time_range = data.get('time_range')
                    
                    if problem_id and time_range:
                        # 分割时间范围字符串
                        # 格式示例: "2025-09-16 23:20:33 ~ 2025-09-16 23:30:33"
                        times = time_range.split(' ~ ')
                        if len(times) == 2:
                            problem_time_map[problem_id] = (times[0], times[1])
                except json.JSONDecodeError:
                    print(f"警告: 无法解析题目文件中的行: {line[:50]}...")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {questions_path}")
        return

    # 2. 读取 b_gt.jsonl，处理数据并准备写入 CSV
    rows = []
    print(f"正在读取 Ground Truth 文件: {b_gt_path} ...")
    try:
        with open(b_gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    problem_id = data.get('problem_id')
                    root_causes = data.get('root_causes', [])
                    
                    # 获取时间信息，如果没有则留空
                    start_time, end_time = problem_time_map.get(problem_id, ("", ""))
                    
                    for rc in root_causes:
                        # 解析 root_cause 字符串 (格式: instance.fault_type)
                        # 使用 rsplit 从右边分割一次，以防 instance 名称中包含点号
                        if '.' in rc:
                            instance, fault_type = rc.rsplit('.', 1)
                        else:
                            instance = rc
                            fault_type = "unknown"
                        
                        # 判断 instance_type
                        if instance.startswith("i-m5e"):
                            instance_type = "node"
                        else:
                            instance_type = "service"

                        if instance_type == "node":
                            fault_type = f"node {fault_type}"
                        
                        rows.append({
                            "problem_id": problem_id,
                            "start_time": start_time,
                            "end_time": end_time,
                            "instance_type": instance_type,
                            "instance": instance,
                            "fault_type": fault_type
                        })
                except json.JSONDecodeError:
                    print(f"警告: 无法解析 GT 文件中的行: {line[:50]}...")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {b_gt_path}")
        return

    # 3. 写入 CSV 文件
    headers = ["problem_id", "start_time", "end_time", "instance_type", "instance", "fault_type"]
    print(f"正在写入结果到: {output_csv_path} ...")
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"成功! 已生成 {len(rows)} 条记录。")
    except IOError as e:
        print(f"写入 CSV 失败: {e}")

if __name__ == "__main__":
    # 配置输入文件路径
    # 请确保这两个路径与您实际的文件位置一致
    B_GT_FILE = 'dataset/b_ground_truth.jsonl'
    QUESTIONS_FILE = 'dataset/B榜题目.jsonl'  # 或者 'B榜题目.jsonl'
    OUTPUT_FILE = 'dataset/b_gt.csv'
    
    # 如果文件在当前目录下，可以取消注释下面这行来测试
    # QUESTIONS_FILE = 'B榜题目.jsonl' 

    generate_csv(B_GT_FILE, QUESTIONS_FILE, OUTPUT_FILE)