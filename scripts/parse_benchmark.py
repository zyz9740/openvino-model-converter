#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import sys
import io
# 强制标准输出使用UTF-8编码，避免Windows控制台中文乱码
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

def parse_benchmark_log(log_content):
    # 定义各个字段的正则匹配规则
    patterns = {
        "openvino_version": r"Build \.+ (.+)",
        "input_shape": r"Model inputs:\s+\[ INFO \] x \(node: x\) : f32 / \[\.\.\.\] / (\[.+\])",
        "infer_precision": r"INFERENCE_PRECISION_HINT: (.+)",
        "performance_hint": r"PERFORMANCE_HINT: (.+)",
        "execution_mode": r"EXECUTION_MODE_HINT: (.+)",
        "execution_device": r"EXECUTION_DEVICES: (.+)",
        "latency_median": r"Median: (.+)",
        "throughput": r"Throughput: (.+)"
    }
    
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, log_content)
        if match:
            result[key] = match.group(1).strip()
    
    # 格式化输出要求的内容
    output = []
    output.append("1. openvino 版本信息")
    output.append("OpenVINO:")
    output.append(f"[ INFO ] Build ................................. {result.get('openvino_version', 'N/A')}")
    output.append("")
    output.append("2. 模型输入尺寸")
    output.append("Model inputs:")
    output.append(f"[ INFO ] x (node: x) : f32 / [...] / {result.get('input_shape', 'N/A')}")
    output.append("")
    output.append(f"3. 推理精度：INFERENCE_PRECISION_HINT: {result.get('infer_precision', 'N/A')}")
    output.append(f"4. PERFORMANCE_HINT: {result.get('performance_hint', 'N/A')}")
    output.append(f"5. EXECUTION_MODE_HINT: {result.get('execution_mode', 'N/A')}")
    output.append(f"6. EXECUTION_DEVICES: {result.get('execution_device', 'N/A')}")
    output.append("")
    output.append("7. 延迟中位数：Latency:")
    output.append(f"[ INFO ] Median: {result.get('latency_median', 'N/A')}")
    output.append(f"8. FPS: Throughput: {result.get('throughput', 'N/A')}")
    
    return "\n".join(output)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  1. 从日志文件读取: python parse_benchmark.py <benchmark_log.txt>")
        print("  2. 从管道读取: benchmark_app ... | python parse_benchmark.py")
        sys.exit(1)
    
    if sys.argv[1] == "-":
        # 从标准输入读取
        log_content = sys.stdin.read()
    else:
        # 从文件读取
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            log_content = f.read()
    
    print(parse_benchmark_log(log_content))