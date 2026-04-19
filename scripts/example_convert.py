#!/usr/bin/env python3
"""
OpenVINO 模型转换示例脚本
修改以下配置参数即可直接运行
"""
import subprocess
import os

# ====================== 配置参数 - 请根据实际情况修改 ======================
INPUT_MODEL_PATH = "your_model.onnx"  # 输入模型路径，支持.onnx/.pb/.caffemodel等
OUTPUT_DIR = "./openvino_output"      # 输出目录
INPUT_SHAPE = [1, 3, 224, 224]        # 输入形状，动态形状可设为None
MEAN_VALUES = [123.675, 116.28, 103.53]  # 预处理均值，和训练保持一致
SCALE_VALUES = [58.395, 57.12, 57.375]   # 预处理缩放值，和训练保持一致
DATA_TYPE = "FP16"                    # 精度：FP32/FP16/INT8
MODEL_NAME = "converted_model"        # 输出模型名称
REVERSE_CHANNELS = False              # 是否反转输入通道（BGR<->RGB）
# =========================================================================

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 构建命令
    cmd = [
        "mo",
        "--input_model", INPUT_MODEL_PATH,
        "--output_dir", OUTPUT_DIR,
        "--data_type", DATA_TYPE,
        "--model_name", MODEL_NAME
    ]
    
    if INPUT_SHAPE is not None:
        shape_str = "[" + ",".join(map(str, INPUT_SHAPE)) + "]"
        cmd.extend(["--input_shape", shape_str])
    
    if MEAN_VALUES is not None:
        mean_str = "[" + ",".join(map(str, MEAN_VALUES)) + "]"
        cmd.extend(["--mean_values", mean_str])
    
    if SCALE_VALUES is not None:
        scale_str = "[" + ",".join(map(str, SCALE_VALUES)) + "]"
        cmd.extend(["--scale_values", scale_str])
    
    if REVERSE_CHANNELS:
        cmd.append("--reverse_input_channels")
    
    print("执行转换命令：")
    print(" ".join(cmd))
    print("\n转换中...")
    
    # 执行转换
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n✅ 转换成功！")
        print(f"输出文件保存在：{os.path.abspath(OUTPUT_DIR)}")
        print(f"生成文件：{MODEL_NAME}.xml、{MODEL_NAME}.bin")
    else:
        print("\n❌ 转换失败！")
        print("错误信息：")
        print(result.stderr)

if __name__ == "__main__":
    main()