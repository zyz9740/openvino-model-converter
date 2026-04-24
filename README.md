# OpenVINO Model Converter Skill

Convert AI models (PyTorch, TensorFlow, ONNX, YOLO, etc.) to Intel OpenVINO IR format with automated workflow: model acquisition, conversion, benchmarking, numerical validation, and inference verification.

## Installation

```bash
./install-skill.sh
```

Works on Windows (Git Bash), Linux, and macOS. Installs to `~/.claude/skills/openvino-converter/`.

**Uninstall**: `rm -rf ~/.claude/skills/openvino-converter`

## Usage

Once installed, Claude Code auto-invokes this skill for model conversion tasks:

```
"Convert YOLOv8 to OpenVINO IR"
"Deploy PyTorch ResNet50 on Intel GPU"
"Export this TensorFlow model to OpenVINO and benchmark it"
```

## Workflow

1. **Acquire** - Clone repo + download weights
2. **Convert** - Export to ONNX (optional) → OpenVINO IR
3. **Benchmark** - Run `benchmark_app` on CPU and GPU
4. **Validate** - Compare OpenVINO IR (GPU) vs original (CPU) outputs
5. **Demo** - Create inference demo with real data
6. **Document** - Generate reports and README
7. **Deliver** - Organize in standard export directory

## Output Structure

```
export_<model_name>/
├── <model_name>/           # Source code
├── converter/              # IR files (.xml, .bin) and convert script
├── benchmark/              # CPU/GPU benchmark results
├── validation/             # Numerical validation reports
├── demo/                   # Inference demo + test data
└── README.md
```

## Features

- HuggingFace mirror support for weight downloads
- Handles large files (>1GB) with user confirmation
- Failure reports with root cause analysis
- Cross-platform (Windows/Linux/macOS)
