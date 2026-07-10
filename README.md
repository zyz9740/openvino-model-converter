# OpenVINO Model Converter Skill (Lite)

Convert AI models (PyTorch, TensorFlow, ONNX, YOLO, etc.) to Intel OpenVINO IR format.
This is the fast/token-lite variant: single conversion path, GPU-only benchmark, quick
numerical validation. See `SKILL.md` -> "Limitations" for what it trades away versus a
fully rigorous conversion (dual-path comparison, CPU+GPU benchmark, tight/loose A-B
validation gate, CUDA fused-op migration).

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
"Export this TensorFlow model to OpenVINO"
```

## Workflow

1. **Acquire** - Clone repo; hand the user a weight download link and direct command (no auto-download)
2. **Convert** - Try direct export, fall back to ONNX; first success ships
3. **Benchmark** - Run `benchmark_app` on GPU only
4. **Validate** - OV-GPU-FP16 vs original framework, ~10 random inputs, no plots
5. **Demo** - Minimal inference script with real or random input
6. **Deliver** - Simple export directory + short report

## Output Structure

```
export_<model_name>/
├── <model_name>/           # Source code (weights excluded)
├── converter/              # IR files (.xml, .bin) and convert.py
├── benchmark/               # GPU benchmark log
├── validation/              # validate.py + validation_report.md
├── demo/                    # infer_demo.py + test data
└── README.md
```

## Features

- HuggingFace mirror guide for user-driven weight downloads
- Weight download is always manual, but the skill must give the user a copy/paste command such as
  `Invoke-WebRequest -Uri "<url>" -OutFile "<weights_path>"`,
  `curl -L "<url>" -o "<weights_path>"`, or
  `huggingface-cli download <org>/<model> --local-dir "<weights_dir>" --local-dir-use-symlinks False`
- No auto-fetch, no polling
- Failure reports with root-cause notes when conversion doesn't succeed
- Cross-platform (Windows/Linux/macOS)
