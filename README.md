# OpenVINO Model Converter Skill (Lite)

Convert AI models (PyTorch, TensorFlow, ONNX, YOLO, etc.) to Intel OpenVINO IR format.
This is the fast/token-lite variant: single conversion path, GPU-only benchmark, quick
numerical validation, no CUDA fused-op migration. See `SKILL.md` -> "Limitations" for what
it trades away versus a fully rigorous conversion (dual-path comparison, CPU+GPU benchmark,
tight/loose A-B validation gate, CUDA fused-op migration). File-management and reproducibility
discipline -- git hygiene, `.gitignore`, `fetch_assets.py`, push-size audits -- is unchanged
from the full skill; that part isn't where the token cost was.

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

1. **Acquire** - Clone repo (nested `.git` removed); write `fetch_assets.py` for weights, no auto-download
2. **Convert** - Try direct export, fall back to ONNX; first success ships
3. **Benchmark** - Run `benchmark_app` on GPU only
4. **Validate** - OV-GPU-FP16 vs original framework, ~10 random inputs, no plots
5. **Demo** - Minimal inference script with real or random input
6. **Deliver** - Reproducible export directory (git hygiene, size audit) + short report

## Output Structure

```
export_<model_name>/
├── .gitignore               # excludes .bin/.onnx/pretrained weights (derived files)
├── scripts/
│   └── fetch_assets.py      # weight + large test-data manifest, SHA256-verified
├── <model_name>/            # Source code, nested .git removed (weights excluded)
├── converter/                # IR files (.xml committed, .bin derived) and convert.py
├── benchmark/                 # GPU benchmark log
├── validation/                # validate.py + validation_report.md
├── demo/                      # infer_demo.py + test data
└── README.md
```

## Features

- HuggingFace mirror guide for user-driven weight downloads (via `fetch_assets.py`)
- Weight download is always manual -- the skill writes the `fetch_assets.py` manifest, the user
  runs it; no auto-fetch, no polling in-session
- Committed-vs-derived file policy and a pre-push size audit (95 MB threshold), no Git LFS
- Failure reports with root-cause notes when conversion doesn't succeed
- Cross-platform (Windows/Linux/macOS)
