---
name: openvino-converter
description: >
  Convert mainstream AI models (PyTorch, TensorFlow, ONNX, YOLO series, point cloud models, etc.)
  to Intel OpenVINO IR format, with automated model repo cloning, weight downloading, format
  conversion, performance benchmarking, numerical validation, and inference verification. Includes
  mandatory cross-platform validation to ensure OpenVINO IR (GPU) outputs match original framework
  (CPU) outputs. Use this skill whenever the user mentions model conversion to OpenVINO, deploying
  models on Intel CPU/GPU, IR format export, model optimization for Intel hardware, or converting
  any deep learning model for Intel inference -- even if they don't explicitly say "OpenVINO".
---

# OpenVINO Model Converter

Convert AI models to OpenVINO IR format with full source code acquisition, weight download, conversion, benchmarking, numerical validation, and inference verification -- all outputs 100% reproducible.

## Workflow Overview

Follow this sequence for every conversion, whether it succeeds or fails:

1. **Acquire model & weights** -- clone source repo and download pretrained weights
2. **Convert model** -- export to ONNX, optimize, convert to OpenVINO IR
3. **Benchmark** -- run `benchmark_app` on CPU and GPU
4. **Validate** -- verify OpenVINO IR (GPU) outputs match original framework (CPU) outputs
5. **Verify** -- write and run an inference demo with real data, show results to user
6. **Document** -- write conversion report and README
7. **Deliver** -- organize all files into the standard export directory

If conversion fails, still complete steps 6-7 (failure report + root cause analysis).

---

## 1. Model & Weight Acquisition

Getting the complete model source code is critical -- many models have custom operators, preprocessing logic, or export scripts that live in the repo and won't work without them.

### Source code

- Clone the full model repository from GitHub (or the original hosting platform). The complete source code will be placed in the `<model_name>/` subdirectory of the export directory, so the user has everything needed to understand and reproduce the conversion.
- If the repo is very large, use shallow clone (`--depth 1`) to save time and disk space.

### Pretrained weights

- Actively search for official pretrained weights -- check the model repo's README, release pages, and linked resources. Having real weights is important because it enables meaningful inference verification with real data rather than random noise.
- **If weights are hosted on HuggingFace**: use the hf-mirror.com mirror for downloading. Read `references/hf-mirror-guide.md` for the exact configuration steps. This is necessary because huggingface.co is often inaccessible or very slow in China.
- **If weights > 1GB**: do not download automatically. Instead, provide the user with the direct download link, file size, and suggested download location so they can manage the download themselves. Large downloads can fail silently, consume unexpected disk space, or tie up the user's connection.
- **Network failure handling**: try multiple sources in order (official URL, mirror sites, curl/wget alternate URLs). If all fail, wait 1 minute and retry up to 3 times. Only after all retries are exhausted, report to the user with every method attempted.

## 2. Model Conversion

There are two main paths to get an OpenVINO IR model. Which one to use depends on the model framework and what the model repo provides. Refer to `scripts/example_convert.py` as a starting template and adapt it to the specific model.

### Path A: Direct export to OpenVINO IR (preferred when available)

Some frameworks and models support direct conversion to OpenVINO IR without the ONNX intermediate step. This is simpler and avoids potential ONNX compatibility issues.

- **OpenVINO `ovc` (Model Converter)** can directly convert PyTorch (`torch.nn.Module`), TensorFlow (SavedModel, `.pb`), PaddlePaddle, and TensorFlow Lite models
- Some model repos provide their own OpenVINO export scripts -- check for these first
- Use `openvino.convert_model()` in Python to convert directly from a PyTorch model object or TensorFlow SavedModel path

### Path B: Via ONNX intermediate format

When direct conversion isn't available or fails, go through ONNX:

1. **Export to ONNX** -- use the model's own export script if available in the cloned repo, otherwise write a custom export script based on the model architecture. Set appropriate input shapes and opset version.
2. **Simplify ONNX** (recommended) -- run `onnxsim` to fold constants and remove redundant ops. This often fixes compatibility issues with OpenVINO and improves inference speed.
3. **Convert to OpenVINO IR** -- use `mo` or `ovc` to produce `.xml` + `.bin` files.

### Decision guide

| Situation | Recommended path |
|-----------|------------------|
| PyTorch model with standard ops | Try Path A first (`ovc` / `openvino.convert_model`), fall back to Path B |
| TensorFlow SavedModel | Path A (`ovc` handles it directly) |
| Model repo provides ONNX export script | Path B (author-tested export is reliable) |
| Model repo provides `.onnx` file | Path B (skip export, go straight to simplify + convert) |
| Custom/exotic operators | Path B with careful opset selection, or Path A with `ov_extension` |

Always default to FP16 precision for a good balance of model size and accuracy.

### Key considerations

- Check the model repo for existing export examples or scripts (ONNX or OpenVINO) -- reuse them rather than writing from scratch, because model authors know their own edge cases best
- Handle dynamic shapes carefully -- some models need explicit static shape setting for OpenVINO compatibility
- If one path fails, try the other before giving up. Document every attempt in the failure report.

## 3. Performance Benchmarking

Use OpenVINO's official `benchmark_app` tool exclusively. Custom Python timing scripts introduce measurement noise (GIL contention, warmup variance, etc.) and produce numbers that can't be compared against published benchmarks. `benchmark_app` is the industry-standard tool that Intel and the community rely on.

### Required command

Run `benchmark_app` with the exact flags below on both CPU and GPU. The `-hint latency` + `-infer_precision f16` combination matches how the model is exported (FP16) and produces latency numbers that reflect real single-request deployment:

```bash
benchmark_app -m <model.xml> -d CPU -hint latency -infer_precision f16
benchmark_app -m <model.xml> -d GPU -hint latency -infer_precision f16
```

Do not substitute other hints (e.g. `throughput`) or precisions unless the user explicitly requests it -- the FP16 latency hint is the contract this skill delivers against.

- Save raw output logs -- these are the authoritative source of truth:
  - `benchmark_cpu_result.txt` -- full CPU test log
  - `benchmark_gpu_result.txt` -- full GPU test log
  - `benchmark_app_usage.md` -- the exact commands used, parameter explanations, how to read results
- Use `scripts/parse_benchmark.py` to extract key metrics (latency, throughput, device info) from raw logs for structured summaries
- When reporting performance numbers in README or to the user, always cite the exact log file path. Every number must trace back to `benchmark_app` output -- never approximate or editorialize performance data, because users rely on these numbers for hardware purchasing and deployment decisions

## 4. Inference Verification & Validation

A converted model is only useful if someone can actually run it. The demo serves as both a correctness check and a quickstart for the user.

### Test data selection (in priority order)

The choice of test data matters -- random noise through a real model produces meaningless output that tells you nothing about whether the conversion preserved model behavior.

1. **With pretrained weights**: use real, meaningful input data:
   - Look in the model repo for sample images, example inputs, or recommended test files
   - Check if the repo references a dataset -- download just 1 sample input (a single image, point cloud, etc.)
   - If the dataset is too large (hundreds of MB+), do not download it. Tell the user the dataset name and link, and ask if they want to provide their own test input
2. **Without pretrained weights** (or no suitable real data found): generate random input tensors matching the model's expected input shape and dtype. This at least verifies the model runs end-to-end without errors.

### Demo requirements

- Write an `infer_demo.py` that works out of the box with the included test data
- Show inference results to the user via `view_image`/`view_video` tools -- seeing actual output builds confidence the conversion is correct
- Support custom data replacement with clear instructions, so the user can swap in their own inputs

### Numerical Validation (Required)

To ensure the OpenVINO IR model preserves the original model's accuracy, perform a cross-platform numerical validation:

- **Goal**: Verify that OpenVINO IR inference on GPU produces numerically consistent results with the **original PyTorch (or equivalent source framework) inference running on CPU in FP16**. Because the exported IR is FP16, the source-side reference must also run in FP16 -- comparing against FP32 would conflate conversion error with precision-cast error.
- **Method**: Run identical input data through both pipelines in-process and compare output tensors directly. Do not persist `.npy` files -- the comparison happens live and only the resulting metrics/report are kept.
- **Do NOT compare OpenVINO CPU vs OpenVINO GPU outputs.** Both sides come from the same IR and the same runtime -- any diff reflects hardware/kernel numerics, not conversion correctness, and the result is not a meaningful signal. The only valid comparison is `source-framework CPU FP16` vs `OpenVINO IR GPU FP16`.
- **Acceptance criteria** (FP16 vs FP16):
  - mean absolute difference < 1e-3
  - max absolute difference < 1e-2
  - If the model produces classification logits or detection scores, also verify top-k indices match.
- **All validation artifacts go in `validation/` directory**:
  - `validate.py` -- automated comparison script that runs both pipelines in-process and prints/writes metrics (no `.npy` dumps)
  - `validation_report.md` -- results, metrics (mean/max/percentile errors), pass/fail status, and explicit note of the comparison pair (source-CPU-FP16 vs OV-GPU-FP16)
  - `diff_visualization.png` (optional) -- heatmap or distribution plot of errors

This validation is mandatory because deployment on Intel GPUs requires confidence that the converted model maintains accuracy relative to the original research/training code.

## 5. Provenance & Conversion Report

### Provenance tracking

Users need to know exactly where every artifact came from, both for reproducibility and for auditing:

- **Model source code**: GitHub repo URL, commit hash or tag
- **Downloaded weights**: source URL, version, file size, checksum if available
- **Local/bundled weights**: file path, origin description
- **Test data**: source (repo sample / dataset subset / locally generated), basic description

### Conversion report

Write a detailed Markdown report for every conversion -- this is the audit trail that makes the work reproducible.

- **On success**: record each step's command, output, and model metadata
- **On failure**: record every attempted approach, the failing step, full error output, root cause analysis, and viable alternatives. Name it `<model>_OpenVINO_conversion_failure_analysis.md` in the export directory

## 6. Delivery & Reproducibility

Organize all outputs under the model repo root. The goal: anyone can use the export directory to reproduce the entire conversion pipeline.

### Export directory structure

```
export_<model_name>/
  <model_name>/                    # Model source code (cloned repo)
  converter/                       # Everything related to conversion
    convert.py                     # Conversion script (ONNX export + OpenVINO conversion)
    <model_name>_simplified.xml    # OpenVINO model structure
    <model_name>_simplified.bin    # OpenVINO FP16 weights
  benchmark/                       # All benchmark data
    benchmark_cpu_result.txt       # Raw benchmark_app CPU log
    benchmark_gpu_result.txt       # Raw benchmark_app GPU log
    benchmark_app_usage.md         # benchmark_app usage guide
  validation/                      # Numerical accuracy validation
    validate.py                    # Automated in-process comparison: source-CPU-FP16 vs OV-GPU-FP16
    validation_report.md           # Validation results, metrics, and explicit comparison pair
    diff_visualization.png         # Error distribution visualization (optional)
  demo/                            # Inference demo and test data
    infer_demo.py                  # Ready-to-run inference demo
    <sample input files>           # Test image/data used by the demo
    <sample output files>          # Pre-generated inference results for comparison
  README.md                        # Full guide: environment setup, from-scratch export, quick reproduce, custom data usage
```

### README requirements

Include a complete "reproduce from scratch" section:
1. Environment and dependency installation
2. Full ONNX export and conversion steps (pointing to `converter/convert.py`)
3. How to run benchmarks and interpret results
4. How to run numerical validation (pointing to `validation/validate.py` and `validation/validation_report.md`)
5. How to run the inference demo and use custom data

Keep the file listing in README synchronized with actual directory contents.

---

## Bundled Resources

### Scripts (`scripts/`)

- **`example_convert.py`** -- Template conversion script with configurable parameters (input model, output dir, shape, precision, etc.). Use as a starting point when writing `converter/convert.py` for each model.
- **`parse_benchmark.py`** -- Parses `benchmark_app` log files and extracts key metrics (OpenVINO version, latency, throughput, device info). Use this to generate structured benchmark summaries from raw logs.

### References (`references/`)

- **`hf-mirror-guide.md`** -- How to configure HuggingFace mirror (hf-mirror.com) for downloading models and datasets in China. Read this whenever the model or weights are hosted on HuggingFace.

## Platform Notes

- **Encoding (Windows)**: Stick to ASCII-printable characters in all output. GBK encoding on Windows will choke on emoji and many Unicode symbols, causing script failures
- **Paths**: Use absolute paths for all file I/O and tool calls. Relative paths break easily when working directory changes between steps

---

## Completion Checklist

Before delivering results to the user, verify every item. This catches common omissions that break reproducibility.

### Files

- [ ] `export_<model_name>/` directory exists with correct structure
- [ ] `<model_name>/` contains the cloned model source code
- [ ] `converter/convert.py` exists and is runnable end-to-end
- [ ] `converter/<model_name>_simplified.xml` and `.bin` exist and are non-empty
- [ ] `benchmark/benchmark_cpu_result.txt` exists, produced by `benchmark_app -m <model> -d CPU -hint latency -infer_precision f16`
- [ ] `benchmark/benchmark_gpu_result.txt` exists, produced by `benchmark_app -m <model> -d GPU -hint latency -infer_precision f16`
- [ ] `benchmark/benchmark_app_usage.md` exists with the exact commands above and parameter explanations
- [ ] `validation/validate.py` exists and is runnable
- [ ] `validation/validation_report.md` exists with pass/fail status and error metrics
- [ ] Validation compares **source-framework CPU FP16** vs **OpenVINO IR GPU FP16** (never OV-CPU vs OV-GPU)
- [ ] No `.npy` files persisted under `validation/` -- comparison is done in-process
- [ ] `demo/infer_demo.py` exists and runs successfully
- [ ] `demo/` contains sample input data (real image/data or generated tensors)
- [ ] `demo/` contains pre-generated sample output for user comparison
- [ ] `README.md` exists at `export_<model_name>/` root

### README content

- [ ] Environment and dependency installation instructions
- [ ] Complete from-scratch conversion steps referencing `converter/convert.py`
- [ ] Benchmark commands and how to interpret results
- [ ] Validation instructions referencing `validation/validate.py` and results
- [ ] Demo usage with custom data replacement instructions
- [ ] File listing matches actual directory contents

### Data integrity

- [ ] All performance numbers in README trace back to `benchmark_app` log files
- [ ] Provenance documented: model source URL + commit, weight source + size, test data source
- [ ] No hardcoded absolute paths in `convert.py`, `infer_demo.py`, or `README.md` (use relative paths within the export directory for portability)

### Conversion report

- [ ] Success: detailed step-by-step report with commands and outputs
- [ ] Failure: `<model>_OpenVINO_conversion_failure_analysis.md` with all attempts, errors, and root cause
