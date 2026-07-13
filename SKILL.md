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
7. **Deliver** -- organize all files into the standard export directory, then ask the user if they want to pursue optimization; if so, recommend a per-layer IR profile first (see Section 3's "Optional: per-layer IR profiling" and Section 6's closing question)
8. **(Optional) CUDA fused-op migration** -- if the source repo contains hand-written CUDA kernels for fused operations AND the baseline conversion passes cleanly, profile the equivalent subgraph on OV-GPU, propose a custom-op port, and -- only with the user's explicit agreement -- build an `optimize_v2/` rerun. See Section 7.

If conversion fails, still complete steps 6-7 (failure report + root cause analysis). Step 8 is skipped on failed conversions -- there is nothing to optimize on top of a broken baseline.

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

There are two main paths to get an OpenVINO IR model. The conversion toolchain is never a pure formality -- it can leave a different op graph even for the same model, which in turn changes compile time and sometimes runtime speed. So the policy is: **try the direct path first, and when it works, also run the ONNX path so the user can see the real trade-off on their model**.

### Default order: Path A first, then Path B

1. **Always attempt Path A (direct torch/TF -> OpenVINO IR) first.** In practice direct export tends to produce a leaner graph -- fewer `Reshape`/`Unsqueeze` bookkeeping ops, modern fused ops kept whole (e.g. `GroupNormalization`, `ScaledDotProductAttention`) rather than decomposed into ONNX-opset-era primitives. That usually means lower GPU compile time, sometimes lower runtime latency, and a smaller `.xml`. But direct export can also fail on exotic ops or custom control flow, which is why it isn't a hard rule.
2. **If Path A fails, fall back to Path B (ONNX intermediate).** ONNX has broader op coverage and a more mature fallback-friendly toolchain; many models that won't trace through `ov.convert_model()` directly will still export cleanly to ONNX and convert from there.
3. **If Path A succeeds, still run Path B as well.** Build both IRs, benchmark both, compare the graphs. The result is what the user actually cares about -- is the direct path really faster for *this* model? Sometimes it is by double digits, sometimes it's a wash, and occasionally Path B wins because ONNX simplification folds constants more aggressively. The only way to know is to measure. See Section 3 for the comparison requirements.

Refer to `scripts/example_convert.py` as a starting template and adapt it to the specific model.

### Path A: Direct export to OpenVINO IR

- **OpenVINO `ovc` (Model Converter)** can directly convert PyTorch (`torch.nn.Module`), TensorFlow (SavedModel, `.pb`), PaddlePaddle, and TensorFlow Lite models
- Some model repos provide their own OpenVINO export scripts -- check for these first
- Use `openvino.convert_model(model, example_input=...)` in Python to convert directly from a PyTorch module or TensorFlow SavedModel path
- When the torch frontend assigns generic I/O names (like `x.1`, `37`), relabel them on the resulting `ov.Model` via `port.get_tensor().set_names({...})` so the IR contract matches the ONNX-path IR and downstream demo/validation code can stay shared

### Path B: Via ONNX intermediate format

1. **Export to ONNX** -- use the model's own export script if available in the cloned repo, otherwise write a custom export script based on the model architecture. Set appropriate input shapes and opset version.
2. **Simplify ONNX** (recommended) -- run `onnxsim` to fold constants and remove redundant ops. This often fixes compatibility issues with OpenVINO and improves inference speed.
3. **Convert to OpenVINO IR** -- use `ovc` or `openvino.convert_model()` on the `.onnx` file to produce `.xml` + `.bin`.

### Decision guide

| Situation | What to do |
|-----------|------------|
| PyTorch module with standard ops | Path A first; if it succeeds, also run Path B and compare |
| TensorFlow SavedModel | Path A first; if it succeeds, also run Path B and compare |
| Path A fails (tracing error, unsupported op) | Fall back to Path B; document the failure in the conversion report |
| Model repo only provides a `.onnx` file, no source | Path B only -- there is no torch module to hand to `ov.convert_model()` |
| Custom/exotic operators | Path B with careful opset selection, or Path A with `ov_extension` |

Always default to FP16 precision for a good balance of model size and accuracy.

### Key considerations

- When both paths work, keep both IRs in `converter/` (e.g. `<model>_direct.xml` and `<model>_simplified.xml`) and write ONE shared model-build/patch module that both `convert.py` (ONNX path) and `convert_direct.py` (direct path) import. Duplicating the patching logic across two scripts is a maintenance hazard.
- Check the model repo for existing export examples or scripts (ONNX or OpenVINO) -- reuse them rather than writing from scratch, because model authors know their own edge cases best
- Handle dynamic shapes carefully -- some models need explicit static shape setting for OpenVINO compatibility
- If Path A fails, try Path B before giving up. Document every attempt in the failure report.

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
  - `benchmark_cpu_result.txt` -- full CPU test log for the primary IR (Path A if it succeeded, otherwise Path B)
  - `benchmark_gpu_result.txt` -- full GPU test log for the primary IR
  - `benchmark_app_usage.md` -- the exact commands used, parameter explanations, how to read results
- Use `scripts/parse_benchmark.py` to extract key metrics (latency, throughput, device info) from raw logs for structured summaries
- When reporting performance numbers in README or to the user, always cite the exact log file path. Every number must trace back to `benchmark_app` output -- never approximate or editorialize performance data, because users rely on these numbers for hardware purchasing and deployment decisions

### Path A vs Path B comparison (required when both paths succeed)

Whenever Section 2 yields two IRs -- one from direct export (Path A) and one from the ONNX route (Path B) -- you must benchmark **both** and produce an explicit comparison. This is the whole reason we bothered to build two IRs; a conversion report that only includes one set of numbers is throwing away the most useful information the skill produces.

Additional files to save under `benchmark/`:

- `benchmark_cpu_direct_result.txt`, `benchmark_gpu_direct_result.txt` -- logs for the direct-path IR
- (The ONNX-path logs live in the standard `benchmark_cpu_result.txt` / `benchmark_gpu_result.txt`, so both sides are preserved.)
- `comparison_onnx_vs_direct.md` -- written comparison, covering:
  1. **Latency table (CPU & GPU)** -- median, average, min, throughput, first-inference, compile-time for both IRs, with the percent delta. Pull numbers straight from the raw logs.
  2. **Op-graph diff** -- load both IRs with `ov.Core().read_model()`, walk `model.get_ops()`, and print a `collections.Counter` of op type names. Put both counters side-by-side in the report and call out specific differences (e.g. "direct path keeps `GroupNormalization`; ONNX path expands it into `Reshape + MVN + Reshape + Multiply + Add` because opset 16 has no native GroupNorm").
  3. **File size diff** -- `.xml` and `.bin` byte sizes for both IRs. Weights should be identical for the same model; `.xml` size usually drops on the leaner path.
  4. **Recommendation** -- a short paragraph naming which IR is the better default for this model and why. Be honest when the two are within benchmark noise (say so explicitly rather than inventing a winner).

The op-graph diff matters because latency alone doesn't tell you *why* one path is faster. Saying "direct path is 2% faster on GPU and compiles 23% faster because it kept `GroupNormalization` fused and has 131 fewer graph nodes" is an actionable insight; saying "direct path is 2% faster" is a coin flip that could reverse on the next OpenVINO release. Always include the structural explanation.

### Optional: per-layer IR profiling

`benchmark_app`'s headline latency/throughput numbers say *how fast*, not *where the time goes inside the graph*. When the user wants to optimize (rather than just ship the baseline), re-run `benchmark_app` with `-exec_graph_path` to get a per-layer breakdown -- no extra tooling required:

```bash
benchmark_app -m <model.xml> -d GPU -hint latency -infer_precision f16 \
    -niter 50 -exec_graph_path exec_graph.xml
```

This is not part of the required benchmarking flow -- run it on request, or proactively offer it once conversion is complete (see Section 6). Full method, the two available flags (`-pc` vs `-exec_graph_path`), what each layer's attributes mean, and how to read the results: `references/per-layer-profiling.md`.

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

Single-image validation is insufficient -- a model can pass on one input and diverge badly on others due to numerical edge cases (near-zero norms, high-contrast features, out-of-distribution activations). Use diverse inputs and statistical metrics to build real confidence.

#### Reference selection -- run BOTH comparisons

The OV IR is exported in FP16 and runs on GPU. There are two natural references on CPU (the source framework at FP16, and the source framework at FP32), and they answer **different questions**. You must run both, because each one alone is insufficient and -- more importantly -- only the pair lets you attribute a failure correctly.

| Comparison | What it isolates | Threshold style |
|------------|------------------|-----------------|
| **A. OV-GPU-FP16 vs Source-CPU-FP16** *(primary, conversion gate)* | Conversion correctness only. Both sides are FP16, so any difference comes from OV plugin behavior, op fusion, kernel implementation, or layout -- **not** from precision casting. | TIGHT -- this is the pass/fail gate for whether the conversion itself is correct |
| **B. OV-GPU-FP16 vs Source-CPU-FP32** *(secondary, deployment quality)* | End-to-end quality of what the user actually ships. Mixes conversion error AND the FP32->FP16 precision cast, so it cannot separate them on its own. | LOOSE -- raw element diff is expected to be larger; rely on task-level metrics |

Why both, and not just one:

- **Skipping B** ("FP16 vs FP16 passes, ship it") hides the case where FP16 *is* mathematically OK on this network but downstream task quality (mAP, top-1, PSNR) collapses anyway because the model is precision-sensitive. The user needs to know whether to actually deploy in FP16.
- **Skipping A** ("FP16 vs FP32 has 1e-2 max diff, conversion is broken") is the single most common false alarm in this skill's history. A 1e-2 element-wise diff is *normal* for FP16 inference on networks containing batch-norm, softmax, attention, or large reductions -- it is the expected cost of FP16, not a bug. Without A, you cannot tell.

For optimized or decomposed pipelines (split models, custom kernels replacing subgraphs), the FP16 reference becomes the already-validated baseline OV IR on the same device and precision -- the question shifts from "did the conversion preserve the model?" to "did the optimization break anything?", but the principle (compare A like-for-like, then check B for end-to-end quality) is unchanged.

#### Attribution: conversion error vs precision degradation

After running both A and B, classify the result with this table. The whole reason for running two comparisons is so the diagnosis is unambiguous:

| A (FP16 vs FP16) | B (FP16 vs FP32) | Diagnosis & action |
|------------------|------------------|--------------------|
| PASS (tight)     | PASS (loose)     | **Healthy.** Conversion is correct AND FP16 precision is acceptable for this model. Ship the FP16 IR. |
| PASS (tight)     | FAIL             | **Precision degradation, NOT a conversion bug.** The IR faithfully reproduces the FP16 source model; FP16 itself is too lossy for this network. Options: (a) export an FP32 IR and accept the latency hit, (b) use mixed precision -- keep numerically sensitive ops (softmax, layer norm, final logits) in FP32 while the bulk runs FP16, (c) re-train or fine-tune with QAT, (d) verify whether the user's downstream task actually needs the lost precision. Do **not** start hunting through the converter for bugs -- there are none here. |
| FAIL             | (any)            | **Conversion / GPU-plugin bug.** B is meaningless until A is fixed -- a broken conversion will of course also fail vs FP32. Common root causes, in rough order of frequency: op fusion mismatch (e.g. `GroupNormalization` decomposed differently between paths), missing or wrong op extension on a custom op, unexpected layout transform (NCHW vs NHWC), an attribute lost during ONNX simplification, dynamic-shape collapse to wrong static shape, dtype downcast on a constant. Re-run validation after each fix and only re-evaluate B once A passes. |
| FAIL             | PASS             | **Anomalous, almost always a threshold-calibration issue.** Tighten A's thresholds (it is supposed to be the strict gate) and re-run; if A still passes after recalibration, you have a real but unusual case worth a note in the report. |

In the validation report, state the diagnosis explicitly using this table -- do not just dump numbers and let the reader guess. "Comparison A passed (max_abs 4.2e-4); B failed at max_abs 3.1e-2 driven mostly by softmax outputs near class boundaries -> conclusion: conversion is correct, FP16 precision is the limiting factor for this model. Recommend mixed precision for the final softmax." That sentence is the deliverable.

#### Input coverage

Use at least 1 real sample plus N >= 10 synthetic inputs covering multiple distributions. The point is to exercise numerical paths the real image alone cannot reach:

- **Real data**: a sample from the model's domain (demo image, test point cloud, etc.)
- **ImageNet-range**: random pixels in [0,255] with standard normalization -- realistic activation magnitudes
- **Gaussian**: standard normal noise -- covers a broad activation range
- **Near-zero**: values ~1e-3 -- stresses normalization layers, division-by-small-number paths
- **High-contrast**: step patterns with small perturbation -- tests spatial gradient handling

Generate each category with a fixed random seed so results are reproducible.

#### Metrics and acceptance

For each input, run **both** comparisons (A: FP16 vs FP16, B: FP16 vs FP32) and report per-element:

- mean_abs, p95_abs, p99_abs, max_abs
- correlation (for regression outputs like disparity, depth, flow)
- NaN count (the cardinal failure mode -- any NaN is an automatic fail in either comparison)
- fraction of elements exceeding threshold (e.g. >0.5, >1.0, >2.0 output-units)

Acceptance thresholds differ deliberately between A and B because they measure different things. Adapt the absolute numbers to the model's output scale (logits vs disparity-pixels vs probabilities), but keep the relative gap between the two columns -- the asymmetry is the entire point.

| Metric | A: FP16 vs FP16 (conversion gate, TIGHT) | B: FP16 vs FP32 (deployment quality, LOOSE) |
|--------|------------------------------------------|---------------------------------------------|
| mean_abs | < 1e-4 | < 1e-2 |
| max_abs  | < 1e-3 | < 1e-1 (or < 20 output-units for pixel-level tasks) |
| correlation (regression outputs) | > 0.9999 | > 0.99 |
| top-k indices (classification/detection) | must match exactly | top-1 must match; top-5 may differ at boundaries |
| NaN | zero, always | zero, always |

If A fails, do not bother grading B against its thresholds -- fix A first (see attribution table). If A passes but B fails, that is **not** a conversion failure; report it as an FP16 precision finding, not a bug.

Report aggregate statistics across all inputs (min, median, max of per-input metrics) for **both** A and B, so a single outlier doesn't hide behind an average and so the report shows the FP16 vs FP32 spread the user needs to make a deployment-precision decision.

#### Per-stage attribution (recommended for multi-stage pipelines)

When the pipeline has multiple stages (e.g. feature_runner -> custom_kernel -> post_runner), compare intermediate outputs stage-by-stage against the reference to isolate where error originates. This tells you whether error comes from the model decomposition, a custom kernel, or accumulated rounding through the chain.

#### Validation artifacts

All validation artifacts go in `validation/` directory:
  - `validate.py` -- automated comparison script that runs **three** pipelines in one process (Source-CPU-FP32, Source-CPU-FP16, OV-GPU-FP16), feeds the same inputs through all three, computes A and B metrics, and writes results. No `.npy` file dumps.
  - `validation_results.json` -- machine-readable per-input metrics for **both** comparisons (top-level keys `comparison_A_fp16_vs_fp16` and `comparison_B_fp16_vs_fp32`), for CI integration
  - `validation_report.md` -- results summary that MUST include:
    1. Both comparisons' aggregate metrics side-by-side
    2. Pass/fail per comparison against its own threshold tier (A tight, B loose)
    3. Explicit attribution sentence using the diagnosis table above (e.g. "A passed, B failed -> precision degradation, not a conversion bug; recommend mixed precision on softmax")
    4. Reference precisions and devices spelled out (no ambiguity about what was compared to what)

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

Organize all outputs under the model repo root. The goal: anyone can `git clone` the export directory and reproduce the entire conversion pipeline. The structure assumes the export directory is or will be a Git repository pushed to GitHub.

### Export directory structure

Files marked **(derived)** are deliberately NOT committed to Git -- they are reproduced by running the included scripts. See "GitHub-ready cleanup" below for why and how.

```
export_<model_name>/
  .gitignore                       # Excludes derived large files (see policy below)
  requirements.txt                 # Pinned Python deps for the whole pipeline
  scripts/
    fetch_assets.py                # Downloads pretrained weights from origin URLs, with SHA256 verification
  <model_name>/                    # Model source code (cloned repo)
                                   # Pretrained weight files inside (e.g. *.pt/*.ckpt) are (derived) -- not committed
  converter/                       # Everything related to conversion
    convert.py                     # ONNX-path script: torch -> ONNX -> onnxsim -> OV IR
    convert_direct.py              # Direct-path script: torch -> OV IR via ov.convert_model()
                                   # (omit when only Path B was viable; note the reason in the report)
    <model_name>_simplified.xml    # OpenVINO IR graph from ONNX path -- COMMITTED (text, small)
    <model_name>_simplified.bin    # (derived) IR weights -- NOT committed; rebuild with convert.py
    <model_name>_direct.xml        # OpenVINO IR graph from direct path -- COMMITTED
    <model_name>_direct.bin        # (derived) -- NOT committed; rebuild with convert_direct.py
    <model_name>.onnx              # (derived) ONNX intermediate -- NOT committed; rebuild with convert.py
  benchmark/                       # All benchmark data (text logs, all committed)
    benchmark_cpu_result.txt           # ONNX-path CPU log
    benchmark_gpu_result.txt           # ONNX-path GPU log
    benchmark_cpu_direct_result.txt    # Direct-path CPU log (when Path A succeeds)
    benchmark_gpu_direct_result.txt    # Direct-path GPU log (when Path A succeeds)
    comparison_onnx_vs_direct.md       # Side-by-side latency + op-graph diff + recommendation
    benchmark_app_usage.md             # benchmark_app usage guide
  validation/                      # Numerical accuracy validation
    validate.py                    # In-process comparison running BOTH:
                                   #   A. OV-GPU-FP16 vs Source-CPU-FP16 (conversion gate, tight)
                                   #   B. OV-GPU-FP16 vs Source-CPU-FP32 (deployment quality, loose)
    validation_results.json        # Per-input metrics for both A and B
    validation_report.md           # Results + explicit attribution (conversion bug vs precision loss)
  demo/                            # Inference demo and test data
    infer_demo.py                  # Ready-to-run inference demo
    <sample input files>           # Test image/data -- COMMIT only if each <95MB; larger ones are (derived)
                                   # and fetched by scripts/fetch_assets.py
    <sample output files>          # Pre-generated inference results -- same size rule
  optimize_v2/                     # Created ONLY by Stage 8; absent on standard conversions. See Section 7.
    custom_op/                     # Custom GPU op authored via ov-custom-pipeline skill (source committed,
                                   # build/ outputs (derived))
    converter/                     # convert_v2.py + <model>_v2.xml committed, <model>_v2.bin (derived)
    benchmark/                     # v2 benchmark logs + v2_vs_baseline.md
    validation/                    # validate_v2.py with both A and B re-run against baseline IR
    demo/                          # infer_demo_v2.py
    optimize_v2_report.md          # Honest verdict on whether v2 beats baseline
  README.md                        # Full guide: environment setup, fetch_assets, conversion, benchmark, validate, demo
```

### GitHub-ready cleanup (run as the last step before delivery)

GitHub rejects any single file > 100 MB on push, and a polluted history is painful to clean up after the fact. So the rule is: **never commit a file that can be reproduced from source code + pretrained weights**. Do this cleanup as the final step of Section 6, after everything else has run successfully -- not interleaved with conversion, because you want the derived files to actually exist locally during validation/demo.

#### What gets committed vs. what is derived

| Category | Examples | Treatment |
|----------|----------|-----------|
| Source code, configs | `.py`, `.cpp`, `.cl`, `requirements.txt`, `.gitignore` | Commit |
| OV IR graph | `.xml` | Commit (text, usually < 1 MB even for large models) |
| Reports & logs | `.md`, `.txt`, `.json`, benchmark logs | Commit |
| Small test inputs | sample images < 95 MB each | Commit |
| **OV IR weights** | `.bin` | **Derived** -- regenerated by `convert.py` / `convert_direct.py` |
| **ONNX intermediates** | `.onnx` | **Derived** -- regenerated by `convert.py` |
| **Pretrained model weights** | `.pt`, `.pth`, `.ckpt`, `.h5`, `.safetensors` | **Derived** -- fetched by `scripts/fetch_assets.py` |
| **Large test inputs** | any single file > 95 MB | **Derived** -- fetched by `scripts/fetch_assets.py` |
| Build outputs | `optimize_v2/custom_op/build/` | **Derived** -- rebuilt locally |

The 95 MB threshold (not 100) leaves headroom for git pack overhead -- a 99 MB file occasionally fails to push.

#### `.gitignore` template

Write this file at `export_<model_name>/.gitignore`:

```gitignore
# OpenVINO IR weights -- regenerated by converter/convert*.py
*.bin

# ONNX intermediates -- regenerated by converter/convert.py
*.onnx

# Pretrained model weights -- fetched by scripts/fetch_assets.py
*.pt
*.pth
*.ckpt
*.h5
*.safetensors
*.tar
*.tar.gz

# Build artifacts (Stage 8 custom op)
optimize_v2/custom_op/build/

# Python
__pycache__/
*.pyc
.venv/
```

Adapt extensions to the specific model -- e.g. add `*.npz` if the model ships weights that way. The principle is "if a script can recreate it, don't commit it"; the extension list is just the common shapes that takes.

#### `scripts/fetch_assets.py`

Every artifact that won't be committed needs a way back. The script must:

1. Define an explicit asset manifest -- a list of `{url, dest_path, sha256, size_bytes, description}` records, hardcoded in the script (or loaded from a sibling JSON). Do NOT scrape URLs at runtime; the manifest is the audit trail.
2. Download each asset to its **exact relative destination** under the export directory (e.g. weights go straight into `<model_name>/checkpoints/best.pt`, not into a flat `downloads/` folder the user has to sort out).
3. Verify SHA256 after each download. Mismatch is a hard fail -- do not silently proceed; the URL may have been replaced.
4. Skip assets already present with the correct hash, so re-running is cheap.
5. Print one-line progress per asset and a final summary.

For HuggingFace-hosted weights, route through `hf-mirror.com` per `references/hf-mirror-guide.md`. Always keep the original `huggingface.co` URL as an alternate in the manifest -- if the mirror is down, the user can flip to it.

#### Push-time size check

Before the user runs `git push`, the skill must surface a size audit. Run something equivalent to:

```bash
git ls-files -z | xargs -0 -I{} stat -c "%s %n" "{}" | awk '$1 > 95000000 {print}'
```

If anything comes back, **stop and fix the `.gitignore`**. Do not push with `--force` or fight the rejection at the server -- the pre-push check exists to catch it locally where the fix is cheap. Mention to the user any file the audit caught and what the rule says to do with it (commit if it's irreplaceable source data; add to fetch_assets if it's reproducible).

#### What this skill does NOT do

- **No Git LFS.** LFS has a 1 GB/month bandwidth cap on free GitHub accounts shared across the entire user account; one popular repo can exhaust it in a day, after which clones fail. The fetch-from-source approach has no such cap.
- **No GitHub Releases assets as a default path.** They work, but they introduce a second source of truth (the Release page) that drifts from the Git history. Stick to "code in Git, weights at origin, derived at user".
- **No splitting large files into chunks.** That's a hack; the right answer is fetch_assets.py.

### README requirements

Include a complete "reproduce from scratch" section. Order matters here -- a fresh `git clone` will be missing every derived file (`.bin`, `.onnx`, pretrained weights), so the README must walk the user from clone to working demo in one pass:

1. **Environment**: `pip install -r requirements.txt`
2. **Fetch large derived assets**: `python scripts/fetch_assets.py` -- explain that this downloads pretrained weights and any large test inputs from their origins, with SHA256 verification. List which files appear after this step.
3. **Conversion** (rebuilds `.bin` and `.onnx` locally), for **both** paths when available:
   - `converter/convert_direct.py` -- direct torch/TF -> OV IR (Path A, default)
   - `converter/convert.py` -- ONNX intermediate route (Path B, fallback and reference)
4. How to run benchmarks and interpret results, including a pointer to `benchmark/comparison_onnx_vs_direct.md` when both IRs exist
5. How to run numerical validation (pointing to `validation/validate.py` and `validation/validation_report.md`)
6. How to run the inference demo and use custom data

Also include a short "What's in this repo and what isn't" section near the top, listing the (derived) files and pointing at fetch_assets.py / convert.py as the way to recreate them. This pre-empts the "where are the .bin files?" question on every fresh clone.

Keep the file listing in README synchronized with actual directory contents.

### Closing question: offer per-layer profiling

Once delivery is complete (conversion, benchmark, validation, demo, README all done), ask the user once whether they want to pursue optimization. If they do, recommend running a per-layer IR profile first -- it's what turns "make it faster" into a concrete target instead of guesswork. Use wording along these lines:

> The conversion is complete and delivered. If you're interested in optimizing further, I'd recommend starting with a per-layer performance profile of the IR (`benchmark_app -exec_graph_path`) to see exactly which layers dominate inference time before deciding where to focus. Want me to run that?

If the user agrees, follow `references/per-layer-profiling.md` (also summarized in Section 3) and report the top hot layers by name, type, and share of total time. If the hot layers trace back to a hand-written CUDA fused kernel in the source repo, this is also the natural segue into Section 7 (Stage 8) -- mention it as a possible next step, but don't start Stage 8 without going through its own trigger check and decision gate.

Don't run this profiling unprompted -- it's an optional add-on to a completed delivery, not a required step, so wait for the user's answer before spending the extra `benchmark_app` run.

---

## 7. (Optional) CUDA Fused-Op Migration to OpenVINO Custom Op

This stage exists because many high-performance model repos ship hand-written CUDA kernels that fuse multiple ops into one launch (e.g. a fused attention, a fused voxelization, a fused decoder head). On NVIDIA hardware this is a clear win. On Intel GPU through OpenVINO, the equivalent subgraph is whatever the IR's stock op decomposition gives us, which is rarely as tight. If the user is targeting Intel deployment, leaving that fusion on the table is exactly the kind of thing this skill should surface.

But this is **optional, opt-in work**. It costs real time (kernel writing, integration, re-validation), and it is only justified when (a) a fused CUDA kernel actually exists in the source repo and (b) the baseline OV conversion is otherwise healthy. Doing it on a broken baseline just stacks two sources of error.

### 7.1 Trigger conditions -- both must hold

Run this stage only when ALL of the following are true:

- The cloned source repo contains hand-written CUDA / `__global__` / `cudaLaunchKernel` code that implements a **fused operation** (i.e. it does the work of multiple framework-level ops in a single kernel). Plain CUDA-via-PyTorch (e.g. `torch.bmm`) does **not** count -- that is already covered by stock OV ops. What counts is custom `.cu` / `.cuh` files in the repo, or `torch.utils.cpp_extension` / `setup.py` building a CUDA extension that the model's forward path calls into.
- Sections 1-6 all passed: conversion succeeded (Path A and/or Path B), benchmarks ran, both validation comparisons (A: FP16 vs FP16, B: FP16 vs FP32) passed against their thresholds, and the demo runs.

If either condition is missing, skip this section entirely. Mention in the conversion report what you saw and why you did not run Stage 8 -- the user may want to revisit it later.

### 7.2 Required dependent skills -- check first, hard stop if missing

This stage **delegates** to two other skills. Before doing any work in Stage 8:

1. Check that **`vtune-profiler-skill`** is available (used for per-kernel GPU timing of the fused subgraph as it currently runs in OV).
2. Check that **`ov-custom-pipeline`** is available (used for the actual custom-op authoring -- it covers single-graph fusion, multi-model OCL orchestration, and SYCL interop, and decides which approach fits the kernel).

If either skill is missing, **stop immediately**. Do not attempt to write a custom op from scratch without the dependent skill -- the result will diverge from the user's other custom-op work and bypass the validated patterns those skills exist to encode. Tell the user exactly which skill is missing, point them at `skill-creator` to install or build it, and wait. Example wording:

> Stage 8 (CUDA -> OV custom-op migration) needs the `ov-custom-pipeline` skill, which is not currently installed. Please install it (or create it via `skill-creator`) before I continue. I have not started any custom-op work yet -- the baseline conversion in Sections 1-6 is complete and unaffected.

Do not work around the absence by reading documentation and freelancing the kernel. The whole point of the skill dependency is consistency.

### 7.3 Profiling: is the OV-GPU subgraph actually slow enough to be worth porting?

Before proposing the optimization, *measure*. A custom op that saves 50 microseconds out of a 30 ms inference is not worth the maintenance cost. Use `vtune-profiler-skill` to get per-kernel timing of the current OV-GPU run.

Steps:

1. **Identify the target subgraph.** Read the CUDA kernel's source to understand what op sequence it fuses (e.g. "this kernel does scatter -> elementwise mul -> reduce-sum"). Map that to the corresponding region in the OV IR by walking `model.get_ops()` and locating the matching node names.
2. **Run a `gpu-hotspots` analysis on the baseline OV IR** under `benchmark_app -hint latency -infer_precision f16 -d GPU`, following the procedure in `vtune-profiler-skill`. Per-kernel rows -- never averaged.
3. **Sum the kernel-level cost of the target subgraph.** That is your optimization budget: the maximum time a custom op could save if it were free. The actual saving will be less.
4. **Compute the budget as a fraction of total inference latency.** If the fused-equivalent subgraph is < 5% of total wall-clock, the upper bound on improvement is small -- usually not worth a custom op. If it's > 15%, a custom op is likely worth proposing. The 5-15% middle band depends on how mature/tricky the kernel is; lean toward "skip" for one-off models and "do it" for models the user expects to deploy widely.

Save the profiling output under `validation/optimize_v2_feasibility/` (next to the baseline validation, so the user can see the analysis even if they decline to proceed):

```
validation/
  optimize_v2_feasibility/
    vtune_gpu_hotspots/        # raw VTune result directory or its export
    target_subgraph_kernels.md # which kernel rows correspond to the CUDA-fused region, summed cost
    feasibility_report.md      # budget % of total latency, recommendation, and a yes/no proposal
```

### 7.4 Decision gate -- ask the user before any code is written

After profiling, write a short feasibility report and **stop to ask the user**. Do not start writing a custom op proactively. The report and question should look roughly like this:

> Stage 8 feasibility check:
> - The source repo's `<kernel_name>.cu` fuses ops X, Y, Z into one CUDA launch.
> - In the current OV-GPU IR, that subgraph is implemented as N stock ops, taking T ms / inference (P% of total latency T_total).
> - VTune shows the dominant cost is `<dominant kernel>` at K ms (occupancy O%, cache-hit C%, ...).
> - My recommendation: <port / skip / borderline>, because <one-sentence reason>.
>
> Do you want me to proceed with writing an OpenVINO custom op for this? If yes, I will use `ov-custom-pipeline` to choose between single-graph fusion / multi-model OCL / SYCL interop, build under `export_<model_name>/optimize_v2/`, and re-run Sections 2-5 (convert / benchmark / validate / demo) on the optimized version.

If the user declines (or says "later"), stop here. Save the feasibility report and move on. **Do not** start the optimization to "save time" -- the user's agreement is a hard gate.

If the user agrees, proceed to 7.5.

### 7.5 Building `optimize_v2/`

Delegate the custom-op authoring entirely to `ov-custom-pipeline`. Its job is to pick between the three fusion strategies and produce the kernel + integration. This skill's job at this stage is the **harness around it**: directory layout, re-running Sections 2-5 with the new IR as input, and producing a comparison report.

Directory layout under the existing export directory (do not create a new `export_<model_name>_v2/` -- this is the same model, just an alternative IR for the same source):

```
export_<model_name>/
  ... (everything from Sections 1-6 unchanged, untouched) ...
  optimize_v2/
    custom_op/                       # populated by ov-custom-pipeline
      <kernel_source_files>          # .cl / .cpp / .sycl per the chosen approach
      build/                         # compiled artifacts
      build.md                       # build instructions, copied/adapted from ov-custom-pipeline
    converter/
      convert_v2.py                  # builds the optimized IR, invoking the custom op
      <model>_v2.xml                 # optimized OpenVINO IR
      <model>_v2.bin
    benchmark/
      benchmark_cpu_v2_result.txt    # only if the custom op runs on CPU; otherwise note "GPU-only" in v2_summary.md
      benchmark_gpu_v2_result.txt
      v2_vs_baseline.md              # latency table + op-graph diff vs the baseline IR, plus the same VTune
                                     # gpu-hotspots run on v2 so per-kernel cost can be diffed against 7.3
    validation/
      validate_v2.py                 # SAME validation harness as Section 4, with v2 IR as the new "OV-GPU-FP16" side
      validation_results_v2.json
      validation_report_v2.md        # both A and B comparisons, same diagnosis table
    demo/
      infer_demo_v2.py               # demo using v2 IR (can be a thin wrapper around the original)
    optimize_v2_report.md            # narrative: what was fused, which ov-custom-pipeline approach, measured speedup,
                                     # validation status, and an honest "was this worth it?" summary
```

Re-running rules:

1. **Re-run Section 2 against the v2 IR only when it is meaningfully different from baseline.** Do not regenerate `<model>_simplified.xml` etc. -- those are stable.
2. **Re-run Section 3 (benchmark) on v2.** Save logs under `optimize_v2/benchmark/`. Compare against the original `benchmark/benchmark_gpu_result.txt` in `v2_vs_baseline.md`.
3. **Re-run Section 4 (validation, BOTH comparisons) on v2.** This is non-negotiable. A custom op is exactly the kind of change that can pass A (FP16 vs FP16) by chance on one input and explode on another. Use the same diverse input set as the baseline. Apply the same A-tight / B-loose threshold tiers. The reference for A becomes "the baseline OV IR at FP16" rather than the source framework -- you are now answering "did v2 stay equivalent to v1?" -- which is the right question for an optimization, per Section 4's note on optimized pipelines.
4. **Re-run Section 5 (demo)** on at least the real sample input, and visually confirm output matches baseline.

If validation on v2 fails (A fails -> custom op has a bug; B fails -> custom op introduced extra precision loss beyond stock FP16), do **not** silently fall back to baseline. Report the failure, keep v2 artifacts so the user can see what went wrong, and recommend either fixing the kernel via `ov-custom-pipeline` or abandoning the optimization. The user's trust depends on us being honest about when an optimization didn't pan out.

### 7.6 Final report

Append a section to the main conversion report (or write `optimize_v2_report.md` and link it) covering:

- Which CUDA kernel(s) were targeted, with file paths in the source repo
- VTune-measured baseline cost (Section 7.3 numbers)
- Which `ov-custom-pipeline` approach was chosen and why
- v2 vs baseline: latency delta, op count delta, validation status (A and B both)
- Honest verdict: keep v2 as the recommended deployment IR, or keep baseline because v2 didn't beat it by enough / didn't validate / introduced too much precision loss

The verdict matters more than the speedup number. A 30% speedup that fails B is not a win; a 5% speedup that passes both and simplifies the graph might be.

---

## Bundled Resources

### Scripts (`scripts/`)

- **`example_convert.py`** -- Template conversion script with configurable parameters (input model, output dir, shape, precision, etc.). Use as a starting point when writing `converter/convert.py` for each model.
- **`parse_benchmark.py`** -- Parses `benchmark_app` log files and extracts key metrics (OpenVINO version, latency, throughput, device info). Use this to generate structured benchmark summaries from raw logs.
- **`fetch_assets_template.py`** -- Template for the per-export `scripts/fetch_assets.py`. Implements the manifest + SHA256 verify + skip-if-present pattern described in Section 6 "GitHub-ready cleanup". Copy it into each export directory and fill in the manifest for that model's weights and large test inputs.

### References (`references/`)

- **`hf-mirror-guide.md`** -- How to configure HuggingFace mirror (hf-mirror.com) for downloading models and datasets in China. Read this whenever the model or weights are hosted on HuggingFace.
- **`per-layer-profiling.md`** -- How to get per-layer timing out of a converted IR using `benchmark_app -exec_graph_path` (or `-pc`), what each layer's attributes mean, and how to read the results. Read this when the user wants to optimize and needs to know which layers dominate inference time (see Section 3 and the closing question in Section 6).

## Platform Notes

- **Encoding (Windows)**: Stick to ASCII-printable characters in all output. GBK encoding on Windows will choke on emoji and many Unicode symbols, causing script failures
- **Paths**: Use absolute paths for all file I/O and tool calls. Relative paths break easily when working directory changes between steps

---

## Completion Checklist

Before delivering results to the user, verify every item. This catches common omissions that break reproducibility.

### Files

- [ ] `export_<model_name>/` directory exists with correct structure
- [ ] `<model_name>/` contains the cloned model source code
- [ ] **Path A attempted first** (`convert_direct.py` using `ov.convert_model()`) -- success or failure documented in the conversion report
- [ ] If Path A succeeded: `converter/convert_direct.py` exists and is runnable, and `converter/<model>_direct.xml` + `.bin` exist and are non-empty
- [ ] `converter/convert.py` (ONNX-path) exists and is runnable end-to-end
- [ ] `converter/<model_name>_simplified.xml` and `.bin` exist and are non-empty
- [ ] `benchmark/benchmark_cpu_result.txt` exists, produced by `benchmark_app -m <model> -d CPU -hint latency -infer_precision f16`
- [ ] `benchmark/benchmark_gpu_result.txt` exists, produced by `benchmark_app -m <model> -d GPU -hint latency -infer_precision f16`
- [ ] If Path A succeeded: `benchmark/benchmark_cpu_direct_result.txt` and `benchmark/benchmark_gpu_direct_result.txt` also exist
- [ ] If Path A succeeded: `benchmark/comparison_onnx_vs_direct.md` exists with latency table, op-graph counter diff, file-size diff, and a recommendation
- [ ] `benchmark/benchmark_app_usage.md` exists with the exact commands above and parameter explanations
- [ ] `validation/validate.py` exists and runs **both** comparisons in one process: A (OV-GPU-FP16 vs Source-CPU-FP16, tight) and B (OV-GPU-FP16 vs Source-CPU-FP32, loose)
- [ ] `validation/validation_results.json` contains per-input metrics under separate keys for A and B
- [ ] `validation/validation_report.md` reports both comparisons, applies the tight/loose threshold tiers correctly, and includes an explicit attribution sentence (conversion bug vs FP16 precision degradation) drawn from the diagnosis table
- [ ] Validation uses diverse inputs (1 real + N >= 10 synthetic across multiple distributions), not a single image
- [ ] Both reference precisions are spelled out in the report: same-precision FP16 baseline AND FP32 baseline. For optimization workflows, the FP16 baseline becomes the previously-validated OV IR.
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

### GitHub readiness (run as the final step)

- [ ] `export_<model_name>/.gitignore` exists, covering `.bin`, `.onnx`, pretrained-weight extensions used by this model, and `optimize_v2/custom_op/build/` if Stage 8 ran
- [ ] `scripts/fetch_assets.py` exists, with an explicit manifest (URL, dest path, SHA256, size, description) for every derived large file the user will need
- [ ] Running `fetch_assets.py` on a clean clone successfully restores every file the `.gitignore` excludes
- [ ] Pre-push size audit returned zero hits: no tracked file exceeds 95 MB (e.g. `git ls-files -z | xargs -0 stat -c "%s %n" | awk '$1>95000000'`)
- [ ] No Git LFS, no GitHub Releases, no chunk-splitting -- derived files are reproduced from source, not stored externally
- [ ] README includes both a "What's in this repo and what isn't" note AND `fetch_assets.py` as step 2 of the reproduce-from-scratch flow

### Conversion report

- [ ] Success: detailed step-by-step report with commands and outputs
- [ ] Report names which path(s) were attempted and whether each succeeded or failed, with the reason
- [ ] When both paths succeed, the report links to `benchmark/comparison_onnx_vs_direct.md` and states the chosen default IR
- [ ] Failure: `<model>_OpenVINO_conversion_failure_analysis.md` with all attempts, errors, and root cause
- [ ] On success, the user was asked whether they want to pursue optimization, with a recommendation to start with a per-layer IR profile (`references/per-layer-profiling.md`) if so

### Stage 8 (only if triggered -- see Section 7)

- [ ] Trigger conditions verified: source repo contains a hand-written CUDA kernel implementing a fused op AND Sections 1-6 all passed cleanly
- [ ] Both dependent skills (`vtune-profiler-skill` and `ov-custom-pipeline`) confirmed installed BEFORE any Stage 8 work; if either was missing, work was halted and the user was told which skill to install
- [ ] `validation/optimize_v2_feasibility/` exists with VTune `gpu-hotspots` output, target subgraph kernel summary, and a feasibility report stating the target subgraph's % of total latency
- [ ] User was explicitly asked whether to proceed -- and agreed -- before any custom-op code was written
- [ ] `optimize_v2/custom_op/` was authored via `ov-custom-pipeline` (not freelanced), with build instructions
- [ ] `optimize_v2/converter/convert_v2.py` produces `<model>_v2.xml` + `.bin`
- [ ] `optimize_v2/benchmark/benchmark_gpu_v2_result.txt` exists and `v2_vs_baseline.md` includes both a latency table and a per-kernel VTune diff against Section 7.3
- [ ] `optimize_v2/validation/validate_v2.py` re-runs **both** A and B comparisons, with the baseline OV IR as the FP16 reference; `validation_report_v2.md` applies the same A-tight / B-loose tiers and includes an explicit attribution sentence
- [ ] `optimize_v2/optimize_v2_report.md` ends with an honest verdict (keep v2 / keep baseline) -- not just a speedup number
