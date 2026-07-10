---
name: openvino-converter
description: >
  Convert mainstream AI models (PyTorch, TensorFlow, ONNX, YOLO series, point cloud models, etc.)
  to Intel OpenVINO IR format -- fast/token-lite workflow covering model acquisition, format
  conversion, GPU benchmarking, and a quick numerical validation pass. Use this skill whenever
  the user mentions model conversion to OpenVINO, deploying models on Intel GPU, IR format
  export, or converting any deep learning model for Intel inference -- even if they don't
  explicitly say "OpenVINO".
---

# OpenVINO Model Converter (Lite)

Convert AI models to OpenVINO IR fast: acquire source, convert, benchmark on GPU, quick-validate,
demo, deliver. This is the token-lite variant -- see **Limitations** below for what it trades away.

## Limitations (read before running)

This skill optimizes for speed and low token cost over exhaustive rigor:

- **Single validation comparison** (OV-GPU-FP16 vs the original framework's native-precision
  output) cannot cleanly separate a *conversion bug* from *expected FP16 precision loss*. A
  borderline result is ambiguous -- use your judgment or re-run with tighter manual checks.
- **Single conversion path** -- ships whichever of direct-export/ONNX succeeds first. Does not
  build both and compare, so it may not be the leanest/fastest possible IR for that model.
- **GPU-only benchmark** -- no CPU baseline, no cross-device comparison.
- **No auto weight download, no provenance/fetch_assets machinery** -- weight download is left
  to the user, but you must give them a concrete command they can run directly.
- **No CUDA fused-op migration path** -- if a model relies on custom CUDA kernels that block
  conversion, this skill reports the failure and stops; it does not attempt to port them.
- **Bail out on hard problems** -- at any step (convert, benchmark, validate, demo), if an error
  doesn't resolve after a couple of reasonable, targeted attempts, stop. Report the exact error,
  what was tried, and a best-guess root cause to the user instead of continuing to iterate. This
  skill is optimized to fail fast and cheap, not to grind through hard debugging.

If the user needs a hard pass/fail conversion gate, dual-path comparison, or CUDA-kernel
migration, say so explicitly rather than silently doing the lighter version.

## Workflow

1. **Acquire** -- clone source repo; give the user an executable weight download command, don't fetch weights yourself
2. **Convert** -- try direct export, fall back to ONNX; first success ships; both-fail = stop and report
3. **Benchmark** -- `benchmark_app` on GPU only
4. **Validate** -- OV-GPU-FP16 vs original framework, ~10 random inputs, no plots
5. **Demo** -- minimal inference script with real or random input
6. **Deliver** -- simple export directory + short report

## 1. Acquire

- Clone the model repo (`--depth 1` if large) into `<model_name>/` under the export directory.
- Search the repo's README/releases for official pretrained weights. **Do not download them
  yourself.** Give the user the direct URL, approximate file size, where to place the file, and
  one explicit copy/paste command that downloads the weight file(s) into that exact location --
  then wait. No polling, no retries, no background waiting.
- Prefer commands that work in the user's current shell/OS. On Windows PowerShell, use
  `Invoke-WebRequest -Uri "<url>" -OutFile "<absolute_weight_path>"` for direct files, or
  `huggingface-cli download <org>/<model> --local-dir "<absolute_weights_dir>" --local-dir-use-symlinks False`
  for HuggingFace repos. On Linux/macOS, use `curl -L "<url>" -o "<absolute_weight_path>"`
  for direct files, or the same `huggingface-cli download ...` form for HuggingFace repos.
- If weights are on HuggingFace, include the mirror setup inline before the download command
  when useful (for example, `$env:HF_ENDPOINT = "https://hf-mirror.com"` on PowerShell or
  `export HF_ENDPOINT=https://hf-mirror.com` on Linux/macOS), and point the user at
  `references/hf-mirror-guide.md` for details.
- The user-facing instruction must include a fenced command block and be immediately runnable,
  for example:

  ```powershell
  New-Item -ItemType Directory -Force "<absolute_weights_dir>"
  Invoke-WebRequest -Uri "<direct_weight_url>" -OutFile "<absolute_weight_path>"
  ```

  or:

  ```powershell
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  huggingface-cli download <org>/<model> --local-dir "<absolute_weights_dir>" --local-dir-use-symlinks False
  ```
- If no weights are available, proceed with random input for validation/demo and say so.

## 2. Convert

Always target FP16. Try paths **in priority order** and ship the first one that works --
do not build both.

1. **Direct export** (preferred): `openvino.convert_model(model, example_input=...)` from the
   loaded torch/TF module, then `ov.save_model(ov_model, "<model>.xml", compress_to_fp16=True)`.
   Check the repo for its own export script first and reuse it if present.
2. **ONNX fallback** (only if direct export raises): export to ONNX, optionally run `onnxsim`,
   then `openvino.convert_model("<model>.onnx")` / `ovc`.
3. **Both fail** -- this is a hard stop. Do not try workarounds indefinitely. Write the failure
   report (Section 5) with every attempted approach, the exact error, and a root-cause guess
   (e.g. "custom CUDA kernel with no CPU/ONNX equivalent"), then tell the user conversion did
   not succeed. This is the expected outcome for CUDA-only fused ops -- do not attempt to port them.
   (This is the Convert-step instance of the general "bail out on hard problems" rule above.)

Use `scripts/example_convert.py` as a starting template. Write the result as one
`converter/convert.py`. Handle dynamic shapes by setting explicit static shapes if OpenVINO
rejects the dynamic ones.

## 3. Benchmark (GPU only)

```bash
benchmark_app -m <model.xml> -d GPU -hint latency -infer_precision f16
```

Save the full log to `benchmark/benchmark_gpu_result.txt`. Read median latency and throughput
straight off the log when reporting to the user -- don't build a separate parser. No CPU run,
no throughput-hint run, unless the user explicitly asks for one.

If `benchmark_app` fails for a non-obvious reason (driver/plugin issue, unsupported layer at
runtime) and a quick fix doesn't work, don't debug the GPU stack -- report it and stop.

## 4. Validate (fast)

Write `validation/validate.py` that runs the OV-GPU-FP16 IR and the original framework
(native precision, CPU is fine) on the **same inputs** and compares outputs in-process.

- **Inputs**: 1 real sample if one is trivially available, plus **~10 random tensors** with a
  fixed seed matching the model's input shape/dtype. That's enough to catch shape/dtype bugs
  and gross divergence -- this is not meant to be an exhaustive statistical study.
- **Metrics per input**: `max_abs`, `mean_abs`, NaN count. No plots, no percentile breakdowns,
  no per-stage attribution.
- **Verdict**: any NaN = fail. Large max_abs relative to the output's natural scale = flag it
  as a difference and let the user judge whether it's a real conversion bug or FP16 precision
  (see Limitations) -- don't invent a universal numeric threshold.
- Write a short `validation/validation_report.md`: per-input numbers + one-paragraph verdict.
  No `.json` dump required unless the user wants machine-readable output.
- If `validate.py` itself errors (env/import mismatch, shape mismatch) and a quick fix doesn't
  work, report it rather than reworking the harness repeatedly.

## 5. Demo

Write `demo/infer_demo.py` that runs out of the box on the real sample (if available) or a
generated input, and show the result to the user. Keep it short -- this is a quickstart, not a
benchmark harness. If it won't run after a quick fix attempt, report the error rather than
endlessly patching it.

## 6. Deliver

```
export_<model_name>/
  <model_name>/              # cloned source (weights excluded -- user downloads separately)
  converter/
    convert.py                # whichever path succeeded
    <model_name>.xml / .bin   # OpenVINO IR (FP16)
  benchmark/
    benchmark_gpu_result.txt
  validation/
    validate.py
    validation_report.md
  demo/
    infer_demo.py
  README.md                   # setup + how to run each step
  <model>_conversion_report.md          # on success: steps + key results
  <model>_conversion_failure_analysis.md # on failure: attempts + root cause (see 2.3)
```

README should cover: environment setup, where to get weights (link + placement + executable
download command, per Section 1), how to run `convert.py`, `benchmark_app`, `validate.py`, and
`infer_demo.py`. Keep it short.

## Platform Notes

- **Encoding (Windows)**: stick to ASCII-printable characters in output -- GBK chokes on emoji/Unicode.
- **Paths**: use absolute paths for file I/O; relative paths break across working-directory changes.

## Bundled Resources

- **`scripts/example_convert.py`** -- starting template for `converter/convert.py` (direct path,
  ONNX fallback).
- **`references/hf-mirror-guide.md`** -- HuggingFace mirror setup, for when the user downloads
  weights themselves.

## Completion Checklist

- [ ] `export_<model_name>/<model_name>/` has the cloned source; weight download link and a
  directly executable download command (not the file) were handed to the user if weights are
  large/remote
- [ ] `converter/convert.py` exists and produces `<model>.xml` + `.bin` (FP16), OR a failure
      report was written and the user was told conversion did not succeed
- [ ] `benchmark/benchmark_gpu_result.txt` exists from a GPU `-hint latency -infer_precision f16` run
- [ ] `validation/validate.py` ran ~10+ inputs, comparing OV-GPU-FP16 vs the original framework,
      and `validation_report.md` states a verdict
- [ ] `demo/infer_demo.py` runs and its output was shown to the user
- [ ] `README.md` covers setup, weight acquisition with an executable command,
      convert/benchmark/validate/demo steps
- [ ] Conversion report (success) or failure analysis (failure) was written
