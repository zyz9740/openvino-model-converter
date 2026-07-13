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
demo, deliver reproducibly. This is the token-lite variant -- see **Limitations** below for what
it trades away (single conversion path, GPU-only benchmark, single validation comparison, no CUDA
migration). File-management and reproducibility discipline (git hygiene, `.gitignore`,
`fetch_assets.py`, size audits) is kept in full -- that part isn't where the token cost was.

## Limitations (read before running)

This skill optimizes for speed and low token cost over exhaustive rigor:

- **Single validation comparison** (OV-GPU-FP16 vs the original framework's native-precision
  output) cannot cleanly separate a *conversion bug* from *expected FP16 precision loss*. A
  borderline result is ambiguous -- use your judgment or re-run with tighter manual checks.
- **Single conversion path** -- ships whichever of direct-export/ONNX succeeds first. Does not
  build both and compare, so it may not be the leanest/fastest possible IR for that model.
- **GPU-only benchmark** -- no CPU baseline, no cross-device comparison.
- **No auto weight download during the session** -- the skill writes a `scripts/fetch_assets.py`
  manifest (URL, dest path, sha256, size) for weights and large test data, but the user runs it
  themselves; the skill never downloads or polls in-session. File-management/reproducibility
  policy (`.gitignore`, committed-vs-derived, push-size audit) is otherwise unchanged from the
  full skill -- see Section 6.
- **No CUDA fused-op migration path** -- if a model relies on custom CUDA kernels that block
  conversion, this skill reports the failure and stops; it does not attempt to port them.
- **Bail out on hard problems** -- at any step (convert, benchmark, validate, demo), if an error
  doesn't resolve after a couple of reasonable, targeted attempts, stop. Report the exact error,
  what was tried, and a best-guess root cause to the user instead of continuing to iterate. This
  skill is optimized to fail fast and cheap, not to grind through hard debugging.

If the user needs a hard pass/fail conversion gate, dual-path comparison, or CUDA-kernel
migration, say so explicitly rather than silently doing the lighter version.

## Workflow

1. **Acquire** -- clone source repo (de-submodule it), write `fetch_assets.py` for weights, don't fetch them yourself
2. **Convert** -- try direct export, fall back to ONNX; first success ships; both-fail = stop and report
3. **Benchmark** -- `benchmark_app` on GPU only
4. **Validate** -- OV-GPU-FP16 vs original framework, ~10 random inputs, no plots
5. **Demo** -- minimal inference script with real or random input
6. **Deliver** -- simple export directory + short report

## 1. Acquire

### Source code

- Clone the model repo (`--depth 1` if large) into `<model_name>/` under the export directory.
- Cloning leaves a nested `.git` inside `<model_name>/`. Remove it
  (`rm -rf <model_name>/.git`) once cloned -- otherwise, once the export directory itself becomes
  a git repo, that nested `.git` turns into an accidental embedded-repo/submodule reference that
  breaks on a fresh clone elsewhere. Record the source commit hash/tag in the conversion report
  instead of relying on git to track it.

### Pretrained weights

- Search the repo's README/releases for official pretrained weights. **Do not download them
  yourself and do not poll/retry in-session.** Instead, add an entry to
  `scripts/fetch_assets.py`'s manifest (copy `scripts/fetch_assets_template.py`, fill in `url`,
  `dest_path`, `size_bytes`, `description`; if HuggingFace-hosted, use the hf-mirror.com URL as
  `url` and the original huggingface.co URL as an `alternate_urls` entry -- see
  `references/hf-mirror-guide.md`).
- `sha256`: if the source publishes an official checksum, use it. Otherwise leave a placeholder
  (e.g. `"UNVERIFIED-fill-after-first-download"`) and tell the user the script will print the
  actual hash on first run so they can lock it into the manifest.
- Tell the user to run `python scripts/fetch_assets.py` themselves to fetch the weights (and any
  large test inputs) -- then wait. This is the one command they need; don't also invent an ad hoc
  curl/Invoke-WebRequest one-off.
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
   report (Section 6) with every attempted approach, the exact error, and a root-cause guess
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

The export directory is meant to become a git repo pushed to GitHub -- a fresh `git clone` plus
`fetch_assets.py` plus `convert.py` must let *the user* reproduce everything later. The skill
itself never runs `fetch_assets.py` (or waits on a download) during the session -- it only
authors the manifest and hands the one command off, per Section 1.

```
export_<model_name>/
  .gitignore                       # excludes derived/large files (see below)
  requirements.txt
  scripts/
    fetch_assets.py                # weight + large-test-data manifest, SHA256-verified (Section 1)
  <model_name>/                    # cloned source, nested .git removed (Section 1)
  converter/
    convert.py                     # whichever path succeeded
    <model_name>.xml               # OpenVINO IR graph -- COMMIT (text, small)
    <model_name>.bin               # (derived) IR weights -- NOT committed; rebuild with convert.py
  benchmark/
    benchmark_gpu_result.txt
  validation/
    validate.py
    validation_report.md
  demo/
    infer_demo.py
  README.md
  <model>_conversion_report.md          # on success: steps + key results
  <model>_conversion_failure_analysis.md # on failure: attempts + root cause (see Section 2)
```

### Committed vs. derived

Never commit a file that `fetch_assets.py` or `convert.py` can reproduce -- GitHub rejects any
file over 100 MB and a bloated history is painful to fix later.

| Category | Examples | Treatment |
|----------|----------|-----------|
| Source, configs, reports, logs | `.py`, `requirements.txt`, `.md`, `.txt`, `.json` | Commit |
| OV IR graph | `.xml` | Commit (text, usually small) |
| Small test inputs | sample images < 95 MB | Commit |
| OV IR weights | `.bin` | **Derived** -- rebuilt by `convert.py` |
| ONNX intermediate | `.onnx` | **Derived** -- rebuilt by `convert.py` |
| Pretrained weights | `.pt`/`.pth`/`.ckpt`/`.h5`/`.safetensors` | **Derived** -- fetched by `fetch_assets.py` |
| Large test inputs | any file > 95 MB | **Derived** -- fetched by `fetch_assets.py` |

Write `.gitignore` at the export root covering `*.bin`, `*.onnx`, and whatever pretrained-weight
extensions this model uses (`*.pt`, `*.pth`, `*.ckpt`, `*.h5`, `*.safetensors`, ...), plus the
usual `__pycache__/`, `*.pyc`, `.venv/`.

Before telling the user it's ready to push, run a size audit and fix `.gitignore` if anything
over 95 MB comes back (don't fight it at the server with `--force`):

```bash
git ls-files -z | xargs -0 -I{} stat -c "%s %n" "{}" | awk '$1 > 95000000 {print}'
```

No Git LFS (bandwidth-capped on free accounts) and no GitHub Releases assets (a second source of
truth that drifts) -- the fetch-from-source + convert-from-source approach is the only path.

README should cover: environment setup, running `scripts/fetch_assets.py` to restore weights/large
inputs, running `convert.py`, `benchmark_app`, `validate.py`, and `infer_demo.py`, and a short
"what's in this repo and what isn't" note pointing at `fetch_assets.py`/`convert.py`. Keep it short.

## Platform Notes

- **Encoding (Windows)**: stick to ASCII-printable characters in output -- GBK chokes on emoji/Unicode.
- **Paths**: use absolute paths for file I/O; relative paths break across working-directory changes.

## Bundled Resources

- **`scripts/example_convert.py`** -- starting template for `converter/convert.py` (direct path,
  ONNX fallback).
- **`scripts/fetch_assets_template.py`** -- copy into each export's `scripts/fetch_assets.py` and
  fill in the manifest (URL, dest path, sha256, size, description) per Section 1/6. Implements
  SHA256 verification and skip-if-present.
- **`references/hf-mirror-guide.md`** -- HuggingFace mirror setup, for when the user downloads
  weights themselves via `fetch_assets.py`.

## Completion Checklist

- [ ] `export_<model_name>/<model_name>/` has the cloned source with its nested `.git` removed
- [ ] `scripts/fetch_assets.py` exists with a filled-in manifest for weights/large test data;
      the skill did not download them itself
- [ ] `converter/convert.py` exists and produces `<model>.xml` + `.bin` (FP16), OR a failure
      report was written and the user was told conversion did not succeed
- [ ] `benchmark/benchmark_gpu_result.txt` exists from a GPU `-hint latency -infer_precision f16` run
- [ ] `validation/validate.py` ran ~10+ inputs, comparing OV-GPU-FP16 vs the original framework,
      and `validation_report.md` states a verdict
- [ ] `demo/infer_demo.py` runs and its output was shown to the user
- [ ] `.gitignore` excludes `.bin`/`.onnx`/pretrained-weight extensions; a size audit
      (`git ls-files` + `stat` over 95 MB) came back empty before telling the user to push
- [ ] `README.md` covers setup, `fetch_assets.py`, convert/benchmark/validate/demo steps, and
      a short "what's in this repo and what isn't" note
- [ ] Conversion report (success) or failure analysis (failure) was written
