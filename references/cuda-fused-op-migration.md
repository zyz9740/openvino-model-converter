# CUDA Fused-Op Migration to OpenVINO Custom Op (Stage 8)

This is the detailed procedure for the optional Stage 8 referenced in
SKILL.md's Workflow Overview (step 8). Read it only when the trigger
conditions below hold. Throughout this file, "Section N" refers to the
correspondingly numbered section of the main SKILL.md (e.g. "Section 4"
is the numerical validation section), and "Stage 8" is this document.
Note that this file's "case A" / "case B" are unrelated to the main
conversion flow's "Path A" (direct export) / "Path B" (ONNX-intermediate)
from Sections 1-2 -- both can be in play here (e.g. a case-A run where
Path A already succeeded and a Path-B re-conversion is being optimized).

This stage exists because many high-performance model repos ship hand-written CUDA kernels that fuse multiple ops into one launch (e.g. a fused attention, a fused voxelization, a fused decoder head). On NVIDIA hardware this is a clear win. On Intel GPU through OpenVINO, the equivalent subgraph is whatever the IR's stock op decomposition gives us, which is rarely as tight. If the user is targeting Intel deployment, leaving that fusion on the table is exactly the kind of thing this skill should surface.

There are two distinct situations that lead here, and they are **not equally optional**:

- **A. Conversion-blocking case (check this first, higher priority).** The model's own conversion (Path A and Path B of the main conversion flow) both failed, and the documented root cause is that a hand-written CUDA kernel has no traceable ONNX/OV equivalent (e.g. the tracer errors out on the custom autograd `Function` wrapping the kernel, or ONNX export refuses the op). Here Stage 8 is not an optimization -- it's the only way to make the model convertible **at all**. Without it there is no IR, no benchmark, no validation, nothing to deliver. This is functionally required, not a nice-to-have, whenever it applies. The letter "A" here is a mnemonic for "blocked" / failing state, not related to the main flow's Path A.
- **B. Optimization case (check only after ruling out A).** Conversion already succeeded and validated cleanly; the custom CUDA op is a *performance* opportunity, not a blocker. This is genuinely opt-in -- the model already ships without it.

**Always check for case A before considering case B.** If conversion failed, the question is never "is this worth optimizing" -- there is nothing yet to optimize. Only once a working, validated IR exists does the case-B cost/benefit question (is the fusion opportunity big enough to justify the engineering time) become relevant at all.

This is **optional, opt-in work** for case B: it costs real time (kernel writing, integration, re-validation), and is only justified when the baseline OV conversion is otherwise healthy and the measured upside clears the bar in Section 3 below -- optimizing on top of a broken baseline just stacks two sources of error. Case A is **not** opt-in in the same sense: if it's the only way to get a convertible model, "skip it" means "the model can't be delivered," so once the trigger condition is confirmed, proceed unless the user has a reason not to (e.g. CPU-only deployment where the op was never the issue). There's no ROI threshold to clear on case A the way there is on case B -- the alternative is no deliverable at all.

## 1. Trigger conditions -- check case A first, then case B

**Step 1: rule out case A first.** Before considering whether this is an optimization opportunity, check whether conversion actually failed because of this CUDA op:

**Case A -- unblock a failed conversion (higher priority, check this first):**
- The cloned source repo contains hand-written CUDA / `__global__` / `cudaLaunchKernel` code that implements a **fused operation** (i.e. it does the work of multiple framework-level ops in a single kernel). Plain CUDA-via-PyTorch (e.g. `torch.bmm`) does **not** count -- that is already covered by stock OV ops. What counts is custom `.cu` / `.cuh` files in the repo, or `torch.utils.cpp_extension` / `setup.py` building a CUDA extension that the model's forward path calls into.
- Conversion failed (both the direct and ONNX-intermediate routes), and the failure analysis (`<model>_OpenVINO_conversion_failure_analysis.md`) traces the root cause specifically to that CUDA op -- e.g. the tracer can't step through it, or ONNX export has no opset mapping for it. If the failure is unrelated to the CUDA op (missing dependency, unsupported control flow elsewhere, etc.), fix that first; Stage 8 doesn't apply.

If both are true, this is case A: proceed straight to section 2, and treat the work as necessary rather than a discretionary optimization.

**Step 2: only if case A does not apply, consider case B:**

**Case B -- optimize an already-working conversion (lower priority, opt-in):**
- The same hand-written fused CUDA kernel test as case A holds.
- Sections 1-6 all passed: conversion succeeded (Path A and/or Path B of the main conversion flow), benchmarks ran, both validation comparisons (A: FP16 vs FP16, B: FP16 vs FP32) passed against their thresholds, and the demo runs.

If neither case applies, skip this stage entirely. Mention in the conversion report what you saw and why you did not run Stage 8 -- the user may want to revisit it later.

## 2. Required dependent skills -- check first, hard stop if missing

This stage **delegates** to two other skills. Before doing any work in Stage 8:

1. Check that **`vtune-profiler-skill`** is available (used for per-kernel GPU timing of the fused subgraph as it currently runs in OV -- **case B only**; skip this check on case A, there is no working OV run to profile yet).
2. Check that **`ov-custom-pipeline`** is available (used for the actual custom-op authoring -- it covers single-graph fusion, multi-model OCL orchestration, and SYCL interop, and decides which approach fits the kernel). Required on **both** cases.

If a required skill is missing, **stop immediately**. Do not attempt to write a custom op from scratch without the dependent skill -- the result will diverge from the user's other custom-op work and bypass the validated patterns those skills exist to encode. Tell the user exactly which skill is missing, point them at `skill-creator` to install or build it, and wait. Example wording:

> Stage 8 (CUDA -> OV custom-op migration) needs the `ov-custom-pipeline` skill, which is not currently installed. Please install it (or create it via `skill-creator`) before I continue. I have not started any custom-op work yet -- [the baseline conversion in Sections 1-6 is complete and unaffected. | conversion is currently blocked on this CUDA op and nothing else has been attempted yet.]

Do not work around the absence by reading documentation and freelancing the kernel. The whole point of the skill dependency is consistency.

## 3. Scoping the work: is it actually the blocker (case A) or worth porting (case B)?

The two trigger cases need different scoping before writing any code.

**Case A -- confirm the CUDA op is really the blocker, there is nothing to profile yet.** There is no working IR, so skip the VTune step entirely. Instead:

1. **Confirm the failure is specifically the CUDA op**, not something else. Re-check `<model>_OpenVINO_conversion_failure_analysis.md`: the tracer/export error should point at the module wrapping the custom kernel (by name, file, or stack frame), not an unrelated op or missing dependency.
2. **Identify the op sequence the kernel fuses** -- this is what `ov-custom-pipeline` will need to implement as a custom op.
3. Write a short `validation/optimize_v2_feasibility/feasibility_report.md` stating: which CUDA kernel blocks conversion, why (tracer error / no ONNX opset mapping / etc., quoting the actual error), and that a custom op is the proposed way to make the model convertible at all -- there is no baseline latency or budget percentage to report here, since nothing has been benchmarked yet.

**Case B -- measure first.** Before proposing the optimization, *measure*. A custom op that saves 50 microseconds out of a 30 ms inference is not worth the maintenance cost. Use `vtune-profiler-skill` to get per-kernel timing of the current OV-GPU run.

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

## 4. Decision gate -- ask the user before any code is written

After scoping, write a short feasibility report and **stop to ask the user**. Do not start writing a custom op proactively -- the user's explicit agreement is a hard gate on both cases regardless of how necessary the work is. The report and question differ by case, and so does how the recommendation should be framed:

**Case A wording** (note the different framing -- this isn't a cost/benefit call, it's "here's the only way forward"):

> Stage 8 feasibility check:
> - Conversion failed because `<kernel_name>.cu` (fusing ops X, Y, Z) has no traceable ONNX/OV equivalent -- specifically: `<quote the actual tracer/export error>`.
> - This is the only blocker identified; everything else in the model converts cleanly.
> - There is no alternative route to a working IR other than porting this kernel to an OpenVINO custom op -- I recommend proceeding.
>
> Do you want me to proceed? If yes, I will use `ov-custom-pipeline` to choose between single-graph fusion / multi-model OCL / SYCL interop, build under `export_<model_name>/optimize_v2/`, and then run Sections 2-5 (convert / benchmark / validate / demo) for the first time against the resulting IR -- there is no prior baseline to compare against, so `v2_vs_baseline.md` and the "keep v2 or keep baseline" framing in Section 5-6 don't apply; report against Sections 1-6's normal success criteria instead.

Even though case A is effectively required to get a deliverable, still wait for the user's explicit go-ahead before writing code -- they may prefer to stop at the failure report and pursue a different model, rather than invest in a custom op.

**Case B wording:**

> Stage 8 feasibility check:
> - The source repo's `<kernel_name>.cu` fuses ops X, Y, Z into one CUDA launch.
> - In the current OV-GPU IR, that subgraph is implemented as N stock ops, taking T ms / inference (P% of total latency T_total).
> - VTune shows the dominant cost is `<dominant kernel>` at K ms (occupancy O%, cache-hit C%, ...).
> - My recommendation: <port / skip / borderline>, because <one-sentence reason>.
>
> Do you want me to proceed with writing an OpenVINO custom op for this? If yes, I will use `ov-custom-pipeline` to choose between single-graph fusion / multi-model OCL / SYCL interop, build under `export_<model_name>/optimize_v2/`, and re-run Sections 2-5 (convert / benchmark / validate / demo) on the optimized version.

If the user declines (or says "later"), stop here. Save the feasibility report and move on. **Do not** start the optimization to "save time" -- the user's agreement is a hard gate.

If the user agrees, proceed to the next section.

## 5. Building `optimize_v2/`

Delegate the custom-op authoring entirely to `ov-custom-pipeline`. Its job is to pick between the three fusion strategies and produce the kernel + integration. This skill's job at this stage is the **harness around it**: directory layout, running/re-running Sections 2-5 with the new IR as input, and (case B only) producing a comparison report against the working baseline.

On case A there is no prior baseline IR, benchmark, or validation to compare against -- Sections 2-5 are running for the *first* time here, not being re-run. Treat `optimize_v2/` as where the first successful conversion lands, skip every "vs. baseline" comparison below, and report against Sections 1-6's normal pass/fail criteria instead of a speedup/regression framing.

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
      v2_vs_baseline.md              # case B only -- latency table + op-graph diff vs the baseline IR, plus the same
                                     # VTune gpu-hotspots run on v2 so per-kernel cost can be diffed against Section 3
    validation/
      validate_v2.py                 # SAME validation harness as Section 4; case B: v2 IR as the new "OV-GPU-FP16"
                                     # side, compared against the baseline OV IR; case A: v2 IR compared against the
                                     # original source framework, exactly like a normal Section 4 run
      validation_results_v2.json
      validation_report_v2.md        # both A and B comparisons, same diagnosis table
    demo/
      infer_demo_v2.py               # demo using v2 IR (can be a thin wrapper around the original)
    optimize_v2_report.md            # narrative: what was fused, which ov-custom-pipeline approach, case B: measured
                                     # speedup and an honest "was this worth it?" summary; case A: confirmation that
                                     # the model now converts and validates, with the root cause it unblocked
```

Re-running rules (case B -- re-running against a working baseline):

1. **Re-run Section 2 against the v2 IR only when it is meaningfully different from baseline.** Do not regenerate `<model>_simplified.xml` etc. -- those are stable.
2. **Re-run Section 3 (benchmark) on v2.** Save logs under `optimize_v2/benchmark/`. Compare against the original `benchmark/benchmark_gpu_result.txt` in `v2_vs_baseline.md`.
3. **Re-run Section 4 (validation, BOTH comparisons) on v2.** This is non-negotiable. A custom op is exactly the kind of change that can pass A (FP16 vs FP16) by chance on one input and explode on another. Use the same diverse input set as the baseline. Apply the same A-tight / B-loose threshold tiers. The reference for A becomes "the baseline OV IR at FP16" rather than the source framework -- you are now answering "did v2 stay equivalent to v1?" -- which is the right question for an optimization, per Section 4's note on optimized pipelines.
4. **Re-run Section 5 (demo)** on at least the real sample input, and visually confirm output matches baseline.

If validation on v2 fails (A fails -> custom op has a bug; B fails -> custom op introduced extra precision loss beyond stock FP16), do **not** silently fall back to baseline. Report the failure, keep v2 artifacts so the user can see what went wrong, and recommend either fixing the kernel via `ov-custom-pipeline` or abandoning the optimization. The user's trust depends on us being honest about when an optimization didn't pan out.

Running rules (case A -- first successful run, nothing to fall back to):

1. **Run Section 2 (convert) once**, producing `<model>_v2.xml`/`.bin` via `convert_v2.py`. There is no prior IR to diff against.
2. **Run Section 3 (benchmark) once** on GPU. There is no baseline log to compare it to -- report the numbers on their own, not as a delta.
3. **Run Section 4 (validation, BOTH comparisons) exactly as a normal first-time run**: reference is the original source framework (Source-CPU-FP16 for comparison A, Source-CPU-FP32 for comparison B), same as Sections 1-6 would use if conversion had succeeded the first time.
4. **Run Section 5 (demo) once**, same as a normal successful conversion.

If validation fails here, the model still doesn't have a clean path to deployment -- report it the same way Sections 1-6 would report a validation failure (attribution sentence: custom-op bug vs. FP16 precision), and recommend iterating on the custom op via `ov-custom-pipeline` rather than declaring victory on "it converts now."

## 6. Final report

Append a section to the main conversion report (or write `optimize_v2_report.md` and link it) covering:

**Case A:**
- Which CUDA kernel blocked conversion, with file paths in the source repo, and the specific tracer/export error it caused
- Which `ov-custom-pipeline` approach was chosen and why
- Confirmation that Sections 2-5 now run successfully against the resulting IR (there is no "vs baseline" delta to report -- there was no baseline)
- Validation status (A and B both), same attribution sentence as a normal Section 4 run
- Honest verdict: model is now convertible and deployable, or the custom op unblocked conversion but validation still fails and needs further iteration

**Case B:**
- Which CUDA kernel(s) were targeted, with file paths in the source repo
- VTune-measured baseline cost (Section 3 numbers above)
- Which `ov-custom-pipeline` approach was chosen and why
- v2 vs baseline: latency delta, op count delta, validation status (A and B both)
- Honest verdict: keep v2 as the recommended deployment IR, or keep baseline because v2 didn't beat it by enough / didn't validate / introduced too much precision loss

The verdict matters more than the speedup number. A 30% speedup that fails B is not a win; a 5% speedup that passes both and simplifies the graph might be.

## Stage 8 checklist

Before delivering `optimize_v2/` to the user, verify every item. Items marked (A) or (B) apply only to that case; unmarked items apply to both.

- [ ] Trigger conditions verified: source repo contains a hand-written CUDA kernel implementing a fused op, AND either (A) conversion failed and the failure analysis traces the root cause to that CUDA op, or (B) Sections 1-6 all passed cleanly and the user wants to optimize
- [ ] `ov-custom-pipeline` confirmed installed before any Stage 8 work; (B) `vtune-profiler-skill` also confirmed installed. If a required skill was missing, work was halted and the user was told which to install
- [ ] `validation/optimize_v2_feasibility/feasibility_report.md` exists; (A) states the quoted tracer/export error and confirms no unrelated blockers remain; (B) includes VTune `gpu-hotspots` output, target subgraph kernel summary, and % of total latency
- [ ] User was explicitly asked whether to proceed -- and agreed -- before any custom-op code was written
- [ ] `optimize_v2/custom_op/` was authored via `ov-custom-pipeline` (not freelanced), with build instructions
- [ ] `optimize_v2/converter/convert_v2.py` produces `<model>_v2.xml` + `.bin`
- [ ] `optimize_v2/benchmark/benchmark_gpu_v2_result.txt` exists; (B) `v2_vs_baseline.md` includes both a latency table and a per-kernel VTune diff against the baseline profiling
- [ ] `optimize_v2/validation/validate_v2.py` runs **both** A and B comparisons -- (A) against the original source framework, same as a normal first-time Section 4 run; (B) against the baseline OV IR as the FP16 reference; `validation_report_v2.md` applies the correct threshold tiers and includes an explicit attribution sentence
- [ ] `optimize_v2/optimize_v2_report.md` ends with an honest verdict -- (A) model now converts and validates, or still needs iteration; (B) keep v2 / keep baseline, not just a speedup number
