# Per-layer IR profiling with `benchmark_app`

`benchmark_app` (Section 3) reports end-to-end latency/throughput -- it
does not tell you *which layer* in the IR is consuming that time. When
the user wants to optimize a converted model (as opposed to just
shipping the baseline numbers), the next question is always "where does
the time actually go inside the graph?". `benchmark_app` can answer
that directly, with no extra tooling, via two flags.

This is a lighter-weight alternative to a full VTune `gpu-hotspots`
collection (see the `vtune-profiler-skill`'s Capability 5, `exec_graph`)
when the question is per-OpenVINO-layer time and kernel choice, not
per-GPU-kernel occupancy/cache/bandwidth. No VTune installation, no SEP
driver, no Python -- just a re-run of `benchmark_app` with one extra
flag.

## The two flags

| Flag | Output | Use when |
|------|--------|----------|
| `-pc` | Prints a per-layer table straight to stdout/log (layer name, exec status, layer type, exec type, time (ms)) | Quick look, no need to keep the data structured |
| `-exec_graph_path <file>.xml` | Serializes the full post-compilation runtime graph to XML, one node per compiled layer, with `execTimeMcs`, `primitiveType` (chosen kernel/impl), `outputLayouts`, `outputPrecisions`, `originalLayersNames` | Need to parse, diff, or archive the per-layer breakdown |

**Do not pass both together.** Per-layer timing is automatically
enabled by `-exec_graph_path`; adding `-pc` on top only piles on extra
instrumentation overhead with no additional signal. Pick one based on
whether you need structured output.

## Recommended command

Default to GPU only -- it's the deployment target this skill optimizes
for, and profiling both devices by default doubles the runs for
information that's usually not needed:

```bash
benchmark_app -m <model.xml> -d GPU -hint latency -infer_precision f16 \
    -niter 50 -exec_graph_path exec_graph.xml
```

Keep the same `-hint latency -infer_precision f16` contract as the
standard benchmark run (Section 3) so the per-layer numbers are
comparable to the headline latency figure. `-niter 50` gives the
runtime enough iterations to warm up GPU shader compilation before the
graph is captured.

Only add a CPU capture if the user explicitly asks for a CPU/GPU
per-layer comparison:

```bash
benchmark_app -m <model.xml> -d CPU -hint latency -infer_precision f16 \
    -niter 50 -exec_graph_path exec_graph_cpu.xml
```

Outputs:
- `exec_graph.xml` -- the useful artifact, one `<layer>` per compiled node
- `exec_graph.bin` -- empty placeholder, ignore it (not real weights)

## Alternative: Python API with `PERF_COUNT`

Use `benchmark_app` by default -- it's simpler and needs no code. Fall
back to the Python API only when the user needs to compile against a
shared GPU context, or wants to mirror what their own inference
pipeline does rather than a standalone `benchmark_app` process:

```python
import openvino as ov
from openvino import Core

core = Core()
model = core.read_model("model.xml")

cfg = {
    "PERFORMANCE_HINT": "LATENCY",
    "INFERENCE_PRECISION_HINT": "f16",
    "PERF_COUNT": "YES",   # without this, every execTimeMcs is 0
}

compiled = core.compile_model(model, "GPU", cfg)
req = compiled.create_infer_request()

# Warm up -- first inferences include JIT compilation
for _ in range(15):
    req.infer()

# Only the LAST inference's per-layer timing is captured
ov.save_model(compiled.get_runtime_model(), "exec_graph.xml")
```

`PERF_COUNT=YES` is mandatory -- without it the XML still serializes
fine but every `execTimeMcs` is `0`. The resulting `exec_graph.xml` has
the same per-layer attributes and is read the same way as the
`benchmark_app` output above.

## What each layer's `<data>` carries

| Attribute | Meaning |
|-----------|---------|
| `execOrder` | Execution sequence number (0-based) |
| `execTimeMcs` | Execution time in **microseconds**, from the **last** captured inference only |
| `primitiveType` | Kernel/implementation the plugin picked (e.g. `gen_conv`, `permute_ref__f16`, `jit:ir__f16`) |
| `outputLayouts` | Memory layout (`b_fs_yx_fsv16`, `bfyx`, `bfzyx`, ...) |
| `outputPrecisions` | Runtime precision (`f16`, `f32`) |
| `originalLayersNames` | Layer name(s) from the source IR this compiled node maps back to |

## Reading the results

1. Sort layers by `execTimeMcs` descending -- the top few entries are
   almost always where optimization effort should go. Report the top
   10 by name, type, and % of total inference time.
2. Sum `execTimeMcs` across all layers and compare against the
   headline `benchmark_app` latency. The gap is host dispatch overhead
   (not captured per-layer) -- typically a few milliseconds.
3. Group by `originalLayersNames` prefix or by layer type
   (`Convolution`, `MatMul`, `Reorder`, `Concat`, ...) with
   `collections.Counter`-style aggregation to see which *op family*
   dominates, not just which single node.
4. Note `primitiveType` for the hottest layers -- it records which
   kernel implementation the GPU plugin selected for each op. This is
   useful context when correlating a layer's cost with its shape,
   layout, and precision; report it alongside the timing rather than
   reading any single kernel name as good or bad on its own.

## Pitfalls

1. **Single-inference capture.** `execTimeMcs` reflects only the last
   inference in the run, not an average. `-niter 50` (or higher)
   ensures the capture happens after warm-up, not during first-call
   JIT/shader compilation.
2. **Don't compare `-pc` and `-exec_graph_path` runs to each other for
   absolute timing** -- both add some instrumentation overhead, so
   relative (A vs. B, e.g. before/after an optimization) comparisons
   are valid but absolute numbers should be cross-checked against the
   uninstrumented benchmark log. Measure the overhead on your own model
   if you need a concrete figure; it varies with graph size and device.
3. **`exec_graph.bin` is not real weights** -- do not attempt to load
   it as a standalone model.
4. **Thermal/cache state** -- compare two exec_graph captures collected
   in the same session; GPU frequency and cache residency drift across
   cold/hot states and can shift per-layer numbers independent of any
   code change.

## When to go further: VTune `exec_graph` / `gpu-hotspots`

This `benchmark_app`-only method is sufficient for "which layer is
slow and what kernel did it get" questions. If the user needs any of
the following, hand off to the `vtune-profiler-skill` instead (its
Capability 5 covers the same `exec_graph` artifact with parse/compare
tooling, and Capability 2 covers GPU kernel-level occupancy/cache/
bandwidth that `exec_graph` cannot show):

- A structured diff between two exec_graph exports (`compare`
  subcommand, top-N delta by layer, aggregate delta by op type)
- Per-GPU-kernel occupancy, cache hit-rate, or memory bandwidth
  (`gpu-hotspots` characterization -- a different, lower-level
  granularity than exec_graph)
- Per-source-line GPU latency inside a custom SYCL/DPC++ kernel

This decision also feeds directly into Section 7 (Stage 8): the
per-layer profile here is what identifies *candidate* hot layers, and
7.3's `gpu-hotspots` analysis is the deeper measurement used once a
custom-op migration is actually on the table.
