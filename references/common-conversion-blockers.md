# Common Conversion Blockers and Patch Patterns

This is the detailed reference for Section 2's "When conversion fails or
misbehaves" subsection. Read it whenever Path A and/or Path B fail, or
succeed but misbehave (NaN, wrong output shape, GPU-only compile failure)
-- before concluding a model "can't be converted" or reaching for Stage 8
(Section 7). Most conversion blockers fall into a handful of recurring
categories, and most are fixed without touching the vendored model source
at all. Only reach for an actual source patch when the lighter options
below don't apply.

## Fix-order ladder -- try these before patching source

Patching a cloned/submodule repo is the **last** resort, not the first. Work
down this list; stop at the first rung that resolves the failure:

1. **External wrapper module** -- wrap the model in a small `torch.nn.Module`
   that adapts inputs/outputs or drops an unused code path. No upstream file
   is touched. This is almost always tried first and is the most common fix
   by far in practice.
2. **Runtime monkeypatch in the converter script** -- reassign a function
   (`torch.some_op = replacement`) right before tracing/export, scoped to the
   conversion process only. The cloned repo on disk is never modified.
3. **Exporter flag / environment toggle** -- switch `dynamo=True/False`,
   set an env var the upstream code already checks (e.g. to disable an
   optional fast-path dependency), change opset version, or load a
   pre-existing TorchScript checkpoint the authors shipped instead of
   tracing the eager module.
4. **Load only the needed class via `importlib`/AST extraction** -- bypass a
   package's `__init__.py` (which may pull in training-only dependencies)
   by importing the target module/class directly.
5. **Formal source patch (`patches/NNN-description.patch`)** -- only when
   none of the above apply, i.e. the op/logic that breaks conversion lives
   inside the vendored source itself and there's no way to route around it
   from outside. See "Formal patch policy" below.

Rungs 1-4 leave the cloned submodule/repo byte-for-byte identical to
upstream -- `git status` inside it stays clean. That is the default and
preferred outcome. Reach for rung 5 only when the fix genuinely cannot be
expressed as a wrapper, monkeypatch, flag, or import trick.

## Index

| # | Problem | Typical fix |
|---|---------|--------------|
| 1 | `forward()` returns a dict (or dict-of-lists) -- tracer can't map string keys to graph outputs | Wrapper indexes into the dict/list and returns a plain tuple; keep `output_names` positionally in sync |
| 2 | A CUDA-only implementation is what's blocking export -- either a compiled `_C` extension (NMS, ROIAlign, deformable conv) fails to build/import, or an optional accelerated backend (xformers, flash-attention) isn't traceable/installed | Check whether the model/library already exposes a config flag or env var selecting a pure-PyTorch equivalent of the *same op* (`mat_impl="pytorch_1d"` in `export_matchattention`; `XFORMERS_DISABLED=1` in `export_dinov2`) -- zero source changes if so; only patch `__init__.py`'s import with try/except as a last resort when no such switch exists |
| 3 | Old repo uses APIs removed by modern NumPy/PyTorch (`np.float`, `torch.distributed.deprecated`, ...) -- fails before conversion even starts | Small, targeted source patch per broken call site |
| 4 | A specific op (e.g. `nan_to_num`) has no ONNX/OV mapping or isn't traced correctly, and no ready-made alternate implementation exists | Monkeypatch the op with a hand-built equivalent composed from primitive, well-supported ops (e.g. `isfinite` + `where`); verify edge-case behavior matches the original |
| 5 | Inference produces NaN only under FP16 (FP32 is clean) -- usually a tiny `clamp`/epsilon (e.g. `F.normalize(eps=1e-12)`) underflowing FP16's normal-number floor | Patch the `Clamp.min` attribute directly on the exported IR (e.g. `1e-12` -> `1e-4`); verify minimally, don't over-wrap in FP32 |
| 6 | IR loads and compiles on CPU but `compile_model(..., "GPU")` fails on a specific 5D reshape/select/scatter pattern (e.g. `export_matchattention`'s windowed-attention coordinate/window ops) | Formal patch re-expressing the same math in 4D only (split a stacked-then-selected tensor into independent tensors; drop artificial trailing width-1 dims; replace 5D scatter-add with `F.pad` + `add`); verify numerical equivalence, not just "it compiles" |
| 7 | `torch.onnx.export(dynamo=True)` fails (duplicate weight names from a reused submodule, Windows Unicode error in its own error printer) | Pass `dynamo=False` |

## Category 1: Non-tensor / structured outputs

**Symptom:** `forward()` returns a `dict`, or a tuple/list nested inside a
dict. `torch.onnx.export` / `torch.jit.trace` cannot map string dict keys to
graph output edges -- export either raises during tracing or silently
produces a malformed graph.

**Fix (rung 1, wrapper):** wrap the model and manually index into the dict,
returning a plain tuple of tensors. If the dict holds a list per key (e.g. a
per-iteration list from a refinement/RAFT-style model), also slice down to
the entries actually needed for deployment (usually just the last
iteration) -- there is no way to preserve "iteration index" as IR metadata,
so decide explicitly what to keep and drop the rest.

```python
class FlowExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image1, image2):
        out = self.model(image1, image2)   # {"flow": [...], "info": [...]}
        return out["flow"][-1], out["info"][-1]
```

**Do not rely on key names surviving into the IR.** The dict keys
(`"flow"`, `"info"`) are gone the moment the wrapper returns a tuple --
whatever names appear on the final ONNX/OV outputs come **only** from
`output_names=[...]` passed to `torch.onnx.export`, which is a manually
maintained list positionally aligned to the wrapper's `return` statement.
If the wrapper's return order ever changes, `output_names` must be updated
in the same commit, or the IR will have silently mislabeled outputs with no
error at export time. Note this coupling explicitly in the converter script
with a comment, and double check it whenever the wrapper's forward changes.

**Contrast with plain multi-output models:** a `forward()` that returns a
plain tuple (not a dict) generally exports fine without a wrapper -- ONNX
natively supports multiple outputs. A wrapper is still worth adding there,
but for interface cleanliness (trim to the one output deployment needs),
not because export would otherwise fail. Don't conflate the two cases in
the conversion report -- one is "required to make export succeed", the
other is "done for a cleaner deployment contract."

## Category 2: CUDA-only implementations with a switchable fallback

**Symptom:** the vendored repo imports a compiled `_C` extension (built via
`torch.utils.cpp_extension` / `setup.py`) for ops like NMS, ROIAlign,
ROIPool, deformable conv, or a fused custom kernel -- and the extension
either fails to build at all on the conversion machine (wrong torch ABI,
no CUDA toolchain, Windows), or builds but obviously can't trace through
OV/ONNX. A closely related variant of the same underlying situation: the
model doesn't fail to *import* anything, but a CUDA-optimized code path
(a fused attention kernel, an accelerated attention *library* like
xformers/flash-attention) is what actually runs, and *that* is what
doesn't trace -- the failure mode differs (import-time vs. trace-time) but
the fix category is the same, because in both cases the repo already
contains (or can already select) a non-CUDA implementation of the same
op.

**Check first: is the op actually reachable from the exported forward
path?** Many detection/segmentation repos import these extensions in
`layers/__init__.py` at package-load time even when the specific model
variant being exported never calls them (e.g. BlendMask doesn't use
`BezierAlign`/`DefROIAlign`; RetinaMask's classification/regression heads
don't need the CUDA NMS layer to produce raw output tensors -- NMS is
normally applied as a post-processing step outside the traced graph
anyway). If the op is unreachable, the failure is a load-time
`ImportError`, not an export-time limitation.

**Check second, before patching: does the model already ship a
non-CUDA implementation of the same op behind a config flag?** Some repos
implement the expensive op twice -- once as a fused CUDA kernel for
training/production speed, once as a plain-PyTorch reference/fallback
implementation of the *same math* -- and expose a constructor argument or
config field that picks which one runs. If so, this is rung 3 (exporter/
config flag) and is strictly preferable to rung 5: no source is touched,
and the two implementations are already numerically equivalent by the
repo's own design (the PyTorch path usually exists precisely so training
can run on CPU-only dev boxes, or as a correctness reference for the CUDA
kernel), so there's no behavior-change risk the way there is with
"unreachable op -> stub it out."

Concretely, in `export_matchattention`, `MatchAttentionBlock`/`MatchAttentionOp`
take a `mat_impl` argument with values `cuda` (the actual fused-CUDA-kernel
path, `match_attention::fused_forward_ops`, backed by a `.cu` extension --
un-exportable) vs. `pytorch_1d` / `pytorch_y1d` / `pytorch` (pure-PyTorch
implementations of the identical windowed-correlation/attention math, in
`models/mat_pytorch_impl.py`). The converter simply constructs the model
with `mat_impl="pytorch_1d"` instead of `"cuda"` -- no patch file, no
wrapper, just picking the export-friendly branch the authors already wrote:

```python
# converter/build_model.py
args = SimpleNamespace(variant=variant, mat_impl="pytorch_1d", field_up_only=True)
model = MatchAttention(args)
```

Before assuming Category 2 requires a formal patch, grep the repo for the
CUDA extension's call sites and check whether they're gated by a
constructor arg, config field, or similarly-named plain-Python sibling
function (`*_pytorch`, `*_native`, `*_reference`, `*_fallback`) -- it's a
common enough pattern in performance-oriented repos (attention, cost-volume/
correlation, voxelization, deformable ops) that it's worth checking before
reaching for rung 5.

**Same pattern, library-level instead of model-level: `export_dinov2`.**
DINOv2 conditionally uses `xformers`' memory-efficient attention when the
package is importable, falling back to plain
`torch.nn.functional.scaled_dot_product_attention` otherwise -- the
fallback already exists in the upstream code for non-GPU / no-xformers
environments, it doesn't need to be written. Setting `XFORMERS_DISABLED=1`
(an env var the upstream code already checks) before model construction
selects the traceable fallback with zero code changes:

```python
import os
os.environ["XFORMERS_DISABLED"] = "1"   # forces the plain SDPA fallback path
```

Whether the switch is a model constructor argument (`mat_impl=`) or an
environment variable the library checks at import time (`XFORMERS_DISABLED`),
the principle is identical: **check for an existing disable switch before
writing any patch or wrapper.** This is rung 3 (exporter/environment
toggle) territory, not rung 5 -- always check for it first.

**This config-switch fix is the right stopping point for getting the
model converted -- don't reach for a custom kernel here.** Selecting the
existing PyTorch-equivalent implementation is sufficient for baseline
conversion/delivery; do not, on your own initiative, go further and author
an OpenCL/SYCL kernel for the original CUDA op just because one exists.
That is a separate, opt-in performance track (Section 7's custom-op
migration): once the fallback-based IR converts and validates, mention to
the user that the CUDA op you bypassed is a candidate for a hand-written
OV custom op if they want to close the performance gap later, and only
start that work if they say yes.

**Fix (rung 5, formal patch -- only when no fallback config exists):** wrap
the failing import in `try/except` inside the package's `__init__.py`, so
the module loads even when the extension isn't built:

```python
# patches/001-layers-init-skip-C-extension.patch (example)
try:
    from .roi_align import ROIAlign
    from .nms import nms
except ImportError:
    ROIAlign = None
    nms = None
```

This has to be a real source patch (not a wrapper) because the failure
happens at **import time**, before any wrapper code gets a chance to run --
you can't monkeypatch a module that fails to import in the first place.
Keep the patch minimal: only guard the specific import, don't restructure
the file. If the op genuinely IS reachable from the exported path (e.g. a
custom deformable-conv actually used in the forward pass), this becomes a
Stage 8 case (Section 7) instead -- port it to an OV custom op rather than
stubbing it out, since stubbing would silently change model behavior.

## Category 3: Legacy / stale upstream code vs. modern toolchain

**Symptom:** the repo hasn't been touched in years and uses APIs since
removed or changed: `np.float` (removed in NumPy >= 1.24, use
`np.float64`/`float`), `torch.distributed.deprecated` (removed), Python 2
print statements, old `einops`/`torchvision` call signatures, etc. These
raise plain `AttributeError`/`ImportError`/`SyntaxError` unrelated to
tracing or ONNX at all -- they fail before conversion even starts.

**Fix (rung 5, formal patch):** small, targeted patches per broken call
site -- e.g. `np.float` -> `np.float64`, remove/guard a dead
`torch.distributed.deprecated` import, wrap another CUDA-extension import
per Category 2. Each hunk should be independently explainable ("removed
NumPy alias", "removed torch API"), not a general modernization pass --
resist the urge to reformat or upgrade unrelated code in the same patch.

**Before patching, confirm it's really an environment mismatch, not a
model limitation:** pin an older NumPy/PyTorch in a throwaway venv first if
there's any doubt about whether the removed API still has a documented
replacement. Patching should target the minimal diff that lets modern
tooling load the file, not a rewrite.

## Category 4: Op not supported/traceable by the exporter -- substitute an equivalent

**Symptom:** export fails specifically because one op has no ONNX/OV
mapping, or the exporter can't trace through it -- distinct from Category 2
(a whole compiled extension failing to *import*) in that here the op is
pure Python/ATen but the *specific function* (e.g. `torch.nan_to_num`,
an uncommon interpolation mode, a custom autograd `Function`) isn't
export-friendly.

**Fix (rung 2, monkeypatch -- or rung 1, wrapper, if the call site is
inside code you already wrap):** replace the unsupported op with an
equivalent built from more primitive, well-supported ops that produce the
same result, scoped only to the export process. `torch.nan_to_num` is the
running example: it lowers to `aten::nan_to_num`, which the exporter
doesn't capture correctly in this pipeline, so replace it with an
`isfinite` + `where` combination -- both are basic, broadly-supported ops:

```python
def patch_nan_to_num_for_export():
    def export_friendly_nan_to_num(x, nan=0.0, posinf=None, neginf=None, out=None):
        replacement = torch.full_like(x, float(nan))
        return torch.where(torch.isfinite(x), x, replacement)
    torch.nan_to_num = export_friendly_nan_to_num
```

Watch for silent semantic narrowing when doing this: the replacement above
drops `posinf`/`neginf` handling entirely and maps every non-finite value
to `nan`. Check call sites in the model for whether they rely on
`posinf`/`neginf` being different from `nan` before assuming the simplified
version is equivalent -- **every substitution in this category needs the
same equivalence check**: enumerate the original op's documented
edge-case behaviors (special values, out-of-range inputs, dtype promotion
rules) and confirm the replacement matches them, not just the common case.

**General pattern beyond `nan_to_num`:** the same approach applies to any
op the exporter chokes on -- an interpolation mode, a sparse/scatter
variant, a custom autograd function wrapping otherwise-ordinary math.
Before assuming it's unfixable:

1. Check whether OpenVINO's supported-op list already covers a
   differently-named but semantically equivalent op (e.g. a newer PyTorch
   convenience function that's sugar over two or three older ops).
2. If not, decompose the failing op into the primitive ops it's
   documented to be equivalent to (its own docstring or a reference
   implementation is usually the best source), and monkeypatch/wrap that
   in.
3. Verify the decomposition against the original at the op level (unit
   test the replacement function against `torch`'s own implementation
   across a range of inputs including edge cases) before relying on
   end-to-end model validation to catch a mismatch -- op-level testing
   isolates the substitution from everything else that could also be
   wrong.

**Convert first with the op substitution; don't jump to a custom kernel.**
Categories 2 and 4 both end up replacing an unsupported/CUDA-only op with
an existing, decomposable equivalent -- that substitution alone is the
right (and sufficient) fix for getting the model converted and delivered.
Writing a hand-authored OpenCL/SYCL kernel for the original op is a
*separate, opt-in optimization*, not part of getting the baseline
conversion working, and it's meaningfully more effort for uncertain
payoff. Once the substituted-op IR converts, benchmarks, and validates
cleanly, mention to the user (per Section 6's closing question) that the
op you swapped out is also a candidate for Section 7's custom-op
migration path if they want to chase better performance later -- but do
not start that work unprompted.

## Category 5: Numerical instability / FP16 overflow at inference time

**Symptom:** conversion succeeds and the IR loads and compiles fine, but
inference output is NaN (or wildly wrong) specifically under FP16 -- FP32
or CPU-FP32 reference runs are clean. This is a **precision** problem, not
an export/graph-structure problem, and the fix is almost always a small,
targeted numeric adjustment rather than a structural change to the model.

**First, localize the exact op producing the first NaN -- don't guess from
architecture.** Load the IR, expose intermediate outputs, and run under
`INFERENCE_PRECISION_HINT=f16` while diffing against the same intermediate
in FP32. Walk backward from wherever NaN first appears until you find the
single op whose *inputs* are finite but *output* isn't -- that is the true
root cause, and it is usually one specific op, not "the network in
general."

**Common root cause: an epsilon/clamp value that underflows FP16.** A
very common source is `F.normalize(x, eps=1e-12)` and similar
guard-against-divide-by-zero patterns (`x / clamp(norm, min=eps)`). FP16's
smallest positive normal number is ~6.1e-5 -- `1e-12` is unrepresentable
and flushes to zero under FP16 storage/denormal handling on many
GPUs. When the tensor being normalized has a genuinely-zero-norm slice
(e.g. a zero-padded channel from `F.pad`), the clamp that was supposed to
prevent `0/0` silently becomes `Clamp(0, min=0) = 0`, and the divide
produces NaN that then propagates through the rest of the graph.

Fast-FoundationStereo hit exactly this: `F.normalize` on the GWC
cost-volume lowers to `/Div = target_volume / Clamp(ReduceL2(target_volume), min=1e-12)`.
Zero-padded disparity-shift slices in the cost volume have `L2 norm = 0`,
the `1e-12` clamp floor flushes to zero in FP16, and `0/0 = NaN` at that
node, then fans out everywhere downstream.

**Fix (IR-level attribute patch -- distinct from every other rung above,
apply it to the exported IR itself, not the PyTorch source):** once the
offending pattern is confirmed to be a too-small `Clamp.min`, raise it to
a value safely above the FP16 normal-number floor, directly on the
OpenVINO IR, after conversion:

```python
import openvino as ov

FP16_SAFE_EPS = 1e-4          # >> FP16 smallest positive normal (~6.1e-5)
TINY_EPS_THRESHOLD = 1e-6     # anything below this is almost certainly an
                               # F.normalize-style guard eps, not a real clamp

def patch_clamp_eps(model, old_eps_threshold=TINY_EPS_THRESHOLD, new_eps=FP16_SAFE_EPS):
    patched = []
    for n in model.get_ops():
        if n.get_type_name() != 'Clamp':
            continue
        old_min = n.get_attributes().get('min')
        if old_min is not None and 0 < old_min < old_eps_threshold:
            n.set_min(new_eps)
            patched.append((n.get_friendly_name(), old_min))
    return patched

core = ov.Core()
model = core.read_model('simplified.xml')
patch_clamp_eps(model)
model.validate_nodes_and_infer_types()
ov.save_model(model, 'mixed.xml', compress_to_fp16=True)
```

This is deliberately an **attribute-only** patch: op count, op types,
edges, and weights stay bit-identical -- only the `Clamp.min` scalar
changes on the handful of nodes matching the tiny-eps pattern. That
minimalism is itself a design choice worth keeping: an earlier version of
this exact fix also wrapped `Softmax`/`ReduceL2` in `Convert(f32)`
islands "to be safe," but ablation (toggling each wrap independently and
comparing output) showed those wraps changed nothing -- `Softmax` already
has a numerically-stable max-subtract implementation, and `ReduceL2`
inputs were never actually zero on those paths. The redundant `Convert`
ops cost real compile time and CPU latency for zero numerical benefit.
**Always ablate a proposed FP16 fix against the minimal version before
shipping it** -- "wrap more in FP32 to be safe" is not free, and belt-and-
suspenders patches that aren't verified to be necessary are a maintenance
liability disguised as caution.

Pick the new epsilon conservatively: large enough to clear the FP16
normal-number floor with margin (`1e-4` is ~1.6x the floor; don't cut it
close), small enough that it doesn't perturb genuinely non-zero norms
(since the value only matters when the true norm is at or near zero, the
exact choice barely affects finite-norm divides at all).

**Verify across many inputs and shapes, not just the one that first
showed NaN** -- FP16 underflow is data-dependent (it only triggers when a
real vector norm/reduction happens to land at zero, e.g. zero-padding at
specific disparity shifts), so a fix that looks complete on one demo image
can still miss shapes or inputs that hit the zero-norm path differently.
Re-run the skill's Section 4 validation (both A: FP16-vs-FP16 and B:
FP16-vs-FP32) across the full diverse-input set after the patch, not just
a single before/after comparison on the original failing case.

**If comparison A already passes and only B fails** (see Section 4's
attribution table), that is a *different* situation from this category --
it means the conversion is faithful to the FP16 source and the precision
loss is inherent to the architecture at this scale, not a fixable
underflow bug. Don't go hunting for a Clamp/eps issue that isn't there;
follow Section 4's mixed-precision guidance instead (keep specific
sensitive ops in FP32 via `INFERENCE_PRECISION_HINT` op-level overrides or
a decomposed pipeline).

## Category 6: Backend/plugin-specific compile failures on otherwise-valid graphs

**Symptom:** the ONNX/OV IR is produced successfully and `ov.Core().read_model()`
loads it without error, but `compile_model(model, "GPU")` fails (CPU
compiles and runs fine) with something like *"could not create a
primitive descriptor"* on a `Reshape`/`reorder` node -- specifically when
the reshape target has rank 5 with a trailing singleton dimension (e.g.
`[B, N, h, M, 1]`). The `intel_gpu` plugin has no kernel implementation to
select for that shape/op combination, even though the shape is
numerically trivial (a size-1 axis carrying no information).

**Concrete example -- `export_matchattention`.** The windowed match-attention
op builds a 5D `coords` tensor via `torch.stack((dx, dy), dim=-1)` and then
selects `coords[..., 0]` / `coords[..., 1]`; on GPU this `select` lowers to
a rank-5 `Gather` the plugin can't compile. Two sibling functions in the
same file have the same shape problem: `attn_scatter_1d` reshapes to
`[B, N, h, 2*win_r+2, 1]` before slicing (rank-5 `Reshape` -> `reorder`
failure), and `attn_gather_1d` does an in-place accumulate into a
`[B, N, h, 2*win_r+2, 1]` buffer (same failure on the write side).

**Fix (rung 5, formal patch to the model's forward-path math -- lighter
rungs don't apply because the problem is the tensor-rank shape of real
computation, not an import or a swappable backend):** rewrite each
function to do the same math without ever materializing the problematic
rank-5 shape, using only 4D ops:

- `compute_match_attention_1d`: instead of stacking `(dx, dy)` into a `[...,
  M, 2]` tensor and selecting axis 0/1 out of it, compute `x_coords` and
  `y_coords` directly as two independent 4D tensors (`x_center + dx`,
  `y_center.expand(...)`) -- no 5D tensor is ever created, so there is
  nothing for the plugin to fail to reshape.
- `attn_scatter_1d`: drop the artificial trailing width-1 dimension
  entirely and slice the 4D `attn` tensor directly (`attn[..., :2*win_r+1]`
  instead of reshaping to 5D first, then slicing, then reshaping back).
- `attn_gather_1d`: replace the 5D in-place scatter-add into a
  `[..., 2*win_r+2, 1]` buffer with `F.pad` on each 4D branch followed by a
  plain `add` (`F.pad(win_left, (0,1)) + F.pad(win_right, (1,0))`) --
  mathematically identical accumulation, no 5D tensor, no in-place write.

Each hunk keeps the exact same arithmetic, just re-expressed to avoid ever
constructing the rank-5 intermediate. This is the one category where the
patch touches actual computation (not just imports/dtypes), so it needs
correctness verification, not just "does it compile now":

- Diff the op graph before/after the patch for the affected subgraph.
- Run the skill's Section 4 dual validation (A: FP16 vs FP16, B: FP16 vs
  FP32) on the **patched** model against the **unpatched** model's CPU
  output as an extra reference, in addition to the normal comparisons --
  the patch changed real ops, so it needs the same evidence bar as any
  other correctness-sensitive change, not just "GPU compile succeeded."
- Document the exact plugin error message and the shape that triggered it
  in the conversion report; this is the kind of failure that's easy to
  reintroduce if the patch is dropped during a future upstream sync.

## Category 7: Dynamo exporter incompatibilities

**Symptom:** `torch.onnx.export(..., dynamo=True)` (the newer exporter)
fails where the legacy TorchScript-based exporter (`dynamo=False`) would
succeed -- e.g. duplicate weight/initializer names when a single `nn.Module`
instance is called multiple times in one `forward()` (common in recurrent
refinement/UNet-with-shared-blocks architectures), or a Windows-only
`UnicodeEncodeError` inside dynamo's own error-printing path when it hits
an unrelated unsupported pattern.

**Fix (rung 3, exporter flag):** pass `dynamo=False` and note in the
converter script *why* (which symptom was seen), so a future
maintainer doesn't "helpfully" flip it back to the new default. This is a
one-line, zero-risk fix -- always try it before anything more invasive
when the symptom matches.

## Formal patch policy (rung 5)

When nothing above resolves it and an actual source edit is required:

- Store the patch under `converter/patches/NNN-short-description.patch` (or
  `patches/` at the export root, matching this skill's existing examples),
  numbered in application order.
- Keep each patch minimal and single-purpose -- one concern per patch file,
  named after that concern (`001-layers-init-skip-C-extension.patch`, not
  `001-misc-fixes.patch`).
- Apply the patch as a reproducible step in the converter script or a
  documented `git apply` command in the README -- never hand-edit the
  submodule and leave it uncommitted with no record of what changed or why.
- In the conversion report, state explicitly: which file/function was
  patched, the exact error it fixes (paste the real error message), and
  why none of rungs 1-4 could reach the problem from outside. This
  distinction matters on re-review -- a reader should be able to tell
  "this patch was unavoidable" from "this patch was convenience" at a
  glance.
- Never patch to silence a correctness check (e.g. loosen a downstream
  validation threshold) -- patches in this skill exist to make the *model
  graph* exportable/compilable, not to make failing validation numbers
  pass.
