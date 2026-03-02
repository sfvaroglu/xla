# Staging Branch: nv-fp4-training/staging (Open)

**Baseline**: `30cd090178` (upstream/main)
**Compatible JAX version**: `0.9.1.dev20260227+df6415b48`

## Cherry-picked PRs

### PR #33260: [GPU] Optimize all-gathers on non-major dimension using a single transpose (Open)
- **URL**: https://github.com/openxla/xla/pull/33260
- **Description**: All-gathers on non-major dimensions previously required two transposes. This pass rewrites them to gather on dimension 0 in the original layout, then apply a single transpose to the output.

### PR #36198: Algebraic simplifier: handle bitcast(unary_elementwise(bitcast())) (Open)
- **URL**: https://github.com/openxla/xla/pull/36198
- **Description**: Squashes pairs of bitcasts around unary elementwise ops, simplifying index computations. Skips bitcasts that change element type or tiling.

### PR #38174: [GPU] Unroll and vectorize up to 32 elements / 256 bits on Blackwell (Open)
- **URL**: https://github.com/openxla/xla/pull/38174
- **Description**: Increases maximum unroll factor and vector width for Blackwell GPUs. Restricted to CUDA 12.9+.

### PR #37806: [GPU] Emitters: improve loop unrolling heuristic (Open)
- **URL**: https://github.com/openxla/xla/pull/37806
- **Description**: Makes the unrolling heuristic traverse `xla.pure_call` computations while estimating cost. Increases cost of `math.exp`. Prevents excessive unrolling and register pressure.
