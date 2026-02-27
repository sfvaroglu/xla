# NV FP4 + Scale Training Staging Branch

This branch (`nv-fp4-scale-training/staging`) consolidates unmerged PRs for NVIDIA FP4 and large-scale training work while upstream review is pending.

## Branch Purpose

- Integrate PRs that are in review and waiting for upstream merge
- Provide a stable base for [JAX Toolbox](https://github.com/NVIDIA/JAX-Toolbox) builds
- Enable large-scale training development to continue unblocked

## Repository

- **Staging repo:** https://github.com/sfvaroglu/xla
- **Branch:** `nv-fp4-scale-training/staging`
- **Baseline**: `30cd090178` (upstream/main)
- **Compatible JAX version**: `0.9.1.dev20260227+df6415b48`

## Tracked PRs

### From nv-fp4-training/staging

| PR | Description | Status |
|----|-------------|--------|
| [#33260](https://github.com/openxla/xla/pull/33260) | [GPU] Optimize all-gathers on non-major dimension using a single transpose | In review |
| [#36198](https://github.com/openxla/xla/pull/36198) | Algebraic simplifier: handle bitcast(unary_elementwise(bitcast())) | In review |
| [#38174](https://github.com/openxla/xla/pull/38174) | [GPU] Unroll and vectorize up to 32 elements / 256 bits on Blackwell | In review |
| [#37806](https://github.com/openxla/xla/pull/37806) | [GPU] Emitters: improve loop unrolling heuristic | In review |

### From nv-scale-training/staging

| PR | Description | Status |
|----|-------------|--------|
| [#33269](https://github.com/openxla/xla/pull/33269) | Add flag to control async compute resource limitation | In review |
| [#33240](https://github.com/openxla/xla/pull/33240) | Add delayMoveToHost heuristic to GPU latency hiding scheduler | In review |
| [#36462](https://github.com/openxla/xla/pull/36462) | Enable nccl symmetric kernels by default | In review |

**Note:** This branch may be force-pushed when rebased on upstream.

## Contact

- **Maintainer:** Sevin F. Varoglu (@sfvaroglu)
