# NV Large-Scale Training Staging Branch

This branch (`nv-scale-training/staging`) consolidates unmerged PRs for NVIDIA large-scale training work while upstream review is pending.

## Branch Purpose

- Integrate PRs that are in review and waiting for upstream merge
- Provide a stable base for [JAX Toolbox](https://github.com/NVIDIA/JAX-Toolbox) builds
- Enable large-scale training development to continue unblocked

## Repository

- **Staging repo:** https://github.com/sfvaroglu/xla
- **Branch:** `nv-scale-training/staging`
- **Upstream:** https://github.com/openxla/xla (`main`)

## Using This Branch

```bash
git remote add nv-staging git@github.com:sfvaroglu/xla.git
git fetch nv-staging
git checkout nv-staging/nv-scale-training/staging
```

This branch is periodically rebased on upstream `main`. Re-run `git fetch nv-staging` to get the latest.

### Stable Tags

Each rebase creates a new tag (e.g., `nv-staging-jax-1adf593cc`) tied to a specific JAX commit. To list PRs included in a tag:

```bash
git tag -n99 nv-staging-jax-1adf593cc
```

## Tracked PRs

### In review

| PR | Description | First Tag |
|----|-------------|-----------|
| [#36224](https://github.com/openxla/xla/pull/36224) | Support all-reduce hoisting for scatter-based accumulation pattern | `nv-staging-jax-1adf593cc` |
| [#36441](https://github.com/openxla/xla/pull/36441) | Support all-reduce hoisting with scalar multiplication pattern | `nv-staging-jax-1adf593cc` |
| [#36462](https://github.com/openxla/xla/pull/36462) | Enable nccl symmetric kernels by default | `nv-staging-jax-1066aa7b8` |
| [#39302](https://github.com/openxla/xla/pull/39302) | Fix dynamic memcpy offset computation for host offloading with collective pipelining | `nv-staging-jax-1adf593cc` |
| [#39604](https://github.com/openxla/xla/pull/39604) | Add annotation to allow scheduling of custom communication kernels | `nv-staging-jax-1adf593cc` |
| [#40316](https://github.com/openxla/xla/pull/40316) | Do aliasing only when collective permute is in-place | `nv-staging-jax-1adf593cc` |
| [#40921](https://github.com/openxla/xla/pull/40921) | Move barrier out of loop in collective permute | `nv-staging-jax-1adf593cc` |

### Merged upstream

| PR | Description | First Tag |
|----|-------------|-----------|
| [#26196](https://github.com/openxla/xla/pull/26196) | Add LHS config to prioritize compute nodes over collective starts | `nv-staging-jax-1066aa7b8` |
| [#33240](https://github.com/openxla/xla/pull/33240) | Add delayMoveToHost heuristic to GPU latency hiding scheduler | `nv-staging-jax-1066aa7b8` |
| [#33269](https://github.com/openxla/xla/pull/33269) | Add flag to control async compute resource limitation | `nv-staging-jax-1066aa7b8` |
| [#40656](https://github.com/openxla/xla/pull/40656) | Wire deadlock prevention for async collective multi-streaming | `nv-staging-jax-1adf593cc` |

**Note:** This branch may be force-pushed when rebased on upstream.

## Building from Source

To build JAX/XLA against this branch, check out the matching JAX commit from the tag name:

```bash
git clone https://github.com/jax-ml/jax.git && cd jax
git checkout <jax-commit>  # e.g. 1adf593cc for nv-staging-jax-1adf593cc

git clone git@github.com:sfvaroglu/xla.git
cd xla && git checkout nv-scale-training/staging && cd ..
```

Then follow your normal build flow.

## Contact

- **Maintainer:** Sevin F. Varoglu (@sfvaroglu)
