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

### Setup

```bash
git remote add nv-staging git@github.com:sfvaroglu/xla.git
git fetch nv-staging
git checkout nv-staging/nv-scale-training/staging
```

### Staying Updated

This branch is rebased on upstream `main` every Friday. To get the latest:

```bash
git fetch nv-staging
git checkout nv-staging/nv-scale-training/staging
```

### Stable Tags

Each rebase creates a new tag (e.g., `nv-staging-jax-7bc8045f3`) tied to a specific JAX commit. To list PRs included in a tag:

```bash
git tag -n99 nv-staging-jax-7bc8045f3
```

## Tracked PRs

| PR | Description | Status | First Tag |
|----|-------------|--------|-----------|
| [#26196](https://github.com/openxla/xla/pull/26196) | Add LHS config to prioritize compute nodes over collective starts | In review | `nv-staging-jax-1066aa7b8` |
| [#33269](https://github.com/openxla/xla/pull/33269) | Add flag to control async compute resource limitation | In review | `nv-staging-jax-1066aa7b8` |
| [#33240](https://github.com/openxla/xla/pull/33240) | Add delayMoveToHost heuristic to GPU latency hiding scheduler | In review | `nv-staging-jax-1066aa7b8` |
| [#36462](https://github.com/openxla/xla/pull/36462) | Enable nccl symmetric kernels by default | In review | `nv-staging-jax-1066aa7b8` |

**Note:** This branch may be force-pushed when rebased on upstream.

## Contact

- **Maintainer:** Sevin F. Varoglu (@sfvaroglu)
