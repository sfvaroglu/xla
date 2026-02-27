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

This branch is periodically rebased on upstream `main`. To get the latest:

```bash
git fetch nv-staging
git checkout nv-staging/nv-scale-training/staging
```

### Stable Tags

Each rebase creates a new tag (e.g., `nv-staging-jax-757c92dbb`) tied to a specific JAX commit. To list PRs included in a tag:

```bash
git tag -n99 nv-staging-jax-757c92dbb
```

## Tracked PRs

| PR | Description | Status |
|----|-------------|--------|
| [#33269](https://github.com/openxla/xla/pull/33269) | Add flag to control async compute resource limitation | In review |
| [#33240](https://github.com/openxla/xla/pull/33240) | Add delayMoveToHost heuristic to GPU latency hiding scheduler | In review |
| [#36462](https://github.com/openxla/xla/pull/36462) | Enable nccl symmetric kernels by default | In review |

[#26196](https://github.com/openxla/xla/pull/26196) (LHS config to prioritize compute over collective starts) was merged upstream; this branch is based on a commit that already includes it.

**Note:** This branch may be force-pushed when rebased on upstream.

## Contact

- **Maintainer:** Sevin F. Varoglu (@sfvaroglu)
