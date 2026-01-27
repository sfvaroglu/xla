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

## Tracked PRs

| PR | Description | Status |
|----|-------------|--------|
| [#26196](https://github.com/openxla/xla/pull/26196) | Add LHS config to prioritize compute nodes over collective starts | In review |

**Note:** This branch may be force-pushed when rebased on upstream.

## Contact

- **Maintainer:** Sevin F. Varoglu (@sfvaroglu)
