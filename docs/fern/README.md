# NeMo Lens Fern Documentation

This directory contains the Fern configuration for the NeMo Lens documentation site. Nightly content lives in the parent `docs/` directory, while frozen stable content lives under `versions/0.1.0/pages/`.

| Resource | Location |
|---|---|
| Published site | https://docs.nvidia.com/nemo/lens |
| Fern dashboard | https://dashboard.buildwithfern.com |
| Site configuration | `docs.yml` |
| Version navigation | `versions/*.yml` |
| CI workflows | `../../.github/workflows/fern-docs-*.yml` |

## Local Setup

Install Node.js 20 or newer. Docker is required for local Python API generation. The npm scripts run the Fern CLI version pinned in `fern.config.json`; no global installation is required.

```bash
npm --prefix docs/fern run login
```

The login must have access to the NVIDIA Fern organization so the local server can fetch `global-theme: nvidia`.

## Validate

```bash
npm --prefix docs/fern run generate:library:local
npm --prefix docs/fern run check
```

Local generation reads `src/nemo/lens`, writes ignored API pages under `product-docs/`, and sanitizes generated MDX.

## Local Preview

After generating the local API reference, temporarily comment out the `nemo-lens-local` library block in `docs.yml`, then run:

```bash
npm --prefix docs/fern run dev
```

Fern serves the preview at `http://localhost:3000`. The temporary edit is required because Fern's local generator supports path-backed libraries while the development server currently validates only git-backed library inputs.

## Hosted Preview

```bash
export FERN_TOKEN="$DOCS_FERN_TOKEN"
npm --prefix docs/fern run preview
```

GitHub Actions publishes pull request previews when `PUBLISH_FERN_PREVIEWS=true` and `DOCS_FERN_TOKEN` are configured.
