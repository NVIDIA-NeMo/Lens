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
npm --prefix docs/fern run check
```

The checked-in configuration uses only the git-backed `nemo-lens` library, so
validation, hosted previews, and publication do not depend on a local checkout
path.

## Local Preview

To generate the Python API reference from the current checkout, temporarily
uncomment the `nemo-lens-local` block in `docs.yml`:

```yaml
  nemo-lens-local:
    input:
      path: ../../src/nemo/lens
    output:
      path: ./product-docs/nemo-lens/Full-Library-Reference
    lang: python
```

Generate and sanitize the local API pages:

```bash
npm --prefix docs/fern run generate:library:local
```

This reads `src/nemo/lens` through Docker and writes ignored API pages under
`docs/fern/product-docs/`.

Comment out the `nemo-lens-local` block again before validating or serving the
site. Fern's local API generator supports path-backed libraries, but remote docs
generation and the development server do not.

```bash
npm --prefix docs/fern run check
npm --prefix docs/fern run dev
```

Fern serves the preview at `http://localhost:3000` using the generated local API
pages.

## Hosted Preview

```bash
export FERN_TOKEN="$DOCS_FERN_TOKEN"
npm --prefix docs/fern run preview
```

GitHub Actions publishes pull request previews when `PUBLISH_FERN_PREVIEWS=true` and `DOCS_FERN_TOKEN` are configured.

Do not enable `nemo-lens-local` for hosted previews or publication. Fern rejects
local `input.path` libraries during remote docs generation.
