# Installation

## Basic install

```bash
pip install nemo-lens
```

This pulls in only `opentelemetry-api` — the no-op implementation. Instrumented code runs correctly but no spans or metrics are exported. Useful for library authors who want to instrument code without forcing downstream users to install the SDK.

## With OTel SDK + OTLP exporters

```bash
pip install 'nemo-lens[sdk]'
```

Adds `opentelemetry-sdk` and the OTLP exporters (`opentelemetry-exporter-otlp-proto-grpc`, `opentelemetry-exporter-otlp-proto-http`). Required on any rank that actually exports telemetry.

## Framework contrib extras

```bash
pip install 'nemo-lens[fastapi]'    # FastAPI server auto-instrumentation
pip install 'nemo-lens[aiohttp]'    # aiohttp client auto-instrumentation
```

Each adds the relevant OTel instrumentation package. See [Contrib](../user-guide/contrib.md) for details.

## Development install

```bash
git clone <repo-url>
cd lens
pip install --group dev -e .
pre-commit install
```

The `dev` dependency group includes the SDK, pytest, pytest-cov, pre-commit, and ruff. See [Developer Guide](../developer/contributing.md).

## Python support

`nemo-lens` requires **Python ≥ 3.13**. It uses the PEP 695 `type` statement (e.g. in `src/nemo/lens/strategies.py`) and other modern typing features.

## Packaging notes

`nemo.lens` is a **PEP 420 namespace package** so it coexists peacefully with NeMo Framework's `nemo.*` packages. The `src/nemo/__init__.py` contains only an explanatory comment (no code), so it does not shadow other `nemo.*` packages on the path.
