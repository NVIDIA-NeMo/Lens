# Logging Bridge

Traces tell you "what happened"; logs tell you "why". The OTel logging bridge correlates Python `logging` records with the active span's trace ID and exports them through the same pipeline as spans — so Kibana shows log lines alongside the trace they came from (or any OTLP-capable log backend).

## Enable

```python
from nemo.lens.logging_bridge import setup_logging_bridge

# After setup_telemetry:
if handle.is_exporting and config.logs_enabled:
    setup_logging_bridge()   # bridges root logger
```

The env var `NEMO_LENS_LOGS_ENABLED=1` only enables the `LoggerProvider` (it sets `config.logs_enabled`); it does NOT install the bridge. You still must call `setup_logging_bridge()`:

```bash
export NEMO_LENS_LOGS_ENABLED=1   # enables the LoggerProvider
```

```python
from nemo.lens.logging_bridge import setup_logging_bridge
setup_logging_bridge()            # installs the Python logging handler
```

## What it does

Installs an `opentelemetry.sdk._logs.LoggingHandler` on the specified Python logger (root by default). The handler:

1. Captures the log record.
2. Reads the current OTel context to get `trace_id` and `span_id`.
3. Emits an OTel `LogRecord` via the `LoggerProvider`.
4. Records are batched and sent via OTLP to your collector.

## Level filtering

By default, bridges `INFO` and above. Pass a custom level:

```python
setup_logging_bridge(level=logging.DEBUG)
```

Be deliberate with `DEBUG` — training codebases log prolifically at that level.

## Bridging a specific logger

```python
setup_logging_bridge(logger_name="megatron.training")
```

Only messages from that logger (and its children) are bridged. Useful when you want to export a specific subsystem's logs without mirroring every third-party library's log output.

## Trace correlation

The key value proposition: every bridged log record carries `trace_id` and `span_id`. In Kibana (or any OTLP log backend), you can:

- Filter logs by trace ID to find every log line produced during a specific trace.
- Click from a Jaeger trace into the corresponding log stream.
- Build dashboards that correlate errors with trace performance.

Without the bridge, logs end up in a different index with no link back to traces.

## When to use it vs the exporter

OTel logs are **in addition to**, not a replacement for, your existing log pipeline (stdout, file, syslog). The bridge is an extra sink — set it up when you have a centralised observability stack (the shipped stack uses Elasticsearch/Kibana; any OTLP-capable log backend works) that benefits from trace correlation.

If you just want stdout logs for a local run, don't bother with the bridge.

## Failure modes

`setup_logging_bridge` fails gracefully:

- If the OTel logs SDK isn't installed → silent skip (ImportError caught).
- If provider setup fails → debug log message, training continues.

It never breaks the application. A failed bridge means "no bridged logs" — not a crash.

## Enabling logs in providers

The bridge needs a `LoggerProvider` to be active. `providers.py:build_providers` sets one up when `config.logs_enabled=True`. Setting `NEMO_LENS_LOGS_ENABLED=1` (config.logs_enabled) makes `setup_telemetry` build the `LoggerProvider`, but you must still call `setup_logging_bridge()` yourself to install the Python logging handler — lens never calls it automatically.

## Performance

Each log record gets serialised via OTLP, which adds overhead. On a rank emitting 10k log lines per step, this can dominate your log pipeline cost. Options:

- Raise the level to `WARNING` or `ERROR` to bridge only meaningful events.
- Bridge only specific loggers (`setup_logging_bridge(logger_name=...)`).
- Stick to stdout logs on hot-path ranks and only bridge from one or two ranks.
