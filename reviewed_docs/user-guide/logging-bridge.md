# Logging Bridge

Traces tell you "what happened"; logs tell you "why." The OTel logging bridge correlates Python `logging` records with the active span's trace ID and exports them through the same pipeline as spans, so Kibana shows log lines alongside their originating trace (or any OTLP-capable log backend).

## Enable the Logging Bridge

```python
from nemo.lens.logging_bridge import setup_logging_bridge

# After setup_telemetry:
if handle.is_exporting and config.logs_enabled:
    setup_logging_bridge()   # bridges root logger
```

The env var `NEMO_LENS_LOGS_ENABLED=1` only enables the `LoggerProvider` (it sets `config.logs_enabled`); it does NOT install the bridge. You must still call `setup_logging_bridge()`:

```bash
export NEMO_LENS_LOGS_ENABLED=1   # enables the LoggerProvider
```

```python
from nemo.lens.logging_bridge import setup_logging_bridge
setup_logging_bridge()            # installs the Python logging handler
```

## What It Does

This function installs an `opentelemetry.sdk._logs.LoggingHandler` on the specified Python logger (which defaults to the root logger). The handler performs the following actions:

1. Captures the log record.
2. Reads the current OTel context to obtain the `trace_id` and `span_id`.
3. Emits an OTel `LogRecord` through the `LoggerProvider`.
4. Batches and sends records through OTLP to the collector.

## Filter by Log Level

By default, the bridge processes log records at the `INFO` level and above. Pass a custom level to adjust this threshold:

```python
setup_logging_bridge(level=logging.DEBUG)
```

Be deliberate with `DEBUG` — training codebases log prolifically at that level.

## Bridge a Specific Logger

```python
setup_logging_bridge(logger_name="megatron.training")
```

Only messages from that logger and its children are bridged. This is useful when you want to export the logs of a specific subsystem without mirroring the log output of every third-party library.

## Trace Correlation

The primary benefit of this bridge is that every bridged log record carries a `trace_id` and `span_id`. In Kibana, or in any OTLP log backend, you can perform the following tasks:

- Filter logs by trace ID to find every log line produced during a specific trace.
- Click from a Jaeger trace into the corresponding log stream.
- Build dashboards that correlate errors with trace performance.

Without the bridge, logs are stored in a different index without a link back to the traces.

## Choose Between the Logging Bridge and the Exporter

OTel logs are in addition to, and not a replacement for, your existing log pipeline (such as stdout, files, or syslog). The bridge acts as an additional sink; install the bridge when you have a centralized observability stack that benefits from trace correlation. The shipped stack uses Elasticsearch and Kibana, but any OTLP-capable log backend is compatible.

If you only require stdout logs for a local run, you do not need to install the bridge.

## Failure Modes

The `setup_logging_bridge` function fails gracefully in the following scenarios:

- If the OTel logs SDK is not installed, the function is silently skipped by catching an `ImportError`.
- If the provider setup fails, the function logs a debug message, and training continues.

The bridge never breaks the application. A failed bridge results in a lack of bridged logs rather than an application crash.

## Enable Logs in Providers

The bridge requires an active `LoggerProvider`. The `build_providers` function in `providers.py` configures one when `config.logs_enabled=True`. Setting `NEMO_LENS_LOGS_ENABLED=1` (which sets `config.logs_enabled`) prompts `setup_telemetry` to build the `LoggerProvider`. However, you must still call `setup_logging_bridge()` to install the Python logging handler, as NeMo Lens does not call this function automatically.

## Performance

Each log record is serialized through OTLP, which adds performance overhead. On a rank emitting 10,000 log lines per step, this serialization can dominate the cost of your log pipeline. Consider the following options to mitigate this overhead:

- Raise the level to `WARNING` or `ERROR` to bridge only meaningful events.
- Bridge only specific loggers (`setup_logging_bridge(logger_name=...)`).
- Use stdout logs on hot-path ranks, and only bridge logs from one or two ranks.
