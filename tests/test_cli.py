# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the nemo-lens command line interface."""

import io

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult

from nemo.lens import cli, providers
from tests.conftest import InMemorySpanExporter


class _FailingSpanExporter(InMemorySpanExporter):
    def export(self, spans):
        return SpanExportResult.FAILURE


class _RaisingSpanExporter(InMemorySpanExporter):
    def export(self, spans):
        raise RuntimeError("export failed")


def test_parse_span_arg_accepts_optional_parent():
    assert cli._parse_span_arg("child,1.25,2.5,parent") == cli._SpanSpec(
        "child", 1.25, 2.5, "parent"
    )


def test_parse_span_arg_warns_for_missing_and_unparseable_timestamps():
    stderr = io.StringIO()

    assert cli._parse_span_arg("missing-start,,2", stderr=stderr) is None
    assert cli._parse_span_arg("bad,one,2", stderr=stderr) is None
    assert cli._parse_span_arg(",1,2", stderr=stderr) is None

    warnings = stderr.getvalue()
    assert "error: skipping 'missing-start': missing timestamp" in warnings
    assert "error: skipping 'bad': unparseable timestamp" in warnings
    assert "error: skipping ',1,2': missing span name" in warnings


def test_parse_span_arg_rejects_end_before_start():
    stderr = io.StringIO()

    assert cli._parse_span_arg("backwards,2,1", stderr=stderr) is None

    assert (
        "error: skipping 'backwards': end timestamp precedes start timestamp" in stderr.getvalue()
    )


def test_order_span_specs_emits_parents_before_children():
    validation = cli._order_span_specs(
        [
            cli._SpanSpec("child", 2.0, 3.0, "root"),
            cli._SpanSpec("root", 1.0, 4.0),
            cli._SpanSpec("grandchild", 2.5, 2.75, "child"),
        ]
    )

    assert validation.invalid_count == 0
    assert [spec.name for spec in validation.order] == ["root", "child", "grandchild"]


def test_order_span_specs_rejects_duplicate_or_missing_parent():
    stderr = io.StringIO()

    validation = cli._order_span_specs(
        [
            cli._SpanSpec("root", 1.0, 4.0),
            cli._SpanSpec("root", 2.0, 3.0),
            cli._SpanSpec("orphan", 1.0, 2.0, "missing"),
        ],
        stderr=stderr,
    )

    assert validation.order == []
    assert validation.invalid_count == 2
    messages = stderr.getvalue()
    assert "error: invalid span 'root': duplicate span name" in messages
    assert "error: invalid span 'orphan': missing parent 'missing'" in messages


def test_main_emit_spans_uses_explicit_times_parent_context_and_env_resource(monkeypatch):
    exporter = InMemorySpanExporter()
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        (
            "service.name=from-env,slurm.job.id=123,slurm.array.count=1,"
            "slurm.nnodes=2,slurm.ntasks=2,slurm.restart_count=0,cluster.name=atlas"
        ),
    )
    args = cli._build_parser().parse_args(
        [
            "emit-spans",
            "--service",
            "nemo-test",
            "--span",
            "child,1700000001.5,1700000002.25,parent",
            "--span",
            "parent,1700000000,1700000003",
        ]
    )

    assert cli._run_emit_spans(args, span_exporter=exporter) == 0

    spans = {span.name: span for span in exporter.get_finished_spans()}
    child = spans["child"]
    parent = spans["parent"]

    assert child.parent.span_id == parent.context.span_id
    assert child.start_time == 1_700_000_001_500_000_000
    assert child.end_time == 1_700_000_002_250_000_000
    assert parent.resource.attributes["service.name"] == "nemo-test"
    assert parent.resource.attributes["slurm.job.id"] == "123"
    assert isinstance(parent.resource.attributes["slurm.job.id"], str)
    assert parent.resource.attributes["slurm.array.count"] == 1
    assert isinstance(parent.resource.attributes["slurm.array.count"], int)
    assert parent.resource.attributes["slurm.nnodes"] == 2
    assert isinstance(parent.resource.attributes["slurm.nnodes"], int)
    assert parent.resource.attributes["slurm.ntasks"] == 2
    assert isinstance(parent.resource.attributes["slurm.ntasks"], int)
    assert parent.resource.attributes["slurm.restart_count"] == 0
    assert isinstance(parent.resource.attributes["slurm.restart_count"], int)
    assert parent.resource.attributes["cluster.name"] == "atlas"


def test_main_emit_spans_service_overrides_env_service_name(monkeypatch):
    exporter = InMemorySpanExporter()
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.name=from-env,cluster.name=atlas")
    args = cli._build_parser().parse_args(
        [
            "emit-spans",
            "--service",
            "from-cli",
            "--span",
            "root,1700000000,1700000001",
        ]
    )

    assert cli._run_emit_spans(args, span_exporter=exporter) == 0

    (span,) = exporter.get_finished_spans()
    assert span.resource.attributes["service.name"] == "from-cli"
    assert span.resource.attributes["cluster.name"] == "atlas"


def test_main_emit_spans_returns_error_when_provider_setup_fails(monkeypatch):
    stderr = io.StringIO()

    def fail_provider_setup(*args, **kwargs):
        raise ImportError("missing OpenTelemetry SDK")

    monkeypatch.setattr(cli, "_build_span_emitter_provider", fail_provider_setup)
    args = cli._build_parser().parse_args(
        [
            "emit-spans",
            "--service",
            "nemo-test",
            "--span",
            "root,1700000000,1700000001",
        ]
    )

    assert cli._run_emit_spans(args, stderr=stderr) == 2
    assert "error: missing OpenTelemetry SDK" in stderr.getvalue()


@pytest.mark.parametrize(
    ("configured_exporter", "expected_exporter"),
    [(None, "otlp"), ("console", "console")],
)
def test_main_emit_spans_uses_lens_exporter_config(
    monkeypatch,
    configured_exporter,
    expected_exporter,
):
    selected = []
    span_exporter = InMemorySpanExporter()
    if configured_exporter is None:
        monkeypatch.delenv("NEMO_LENS_EXPORTER", raising=False)
    else:
        monkeypatch.setenv("NEMO_LENS_EXPORTER", configured_exporter)
    monkeypatch.setattr(
        providers,
        "_build_span_exporter",
        lambda config: selected.append(config.exporter) or span_exporter,
    )
    args = cli._build_parser().parse_args(
        ["emit-spans", "--service", "nemo-test", "--span", "root,1,2"]
    )

    assert cli._run_emit_spans(args) == 0
    assert selected == [expected_exporter]


@pytest.mark.parametrize(
    ("span_exporter", "expected_message"),
    [
        (_FailingSpanExporter(), "error: failed to export spans"),
        (_RaisingSpanExporter(), "error: failed to export spans"),
    ],
)
def test_main_emit_spans_returns_error_when_export_fails(span_exporter, expected_message):
    stderr = io.StringIO()
    args = cli._build_parser().parse_args(
        ["emit-spans", "--service", "nemo-test", "--span", "root,1,2"]
    )

    assert cli._run_emit_spans(args, span_exporter=span_exporter, stderr=stderr) == 2
    assert expected_message in stderr.getvalue()


def test_main_emit_spans_returns_error_when_flush_times_out(monkeypatch):
    stderr = io.StringIO()
    real_builder = cli._build_span_emitter_provider

    def build_provider(*args, **kwargs):
        provider, tracked_exporter = real_builder(*args, **kwargs)
        monkeypatch.setattr(provider, "force_flush", lambda: False)
        return provider, tracked_exporter

    monkeypatch.setattr(cli, "_build_span_emitter_provider", build_provider)
    args = cli._build_parser().parse_args(
        ["emit-spans", "--service", "nemo-test", "--span", "root,1,2"]
    )

    assert (
        cli._run_emit_spans(
            args,
            span_exporter=InMemorySpanExporter(),
            stderr=stderr,
        )
        == 2
    )
    assert "error: timed out while flushing spans" in stderr.getvalue()


@pytest.mark.parametrize(
    ("invalid_spans", "expected_message"),
    [
        (["orphan,1,2,missing"], "error: invalid span 'orphan': missing parent 'missing'"),
        (
            ["cycle-a,1,2,cycle-b", "cycle-b,1,2,cycle-a"],
            "error: invalid span graph: parent cycle",
        ),
        (["root,2,3"], "error: invalid span 'root': duplicate span name"),
        (["malformed,,2"], "error: skipping 'malformed': missing timestamp"),
        (["backwards,2,1"], "error: skipping 'backwards': end timestamp precedes start timestamp"),
        (["not-finite,nan,2"], "error: skipping 'not-finite': timestamp must be finite"),
    ],
)
def test_main_emit_spans_rejects_any_invalid_input_without_partial_emission(
    monkeypatch, invalid_spans, expected_message
):
    stderr = io.StringIO()
    exporter = InMemorySpanExporter()

    def fail_provider_setup(*args, **kwargs):
        raise AssertionError("provider setup should not run for invalid span input")

    monkeypatch.setattr(cli, "_build_span_emitter_provider", fail_provider_setup)

    argv = ["emit-spans", "--service", "nemo-test", "--span", "root,1,2"]
    for span in invalid_spans:
        argv.extend(["--span", span])
    args = cli._build_parser().parse_args(argv)

    assert cli._run_emit_spans(args, span_exporter=exporter, stderr=stderr) == 1
    assert exporter.get_finished_spans() == []
    messages = stderr.getvalue()
    assert expected_message in messages
    assert "nothing emitted: invalid span input" in messages
