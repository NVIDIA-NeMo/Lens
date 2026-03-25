# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for NemoLensConfig."""

import pytest

from nemo.lens.config import NemoLensConfig
from nemo.lens.groups import SpanGroup


class TestNemoLensConfigDefaults:
    def test_default_enabled_is_false(self):
        cfg = NemoLensConfig()
        assert cfg.enabled is False

    def test_default_service_name(self):
        cfg = NemoLensConfig()
        assert cfg.service_name == 'nemo'

    def test_default_export_strategy(self):
        cfg = NemoLensConfig()
        assert cfg.export_strategy == 'single_rank'

    def test_default_export_rank(self):
        cfg = NemoLensConfig()
        assert cfg.export_rank == -1

    def test_default_export_sample_rate(self):
        cfg = NemoLensConfig()
        assert cfg.export_sample_rate == 1.0

    def test_default_traces_enabled(self):
        cfg = NemoLensConfig()
        assert cfg.traces_enabled is True

    def test_default_metrics_enabled(self):
        cfg = NemoLensConfig()
        assert cfg.metrics_enabled is True

    def test_default_logs_enabled(self):
        cfg = NemoLensConfig()
        assert cfg.logs_enabled is False

    def test_default_span_groups(self):
        cfg = NemoLensConfig()
        assert cfg.span_groups == 'default'

    def test_default_exporter(self):
        cfg = NemoLensConfig()
        assert cfg.exporter == 'otlp'

    def test_default_resolved_span_groups(self):
        cfg = NemoLensConfig()
        groups = cfg.resolved_span_groups
        assert SpanGroup.JOB in groups
        assert SpanGroup.CHECKPOINT in groups
        assert SpanGroup.EVALUATE in groups
        assert SpanGroup.STEP not in groups


class TestNemoLensConfigFromEnv:
    def _clear_env(self, monkeypatch):
        for key in (
            'NEMO_LENS_ENABLED', 'NEMO_LENS_EXPORT_STRATEGY',
            'NEMO_LENS_EXPORT_RANK', 'NEMO_LENS_EXPORT_SAMPLE_RATE',
            'NEMO_LENS_TRACES_ENABLED', 'NEMO_LENS_METRICS_ENABLED',
            'NEMO_LENS_LOGS_ENABLED', 'NEMO_LENS_SPAN_GROUPS',
            'NEMO_LENS_EXPORTER', 'OTEL_SERVICE_NAME',
        ):
            monkeypatch.delenv(key, raising=False)

    def test_from_env_returns_defaults_with_no_vars(self, monkeypatch):
        self._clear_env(monkeypatch)
        cfg = NemoLensConfig.from_env()
        assert cfg.enabled is False
        assert cfg.service_name == 'nemo'
        assert cfg.export_rank == -1

    def test_enabled_set_by_env_var(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv('NEMO_LENS_ENABLED', '1')
        cfg = NemoLensConfig.from_env()
        assert cfg.enabled is True

    def test_enabled_false_values(self, monkeypatch):
        self._clear_env(monkeypatch)
        for val in ('0', 'false', 'no', 'off', 'FALSE'):
            monkeypatch.setenv('NEMO_LENS_ENABLED', val)
            cfg = NemoLensConfig.from_env()
            assert cfg.enabled is False, f"Expected False for {val!r}"

    def test_enabled_true_values(self, monkeypatch):
        self._clear_env(monkeypatch)
        for val in ('1', 'true', 'yes', 'on', 'True', 'YES'):
            monkeypatch.setenv('NEMO_LENS_ENABLED', val)
            cfg = NemoLensConfig.from_env()
            assert cfg.enabled is True, f"Expected True for {val!r}"

    def test_prefix_override(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv('MEGATRON_OTEL_ENABLED', '1')
        cfg = NemoLensConfig.from_env(prefix='MEGATRON_OTEL', fallback_prefix='NEMO_LENS')
        assert cfg.enabled is True

    def test_fallback_prefix(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv('NEMO_LENS_EXPORTER', 'console')
        cfg = NemoLensConfig.from_env(prefix='RL_OTEL', fallback_prefix='NEMO_LENS')
        assert cfg.exporter == 'console'

    def test_prefix_takes_precedence(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv('RL_OTEL_EXPORTER', 'console')
        monkeypatch.setenv('NEMO_LENS_EXPORTER', 'otlp')
        cfg = NemoLensConfig.from_env(prefix='RL_OTEL', fallback_prefix='NEMO_LENS')
        assert cfg.exporter == 'console'

    def test_export_strategy_all_ranks(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv('NEMO_LENS_EXPORT_STRATEGY', 'all_ranks')
        cfg = NemoLensConfig.from_env()
        assert cfg.export_strategy == 'all_ranks'

    def test_export_sample_rate(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv('NEMO_LENS_EXPORT_SAMPLE_RATE', '0.1')
        cfg = NemoLensConfig.from_env()
        assert cfg.export_sample_rate == 0.1

    def test_service_name_from_otel_standard_var(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv('OTEL_SERVICE_NAME', 'my-training-run')
        cfg = NemoLensConfig.from_env()
        assert cfg.service_name == 'my-training-run'

    def test_span_groups_per_step(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv('NEMO_LENS_SPAN_GROUPS', 'per_step')
        cfg = NemoLensConfig.from_env()
        assert cfg.span_groups == 'per_step'
        groups = cfg.resolved_span_groups
        assert SpanGroup.STEP in groups

    def test_logs_enabled(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv('NEMO_LENS_LOGS_ENABLED', '1')
        cfg = NemoLensConfig.from_env()
        assert cfg.logs_enabled is True

    def test_invalid_bool_raises(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv('NEMO_LENS_ENABLED', 'maybe')
        with pytest.raises(ValueError, match='NEMO_LENS_ENABLED'):
            NemoLensConfig.from_env()

    def test_invalid_int_raises(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv('NEMO_LENS_EXPORT_RANK', 'last')
        with pytest.raises(ValueError, match='NEMO_LENS_EXPORT_RANK'):
            NemoLensConfig.from_env()

    def test_invalid_float_raises(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv('NEMO_LENS_EXPORT_SAMPLE_RATE', 'high')
        with pytest.raises(ValueError, match='NEMO_LENS_EXPORT_SAMPLE_RATE'):
            NemoLensConfig.from_env()

    def test_span_group_cls_passed_through(self, monkeypatch):
        self._clear_env(monkeypatch)
        from nemo.lens.groups_megatron import MegatronSpanGroup
        cfg = NemoLensConfig.from_env(span_group_cls=MegatronSpanGroup)
        groups = cfg.resolved_span_groups
        assert MegatronSpanGroup.INFERENCE in groups
