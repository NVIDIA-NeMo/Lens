# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the export strategy registry."""

import pytest

from nemo.lens.config import NemoLensConfig
from nemo.lens.handle import _should_export, setup_telemetry
from nemo.lens.strategies import (
    BUILTIN_STRATEGIES,
    ExportStrategy,
    get_export_strategy,
    register_export_strategy,
    registered_strategies,
    unregister_export_strategy,
)


class TestBuiltinsRegistered:
    def test_all_four_builtins_present(self):
        names = registered_strategies()
        assert "all_ranks" in names
        assert "single_rank" in names
        assert "sampled" in names
        assert "first_rank_per_node" in names

    def test_builtin_strategies_constant(self):
        assert (
            frozenset({"all_ranks", "single_rank", "sampled", "first_rank_per_node"})
            == BUILTIN_STRATEGIES
        )

    def test_get_returns_callable(self):
        strat = get_export_strategy("all_ranks")
        cfg = NemoLensConfig(export_strategy="all_ranks")
        assert strat(cfg, 0, 4) is True


class TestFirstRankPerNode:
    def test_local_rank_zero_exports(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "0")
        cfg = NemoLensConfig(export_strategy="first_rank_per_node")
        assert _should_export(cfg, rank=0, world_size=8) is True
        assert _should_export(cfg, rank=4, world_size=8) is True

    def test_local_rank_nonzero_does_not_export(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "3")
        cfg = NemoLensConfig(export_strategy="first_rank_per_node")
        assert _should_export(cfg, rank=3, world_size=8) is False

    def test_local_rank_missing_treated_as_zero(self, monkeypatch):
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        cfg = NemoLensConfig(export_strategy="first_rank_per_node")
        assert _should_export(cfg, rank=0, world_size=1) is True


class TestRegisterCustomStrategy:
    def test_register_then_lookup(self):
        def even_ranks(cfg, rank, ws):
            return rank % 2 == 0

        register_export_strategy("test_even", even_ranks)
        assert get_export_strategy("test_even") is even_ranks
        assert "test_even" in registered_strategies()

    def test_register_dispatches_via_should_export(self):
        register_export_strategy("test_even", lambda c, r, ws: r % 2 == 0)
        cfg = NemoLensConfig(export_strategy="test_even")
        assert _should_export(cfg, rank=0, world_size=4) is True
        assert _should_export(cfg, rank=1, world_size=4) is False
        assert _should_export(cfg, rank=2, world_size=4) is True

    def test_register_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            register_export_strategy("", lambda c, r, ws: True)

    def test_register_duplicate_raises(self):
        register_export_strategy("test_dup", lambda c, r, ws: True)
        with pytest.raises(ValueError, match="already registered"):
            register_export_strategy("test_dup", lambda c, r, ws: False)

    def test_register_with_override(self):
        register_export_strategy("test_dup", lambda c, r, ws: True)
        register_export_strategy("test_dup", lambda c, r, ws: False, allow_override=True)
        cfg = NemoLensConfig(export_strategy="test_dup")
        assert _should_export(cfg, rank=0, world_size=4) is False

    def test_register_builtin_without_override_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register_export_strategy("all_ranks", lambda c, r, ws: False)

    def test_register_builtin_with_override_works(self):
        register_export_strategy("all_ranks", lambda c, r, ws: False, allow_override=True)
        cfg = NemoLensConfig(export_strategy="all_ranks")
        assert _should_export(cfg, rank=0, world_size=4) is False


class TestUnregisterStrategy:
    def test_unregister_custom(self):
        register_export_strategy("test_remove", lambda c, r, ws: True)
        unregister_export_strategy("test_remove")
        assert "test_remove" not in registered_strategies()

    def test_unregister_unknown_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            unregister_export_strategy("never_existed")

    def test_unregister_builtin_raises(self):
        with pytest.raises(ValueError, match="built-in"):
            unregister_export_strategy("all_ranks")
        assert "all_ranks" in registered_strategies()


class TestUnknownStrategyDispatch:
    def test_unknown_strategy_raises_with_listing(self):
        cfg = NemoLensConfig(export_strategy="not_a_real_strategy")
        with pytest.raises(ValueError, match="Unknown export_strategy"):
            _should_export(cfg, rank=0, world_size=4)


class TestSetupTelemetryWithCustomStrategy:
    def test_registered_strategy_via_setup(self):
        register_export_strategy("test_rank2", lambda c, r, ws: r == 2)
        cfg = NemoLensConfig(enabled=True, export_strategy="test_rank2", exporter="console")
        handle_a = setup_telemetry(cfg, rank=2, world_size=4)
        assert handle_a.is_exporting is True

    def test_registered_strategy_via_setup_non_export_rank(self):
        register_export_strategy("test_rank2", lambda c, r, ws: r == 2)
        cfg = NemoLensConfig(enabled=True, export_strategy="test_rank2", exporter="console")
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert handle.is_exporting is False

    def test_export_strategy_arg_overrides_config(self):
        cfg = NemoLensConfig(enabled=True, exporter="console")
        handle = setup_telemetry(cfg, rank=0, world_size=4, export_strategy=lambda c, r, ws: True)
        assert handle.is_exporting is True


class TestExportStrategyTypeAlias:
    def test_type_alias_exists(self):
        strat: ExportStrategy = lambda c, r, ws: True  # noqa: E731
        assert strat is not None
