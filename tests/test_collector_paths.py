"""Pins the contract between the collector's output path and the feature
builder's input path. Reading the same directory means the feature builder
will actually see whatever the collector writes."""

from __future__ import annotations

import importlib

import pytest

from src.collectors.kpx_smp import KpxSmpCollector
from src.utils import io as io_module


@pytest.fixture
def isolated_data_dir(monkeypatch, tmp_path):
    """Point Settings.data_dir at a tmp directory so writes don't pollute the repo."""
    monkeypatch.setattr(
        io_module, "get_settings",
        lambda: type("S", (), {"data_dir": tmp_path})(),
    )
    return tmp_path


def test_collector_write_dir_and_feature_loader_agree(isolated_data_dir, synthetic_smp_dataframe):
    # Sanity: source_root_dir uses the same path-derivation logic the collector
    # invokes via persist_collector_output(...).
    root = io_module.source_root_dir(KpxSmpCollector.source_name)
    assert root == isolated_data_dir / "raw" / "kpx" / "smp_day_ahead"

    # Simulate one collection run: write a parsed_*.parquet under the date dir.
    target_dir = io_module.raw_response_dir(KpxSmpCollector.source_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = target_dir / "parsed_test.parquet"
    synthetic_smp_dataframe.to_parquet(parsed_path, index=False)

    # The feature loader must find that file via rglob from source_root_dir.
    # Reload build_features so it picks up the patched io.get_settings.
    bf = importlib.reload(importlib.import_module("src.pipelines.build_features"))
    loaded = bf._load_collected_smp()
    assert len(loaded) == len(synthetic_smp_dataframe)
    assert set(loaded["area"]) == {"mainland"}


def test_feature_loader_raises_with_actionable_message(isolated_data_dir):
    bf = importlib.reload(importlib.import_module("src.pipelines.build_features"))
    with pytest.raises(FileNotFoundError, match="kpx_smp"):
        bf._load_collected_smp()
