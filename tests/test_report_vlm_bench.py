"""Tests for report_vlm_bench.py metric computation."""

import pytest

from scripts.ocr.report_vlm_bench import (
    compute_metrics,
    parse_model_key,
    pareto_frontier,
)


def test_parse_model_key_splits_host():
    assert parse_model_key("kimi-vl:latest@ares-cpu") == ("kimi-vl:latest", "ares-cpu")


def test_parse_model_key_no_host_returns_none():
    assert parse_model_key("sonnet") == ("sonnet", None)


def test_compute_metrics_basic():
    sonnet = {"s1": "1999-10-03", "s2": "1994-06-22", "s3": "2001-05-17"}
    candidate = [
        {"stem": "s1", "raw_text": "10 3'99", "parsed_date": "1999-10-03", "elapsed_s": 5.0},
        {"stem": "s2", "raw_text": "'94 6 22", "parsed_date": "1994-06-22", "elapsed_s": 7.0},
        {"stem": "s3", "raw_text": "garbled", "parsed_date": None, "elapsed_s": 6.0},
    ]
    m = compute_metrics(sonnet_by_stem=sonnet, rows=candidate)
    assert m["total"] == 3
    assert m["agree_pct"] == pytest.approx(66.67, abs=0.1)
    assert m["unparsed_pct"] == pytest.approx(33.33, abs=0.1)
    assert m["high_conf_wrong_pct"] == 0.0
    assert m["median_s"] == 6.0


def test_compute_metrics_flags_high_conf_wrong():
    sonnet = {"s1": "1999-10-03"}
    candidate = [
        {"stem": "s1", "raw_text": "10 3'98", "parsed_date": "1998-10-03", "elapsed_s": 5.0},
    ]
    m = compute_metrics(sonnet_by_stem=sonnet, rows=candidate)
    assert m["agree_pct"] == 0.0
    assert m["high_conf_wrong_pct"] == 100.0


def test_compute_metrics_timeout_and_oom():
    sonnet = {"s1": "1999-10-03", "s2": "1994-06-22"}
    candidate = [
        {"stem": "s1", "raw_text": "TIMEOUT", "parsed_date": None, "elapsed_s": 180.0},
        {"stem": "s2", "raw_text": "OOM_ERROR", "parsed_date": None, "elapsed_s": 2.0},
    ]
    m = compute_metrics(sonnet_by_stem=sonnet, rows=candidate)
    assert m["timeout_pct"] == 50.0
    assert m["oom_pct"] == 50.0


def test_pareto_frontier_picks_dominant_points():
    points = [
        {"model_key": "a", "agree_pct": 90.0, "imgs_per_sec": 1.0},
        {"model_key": "b", "agree_pct": 85.0, "imgs_per_sec": 5.0},
        {"model_key": "c", "agree_pct": 80.0, "imgs_per_sec": 3.0},
        {"model_key": "d", "agree_pct": 92.0, "imgs_per_sec": 0.5},
    ]
    frontier = pareto_frontier(points)
    keys = {p["model_key"] for p in frontier}
    assert keys == {"a", "b", "d"}
