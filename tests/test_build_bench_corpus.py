"""Tests for the bench corpus builder's stratification logic."""

import pytest

from scripts.ocr.build_bench_corpus import (
    CONFIDENCE_BUCKETS,
    assign_bucket,
    stratified_sample,
)


def test_assign_bucket_boundaries():
    assert assign_bucket(0.0) == "[0.0, 0.3)"
    assert assign_bucket(0.29) == "[0.0, 0.3)"
    assert assign_bucket(0.3) == "[0.3, 0.6)"
    assert assign_bucket(0.59) == "[0.3, 0.6)"
    assert assign_bucket(0.6) == "[0.6, 0.85)"
    assert assign_bucket(0.84) == "[0.6, 0.85)"
    assert assign_bucket(0.85) == "[0.85, 1.0]"
    assert assign_bucket(1.0) == "[0.85, 1.0]"


def test_stratified_sample_equal_buckets():
    # 100 items in each bucket, request 50 per bucket, expect 200 total.
    rows = []
    for conf, bucket in [(0.1, "[0.0, 0.3)"), (0.4, "[0.3, 0.6)"), (0.7, "[0.6, 0.85)"), (0.9, "[0.85, 1.0]")]:
        for i in range(100):
            rows.append((f"stem_{bucket}_{i}", conf))
    sampled, skew = stratified_sample(rows, per_bucket=50, seed=42)
    assert len(sampled) == 200
    counts = {b: 0 for b in CONFIDENCE_BUCKETS}
    for stem, conf in sampled:
        counts[assign_bucket(conf)] += 1
    assert all(c == 50 for c in counts.values())
    assert skew == {}


def test_stratified_sample_underfilled_bucket_reports_skew():
    # [0.0, 0.3) has only 20 items; other buckets have 100 each.
    # With per_bucket=50, we take 20 + 50 + 50 + 50 = 170, then redistribute
    # the deficit (30) across over-filled buckets, capping at available rows.
    rows = []
    for i in range(20):
        rows.append((f"low_{i}", 0.1))
    for conf, bucket in [(0.4, "[0.3, 0.6)"), (0.7, "[0.6, 0.85)"), (0.9, "[0.85, 1.0]")]:
        for i in range(100):
            rows.append((f"stem_{bucket}_{i}", conf))
    sampled, skew = stratified_sample(rows, per_bucket=50, seed=42)
    assert len(sampled) == 200
    counts = {b: 0 for b in CONFIDENCE_BUCKETS}
    for stem, conf in sampled:
        counts[assign_bucket(conf)] += 1
    assert counts["[0.0, 0.3)"] == 20
    # Deficit of 30 should be spread across the 3 over-filled buckets.
    # Exact distribution isn't important; total must be 200.
    assert sum(counts.values()) == 200
    assert "[0.0, 0.3)" in skew
    assert skew["[0.0, 0.3)"]["target"] == 50
    assert skew["[0.0, 0.3)"]["actual"] == 20


def test_stratified_sample_deterministic_with_seed():
    rows = [(f"s{i}", 0.4) for i in range(100)]
    a, _ = stratified_sample(rows, per_bucket=10, seed=42)
    b, _ = stratified_sample(rows, per_bucket=10, seed=42)
    assert [s for s, _ in a] == [s for s, _ in b]
