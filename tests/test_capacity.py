import math

from rag.capacity import Resources, estimate
from rag.config import Settings


def plan_settings(**overrides) -> Settings:
    # _env_file=None: tests must not depend on a developer's .env
    overrides.setdefault("parse_seconds_per_page", 1.0)   # the plan's baseline for these maths tests
    return Settings(_env_file=None, **overrides)


def test_volume_matches_plan_baseline():
    c = estimate(plan_settings(), Resources(cpu_cores=8, qdrant_ram_gb=1))
    assert c.pages_per_day == 150_000
    assert c.chunks_per_day == 450_000
    assert round(c.avg_docs_per_min, 1) == 6.9
    assert c.embed_tokens_per_day == 180_000_000


def test_eight_cores_sustain_10k_per_day_but_backlog_at_peak():
    c = estimate(plan_settings(), Resources(cpu_cores=8, qdrant_ram_gb=1))
    assert c.sustained_ok                      # ~2 workers needed on average, 6 available
    assert c.parse_workers_for_peak > c.parse_workers_available
    assert 0 < c.backlog_drain_minutes < 120   # backlog clears well within the day


def test_enough_workers_means_no_backlog():
    c = estimate(plan_settings(), Resources(cpu_cores=16, qdrant_ram_gb=1, parse_workers=16))
    assert c.peak_backlog_pages == 0
    assert c.backlog_drain_minutes == 0


def test_too_few_workers_never_drain():
    c = estimate(plan_settings(), Resources(cpu_cores=2, qdrant_ram_gb=1, parse_workers=1))
    assert not c.sustained_ok
    assert math.isinf(c.backlog_drain_minutes)


def test_qdrant_ram_scales_with_retention():
    small = estimate(plan_settings(retention_days=30), Resources(cpu_cores=8, qdrant_ram_gb=64))
    year = estimate(plan_settings(retention_days=365), Resources(cpu_cores=8, qdrant_ram_gb=64))
    assert small.qdrant_ok
    assert not year.qdrant_ok
    assert math.isclose(year.qdrant_ram_gb_retained / small.qdrant_ram_gb_retained, 365 / 30)


def test_embedding_budget_uses_80_percent_of_limit():
    s = plan_settings(openai_embed_tpm_limit=700_000)   # peak is 625K TPM
    assert not estimate(s, Resources(cpu_cores=8, qdrant_ram_gb=1)).embed_ok
    s = plan_settings(openai_embed_tpm_limit=1_000_000)
    assert estimate(s, Resources(cpu_cores=8, qdrant_ram_gb=1)).embed_ok
