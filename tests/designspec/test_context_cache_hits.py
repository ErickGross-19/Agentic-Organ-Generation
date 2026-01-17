"""Tests for RunnerContext cache hits."""

import pytest
from designspec.context import RunnerContext, CacheEntry


class TestRunnerContextBasics:
    """Tests for basic RunnerContext functionality."""
    
    def test_create_context(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        assert ctx.spec_hash == "abc123"
        assert ctx.seed == 42
    
    def test_set_and_get(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        ctx.set("key1", "value1")
        
        assert ctx.get("key1") == "value1"
    
    def test_get_missing_returns_default(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        
        assert ctx.get("missing") is None
        assert ctx.get("missing", "default") == "default"
    
    def test_invalidate_removes_entry(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        ctx.set("key1", "value1")
        
        ctx.invalidate("key1")
        
        assert ctx.get("key1") is None


class TestRunnerContextCacheHits:
    """Tests for cache hit tracking."""
    
    def test_cache_hit_increments_on_second_get(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        ctx.set("key1", "value1")
        
        ctx.get("key1")
        ctx.get("key1")
        
        stats = ctx.stats()
        assert stats["hits"] >= 1
    
    def test_cache_miss_on_missing_key(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        
        ctx.get("missing")
        
        stats = ctx.stats()
        assert stats["misses"] >= 1
    
    def test_get_or_compute_caches_result(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        compute_count = [0]
        
        def compute():
            compute_count[0] += 1
            return "computed_value"
        
        result1 = ctx.get_or_compute("key1", compute)
        result2 = ctx.get_or_compute("key1", compute)
        
        assert result1 == "computed_value"
        assert result2 == "computed_value"
        assert compute_count[0] == 1
    
    def test_get_or_compute_skips_recompute(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        
        ctx.get_or_compute("key1", lambda: "first")
        result = ctx.get_or_compute("key1", lambda: "second")
        
        assert result == "first"


class TestRunnerContextKeyGeneration:
    """Tests for cache key generation."""
    
    def test_make_domain_key(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        domain_dict = {"type": "box", "x_min": -1, "x_max": 1}
        
        key = ctx.make_domain_key(domain_dict)
        
        assert "compiled_domain" in key
        assert len(key) > 0
    
    def test_same_domain_same_key(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        domain_dict = {"type": "box", "x_min": -1, "x_max": 1}
        
        key1 = ctx.make_domain_key(domain_dict)
        key2 = ctx.make_domain_key(domain_dict)
        
        assert key1 == key2
    
    def test_different_domain_different_key(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        domain1 = {"type": "box", "x_min": -1, "x_max": 1}
        domain2 = {"type": "box", "x_min": -2, "x_max": 2}
        
        key1 = ctx.make_domain_key(domain1)
        key2 = ctx.make_domain_key(domain2)
        
        assert key1 != key2
    
    def test_make_policy_key(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        policy_dict = {"enabled": True, "max_iterations": 100}
        
        key = ctx.make_policy_key("growth", policy_dict)
        
        assert "growth" in key
        assert len(key) > 0
    
    def test_make_component_key(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        
        key = ctx.make_component_key("net_1", "build")
        
        assert "net_1" in key
        assert len(key) > 0


class TestRunnerContextStats:
    """Tests for cache statistics."""
    
    def test_stats_returns_dict(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        stats = ctx.stats()
        
        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
    
    def test_stats_tracks_entries(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        ctx.set("key1", "value1")
        ctx.set("key2", "value2")
        
        stats = ctx.stats()
        
        assert stats["size"] == 2
    
    def test_stats_after_invalidate(self):
        ctx = RunnerContext(spec_hash="abc123", seed=42)
        ctx.set("key1", "value1")
        ctx.set("key2", "value2")
        ctx.invalidate("key1")
        
        stats = ctx.stats()
        
        assert stats["size"] == 1
