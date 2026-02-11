"""Unit tests for procedural tissue point generation."""

import numpy as np
import pytest
from generation.tissue.procedural import (
    generate_poisson_disk,
    generate_liver_lobule_pattern,
    generate_lung_bronchiole_pattern,
    generate_kidney_nephron_pattern,
)
from generation.core.domain import BoxDomain


@pytest.fixture
def box_domain():
    return BoxDomain(
        x_min=-0.005, x_max=0.005,
        y_min=-0.005, y_max=0.005,
        z_min=-0.005, z_max=0.005,
    )


class TestPoissonDiskSampling:
    def test_basic_generation(self, box_domain):
        pts = generate_poisson_disk(box_domain.get_bounds(), min_distance=0.002, max_points=30, rng=np.random.default_rng(42), domain=box_domain)
        assert pts.shape[0] > 0
        assert pts.shape[1] == 3

    def test_min_distance_respected(self, box_domain):
        min_dist = 0.002
        pts = generate_poisson_disk(box_domain.get_bounds(), min_distance=min_dist, max_points=20, rng=np.random.default_rng(42), domain=box_domain)
        if pts.shape[0] > 1:
                        dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
                        dmat += np.eye(dmat.shape[0]) * 1e6
                        assert float(np.min(dmat)) >= min_dist * 0.95

    def test_reproducibility(self, box_domain):
        pts1 = generate_poisson_disk(box_domain.get_bounds(), min_distance=0.002, max_points=20, rng=np.random.default_rng(42), domain=box_domain)
        pts2 = generate_poisson_disk(box_domain.get_bounds(), min_distance=0.002, max_points=20, rng=np.random.default_rng(42), domain=box_domain)
        np.testing.assert_array_equal(pts1, pts2)

    def test_different_seeds_differ(self, box_domain):
        pts1 = generate_poisson_disk(box_domain.get_bounds(), min_distance=0.002, max_points=20, rng=np.random.default_rng(42), domain=box_domain)
        pts2 = generate_poisson_disk(box_domain.get_bounds(), min_distance=0.002, max_points=20, rng=np.random.default_rng(99), domain=box_domain)
        assert not np.array_equal(pts1, pts2)


class TestLiverLobulePattern:
    def test_basic_generation(self, box_domain):
        pts = generate_liver_lobule_pattern(box_domain, n_lobules=10, points_per_lobule=3, seed=42)
        assert pts.shape[0] > 0
        assert pts.shape[1] == 3

    def test_reproducibility(self, box_domain):
        pts1 = generate_liver_lobule_pattern(box_domain, n_lobules=10, points_per_lobule=3, seed=42)
        pts2 = generate_liver_lobule_pattern(box_domain, n_lobules=10, points_per_lobule=3, seed=42)
        np.testing.assert_array_equal(pts1, pts2)


class TestLungBronchiolePattern:
    def test_basic_generation(self, box_domain):
        pts = generate_lung_bronchiole_pattern(box_domain, n_clusters=8, points_per_cluster=4, seed=42)
        assert pts.shape[0] > 0
        assert pts.shape[1] == 3

    def test_reproducibility(self, box_domain):
        pts1 = generate_lung_bronchiole_pattern(box_domain, n_clusters=8, points_per_cluster=4, seed=42)
        pts2 = generate_lung_bronchiole_pattern(box_domain, n_clusters=8, points_per_cluster=4, seed=42)
        np.testing.assert_array_equal(pts1, pts2)


class TestKidneyNephronPattern:
    def test_basic_generation(self, box_domain):
        pts = generate_kidney_nephron_pattern(box_domain, n_nephrons=10, points_per_nephron=3, seed=42)
        assert pts.shape[0] > 0
        assert pts.shape[1] == 3

    def test_reproducibility(self, box_domain):
        pts1 = generate_kidney_nephron_pattern(box_domain, n_nephrons=10, points_per_nephron=3, seed=42)
        pts2 = generate_kidney_nephron_pattern(box_domain, n_nephrons=10, points_per_nephron=3, seed=42)
        np.testing.assert_array_equal(pts1, pts2)
