"""Unit tests for expanded ODC search space and objective."""

import pytest
from generation.meta.search_space import ODCSearchSpace
from generation.meta.objective import ODCObjective


class TestODCSearchSpace:
    def test_sample_returns_dict(self):
        ss = ODCSearchSpace()
        from unittest.mock import MagicMock
        trial = MagicMock()
        trial.suggest_float = MagicMock(side_effect=lambda name, low, high, **kw: (low + high) / 2)
        trial.suggest_int = MagicMock(side_effect=lambda name, low, high, **kw: (low + high) // 2)
        trial.suggest_categorical = MagicMock(side_effect=lambda name, choices: choices[0])
        params = ss.sample(trial)
        assert isinstance(params, dict)
        assert "influence_radius" in params
        assert "kill_radius" in params
        assert "step_size" in params

    def test_anti_starburst_params_in_sample(self):
        ss = ODCSearchSpace()
        from unittest.mock import MagicMock
        trial = MagicMock()
        trial.suggest_float = MagicMock(side_effect=lambda name, low, high, **kw: (low + high) / 2)
        trial.suggest_int = MagicMock(side_effect=lambda name, low, high, **kw: (low + high) // 2)
        trial.suggest_categorical = MagicMock(side_effect=lambda name, choices: choices[0])
        params = ss.sample(trial)
        assert "min_generations_before_tissue" in params
        assert "max_initial_branches" in params
        assert "force_bifurcation_depth" in params

    def test_branching_params_in_sample(self):
        ss = ODCSearchSpace()
        from unittest.mock import MagicMock
        trial = MagicMock()
        trial.suggest_float = MagicMock(side_effect=lambda name, low, high, **kw: (low + high) / 2)
        trial.suggest_int = MagicMock(side_effect=lambda name, low, high, **kw: (low + high) // 2)
        trial.suggest_categorical = MagicMock(side_effect=lambda name, choices: choices[0])
        params = ss.sample(trial)
        assert "bifurcation_angle_min" in params
        assert "bifurcation_angle_max" in params
        assert "min_branch_length" in params


class TestODCObjective:
    def test_compute_reward_returns_float(self):
        objective = ODCObjective()
        from unittest.mock import MagicMock
        mock_network = MagicMock()
        mock_network.nodes = {}
        mock_network.segments = {}

        mock_tissue_spec = MagicMock()
        mock_tissue_spec.levels = []

        reward = objective.compute_reward(mock_network, mock_tissue_spec)
        assert isinstance(reward, float)

    def test_compute_reward_breakdown_has_anti_starburst(self):
        objective = ODCObjective()
        from unittest.mock import MagicMock
        mock_network = MagicMock()
        mock_network.nodes = {}
        mock_network.segments = {}

        mock_tissue_spec = MagicMock()
        mock_tissue_spec.levels = []

        breakdown = objective.compute_reward_breakdown(mock_network, mock_tissue_spec)
        assert isinstance(breakdown, dict)
        assert "anti_starburst" in breakdown
        assert "branching_regularity" in breakdown

    def test_weights_sum_to_one(self):
        objective = ODCObjective()
        total = (
            objective.coverage_weight
            + objective.ordering_weight
            + objective.murray_weight
            + objective.flow_weight
            + objective.anti_starburst_weight
            + objective.branching_regularity_weight
        )
        assert abs(total - 1.0) < 1e-6
