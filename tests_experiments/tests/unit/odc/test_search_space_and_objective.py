"""Unit tests for expanded ODC search space and objective."""

import pytest
from unittest.mock import MagicMock
from generation.meta.search_space import ODCSearchSpace
from generation.meta.objective import ODCObjective


def _make_trial_mock():
    trial = MagicMock()
    trial.suggest_float = MagicMock(side_effect=lambda name, low, high, **kw: (low + high) / 2)
    trial.suggest_int = MagicMock(side_effect=lambda name, low, high, **kw: (low + high) // 2)
    trial.suggest_categorical = MagicMock(side_effect=lambda name, choices: choices[0])
    return trial


class TestODCSearchSpace:
    def test_suggest_returns_dict(self):
        ss = ODCSearchSpace()
        params = ss.suggest(_make_trial_mock())
        assert isinstance(params, dict)
        assert "influence_radius" in params
        assert "kill_radius" in params
        assert "step_size" in params

    def test_anti_starburst_params_in_suggest(self):
        ss = ODCSearchSpace()
        params = ss.suggest(_make_trial_mock())
        assert "min_generations_before_tissue" in params
        assert "max_initial_branches" in params
        assert "force_bifurcation_depth" in params

    def test_branching_params_in_suggest(self):
        ss = ODCSearchSpace()
        params = ss.suggest(_make_trial_mock())
        assert "bifurcation_angle_min" in params
        assert "bifurcation_angle_max" in params
        assert "min_branch_length" in params


def _make_eval_and_coverage_mocks():
    mock_eval = MagicMock()
    mock_eval.structure.murray_deviation = 0.1
    mock_eval.flow.turbulent_fraction = 0.05
    mock_coverage = MagicMock()
    mock_coverage.overall_coverage = 0.9
    mock_coverage.ordering_compliance = 0.85
    return mock_eval, mock_coverage


class TestODCObjective:
    def test_compute_reward_returns_float(self):
        objective = ODCObjective()
        mock_eval, mock_coverage = _make_eval_and_coverage_mocks()
        reward = objective.compute_reward(mock_eval, mock_coverage)
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0

    def test_compute_reward_with_anti_starburst_info(self):
        objective = ODCObjective()
        mock_eval, mock_coverage = _make_eval_and_coverage_mocks()
        info = {"current_max_generation": 4, "branching_regularity": 0.9}
        reward = objective.compute_reward(mock_eval, mock_coverage, anti_starburst_info=info)
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0

    def test_compute_reward_breakdown_has_anti_starburst(self):
        objective = ODCObjective()
        mock_eval, mock_coverage = _make_eval_and_coverage_mocks()
        breakdown = objective.compute_reward_breakdown(mock_eval, mock_coverage)
        assert isinstance(breakdown, dict)
        assert "anti_starburst_score" in breakdown
        assert "branching_regularity_score" in breakdown

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
