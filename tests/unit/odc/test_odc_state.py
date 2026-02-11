"""Unit tests for ODCState."""

import pytest
from generation.ops.odc_state import ODCState


class TestODCStateConstruction:
    def test_default_fields(self):
        from unittest.mock import MagicMock
        mock_sc = MagicMock()
        mock_sc.tissue_points = []
        mock_sc.active_tissue_indices = set()
        state = ODCState(sc_state=mock_sc)
        assert state.current_level_idx == 0
        assert state.stall_counter == 0
        assert state.global_step == 0
        assert len(state.node_generations) == 0
        assert len(state.force_bifurcate_nodes) == 0

    def test_get_set_node_generation(self):
        from unittest.mock import MagicMock
        mock_sc = MagicMock()
        mock_sc.tissue_points = []
        mock_sc.active_tissue_indices = set()
        state = ODCState(sc_state=mock_sc)
        assert state.get_node_generation(1) == 0
        state.set_node_generation(1, 3)
        assert state.get_node_generation(1) == 3

    def test_levels_unlocked_tracking(self):
        from unittest.mock import MagicMock
        mock_sc = MagicMock()
        mock_sc.tissue_points = []
        mock_sc.active_tissue_indices = set()
        state = ODCState(sc_state=mock_sc)
        state.levels_unlocked[0] = True
        state.levels_unlocked[1] = True
        assert state.levels_unlocked[0] is True
        assert state.levels_unlocked[1] is True
        assert 2 not in state.levels_unlocked
