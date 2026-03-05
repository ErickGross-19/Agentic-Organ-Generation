"""
Unit tests for SessionMemory

Tests the session memory functionality for tracking decisions and error resolutions.
"""

import pytest
from automation.designspec_llm.session_memory import SessionMemory, Decision, ErrorResolution


class TestSessionMemory:
    """Test SessionMemory dataclass and methods."""

    def test_create_new_session_memory(self):
        """Test creating new session memory."""
        memory = SessionMemory.create_new("test_session")

        assert memory.session_id == "test_session"
        assert memory.created_at is not None
        assert len(memory.decisions) == 0
        assert len(memory.error_resolutions) == 0
        assert len(memory.user_preferences) == 0

    def test_add_decision(self):
        """Test adding decisions."""
        memory = SessionMemory.create_new("test_session")
        memory.add_decision(
            turn_number=1,
            decision="Used taper_factor=0.8",
            reasoning="Provides gradual taper"
        )

        assert len(memory.decisions) == 1
        assert memory.decisions[0].turn_number == 1
        assert memory.decisions[0].decision == "Used taper_factor=0.8"
        assert memory.decisions[0].reasoning == "Provides gradual taper"

    def test_add_error_resolution(self):
        """Test adding error resolutions."""
        memory = SessionMemory.create_new("test_session")
        memory.add_error_resolution(
            error_pattern="PITCH_TOO_LARGE",
            error_message="voxel_pitch is 0.01 but domain scale is ~0.02",
            resolution="Set voxel_pitch = 0.0002",
            success=True
        )

        assert len(memory.error_resolutions) == 1
        assert memory.error_resolutions[0].error_pattern == "PITCH_TOO_LARGE"
        assert memory.error_resolutions[0].success is True

    def test_set_preference(self):
        """Test setting user preferences."""
        memory = SessionMemory.create_new("test_session")
        memory.set_preference("default_inlet_radius", 0.0005)
        memory.set_preference("preferred_units", "mm")

        assert memory.user_preferences["default_inlet_radius"] == 0.0005
        assert memory.user_preferences["preferred_units"] == "mm"

    def test_get_recent_decisions(self):
        """Test getting recent decisions."""
        memory = SessionMemory.create_new("test_session")
        memory.add_decision(1, "Decision 1")
        memory.add_decision(2, "Decision 2")
        memory.add_decision(3, "Decision 3")
        memory.add_decision(4, "Decision 4")

        recent = memory.get_recent_decisions(limit=2)

        assert len(recent) == 2
        assert recent[0].turn_number == 4  # Most recent first
        assert recent[1].turn_number == 3

    def test_get_successful_error_resolutions(self):
        """Test filtering successful error resolutions."""
        memory = SessionMemory.create_new("test_session")
        memory.add_error_resolution("ERROR_1", "Message 1", "Fix 1", success=True)
        memory.add_error_resolution("ERROR_2", "Message 2", "Fix 2", success=False)
        memory.add_error_resolution("ERROR_3", "Message 3", "Fix 3", success=True)

        successful = memory.get_successful_error_resolutions()

        assert len(successful) == 2
        assert all(er.success for er in successful)

    def test_to_dict_from_dict(self):
        """Test serialization and deserialization."""
        memory = SessionMemory.create_new("test_session")
        memory.add_decision(1, "Test decision", "Test reasoning")
        memory.add_error_resolution("ERROR_1", "Message", "Fix", True)
        memory.set_preference("key", "value")

        # Serialize
        data = memory.to_dict()

        # Deserialize
        restored = SessionMemory.from_dict(data)

        assert restored.session_id == memory.session_id
        assert len(restored.decisions) == len(memory.decisions)
        assert len(restored.error_resolutions) == len(memory.error_resolutions)
        assert restored.user_preferences == memory.user_preferences

    def test_to_summary_dict(self):
        """Test creating summary dictionary."""
        memory = SessionMemory.create_new("test_session")
        memory.add_decision(1, "Decision 1")
        memory.add_decision(2, "Decision 2")
        memory.add_decision(3, "Decision 3")
        memory.add_decision(4, "Decision 4")
        memory.add_error_resolution("ERROR_1", "Msg", "Fix", True)
        memory.set_preference("pref1", "value1")

        summary = memory.to_summary_dict()

        assert "recent_decisions" in summary
        assert len(summary["recent_decisions"]) <= 3  # Limited to 3
        assert "successful_error_resolutions" in summary
        assert "user_preferences" in summary

    def test_to_prompt_text(self):
        """Test conversion to prompt text."""
        memory = SessionMemory.create_new("test_session")
        memory.add_decision(1, "Decision 1", "Reasoning 1")
        memory.add_error_resolution("ERROR_1", "Msg", "Fix", True)
        memory.set_preference("pref1", "value1")

        text = memory.to_prompt_text()

        assert "Decision 1" in text
        assert "Reasoning 1" in text
        assert "ERROR_1" in text
        assert "pref1" in text
