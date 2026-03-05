"""
Unit tests for TaskContext

Tests the task context tracking functionality.
"""

import pytest
from datetime import datetime
from automation.designspec_llm.task_context import TaskContext


class TestTaskContext:
    """Test TaskContext dataclass and methods."""

    def test_create_new_task_context(self):
        """Test creating a new task context."""
        task = TaskContext.create_new("Create channels with ridge")

        assert task.goal == "Create channels with ridge"
        assert task.started_at is not None
        assert task.last_updated is not None
        assert len(task.sub_tasks_completed) == 0
        assert task.current_sub_task is None
        assert len(task.blockers) == 0

    def test_update_current_sub_task(self):
        """Test updating current sub-task."""
        task = TaskContext.create_new("Test goal")
        task.update(current_sub_task="Adding domain")

        assert task.current_sub_task == "Adding domain"

    def test_complete_sub_task(self):
        """Test marking sub-task as completed."""
        task = TaskContext.create_new("Test goal")
        task.update(completed_sub_task="Added domain")
        task.update(completed_sub_task="Added ports")

        assert len(task.sub_tasks_completed) == 2
        assert "Added domain" in task.sub_tasks_completed
        assert "Added ports" in task.sub_tasks_completed

    def test_add_blocker(self):
        """Test adding a blocker."""
        task = TaskContext.create_new("Test goal")
        task.update(new_blocker="Pitch too coarse")
        task.update(new_blocker="Units mismatch")

        assert len(task.blockers) == 2
        assert "Pitch too coarse" in task.blockers

    def test_clear_blockers(self):
        """Test clearing blockers."""
        task = TaskContext.create_new("Test goal")
        task.update(new_blocker="Blocker 1")
        task.update(new_blocker="Blocker 2")
        task.update(clear_blockers=True)

        assert len(task.blockers) == 0

    def test_to_dict_from_dict(self):
        """Test serialization and deserialization."""
        task = TaskContext.create_new("Test goal")
        task.update(current_sub_task="Current task")
        task.update(completed_sub_task="Completed task")
        task.update(new_blocker="Test blocker")

        # Serialize
        data = task.to_dict()

        # Deserialize
        restored = TaskContext.from_dict(data)

        assert restored.goal == task.goal
        assert restored.current_sub_task == task.current_sub_task
        assert restored.sub_tasks_completed == task.sub_tasks_completed
        assert restored.blockers == task.blockers

    def test_to_prompt_text(self):
        """Test conversion to prompt text."""
        task = TaskContext.create_new("Test goal")
        task.update(current_sub_task="Current task")
        task.update(completed_sub_task="Task 1")
        task.update(new_blocker="Blocker 1")

        text = task.to_prompt_text()

        assert "Test goal" in text
        assert "Current task" in text
        assert "Task 1" in text
        assert "Blocker 1" in text

    def test_last_updated_changes(self):
        """Test that last_updated changes on update."""
        task = TaskContext.create_new("Test goal")
        original_update_time = task.last_updated

        import time
        time.sleep(0.01)  # Small delay

        task.update(current_sub_task="New task")

        assert task.last_updated != original_update_time
