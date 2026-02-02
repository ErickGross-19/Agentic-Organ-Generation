"""
Task Context for DesignSpec LLM Agent

This module provides persistent tracking of the current task and goals
across multiple conversation turns. The agent can remember what it's
working on, completed sub-tasks, and current blockers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class TaskContext:
    """
    Persistent context of current work.

    Tracks the high-level goal, progress through sub-tasks,
    and any blockers encountered.
    """
    goal: str                           # "Create tapered channels with ridge"
    started_at: str                     # ISO timestamp
    sub_tasks_completed: List[str] = field(default_factory=list)     # ["added domain", "added ports"]
    current_sub_task: Optional[str] = None    # "debugging union failure"
    blockers: List[str] = field(default_factory=list)                # ["pitch too coarse"]
    last_updated: str = ""              # ISO timestamp

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "goal": self.goal,
            "started_at": self.started_at,
            "sub_tasks_completed": self.sub_tasks_completed,
            "current_sub_task": self.current_sub_task,
            "blockers": self.blockers,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskContext":
        """Load from dictionary."""
        return cls(
            goal=data.get("goal", ""),
            started_at=data.get("started_at", ""),
            sub_tasks_completed=data.get("sub_tasks_completed", []),
            current_sub_task=data.get("current_sub_task"),
            blockers=data.get("blockers", []),
            last_updated=data.get("last_updated", ""),
        )

    @classmethod
    def create_new(cls, goal: str) -> "TaskContext":
        """Create a new task context."""
        now = datetime.now().isoformat()
        return cls(
            goal=goal,
            started_at=now,
            last_updated=now,
        )

    def update(
        self,
        current_sub_task: Optional[str] = None,
        completed_sub_task: Optional[str] = None,
        new_blocker: Optional[str] = None,
        clear_blockers: bool = False,
    ) -> None:
        """
        Update the task context.

        Parameters
        ----------
        current_sub_task : str, optional
            Set the current sub-task
        completed_sub_task : str, optional
            Mark a sub-task as completed
        new_blocker : str, optional
            Add a new blocker
        clear_blockers : bool
            Clear all blockers
        """
        if current_sub_task is not None:
            self.current_sub_task = current_sub_task

        if completed_sub_task is not None:
            if completed_sub_task not in self.sub_tasks_completed:
                self.sub_tasks_completed.append(completed_sub_task)

        if new_blocker is not None:
            if new_blocker not in self.blockers:
                self.blockers.append(new_blocker)

        if clear_blockers:
            self.blockers.clear()

        self.last_updated = datetime.now().isoformat()

    def to_prompt_text(self) -> str:
        """Convert to text for LLM prompt."""
        lines = []
        lines.append("## Current Task")
        lines.append(f"Goal: {self.goal}")

        if self.current_sub_task:
            lines.append(f"Current sub-task: {self.current_sub_task}")

        if self.sub_tasks_completed:
            lines.append("Completed sub-tasks:")
            for task in self.sub_tasks_completed:
                lines.append(f"  - {task}")

        if self.blockers:
            lines.append("Current blockers:")
            for blocker in self.blockers:
                lines.append(f"  - {blocker}")

        return "\n".join(lines)
