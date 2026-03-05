"""
ID generation for vascular network elements.
"""

import numpy as np


class IDGenerator:
    """Deterministic ID generator with seed support."""
    
    def __init__(self, seed: int = None, start_id: int = 0):
        """
        Initialize ID generator.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for deterministic ID generation
        start_id : int
            Starting ID value (default: 0)
        """
        self.current_id = start_id
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else None
    
    def next_id(self) -> int:
        """Get next ID."""
        id_val = self.current_id
        self.current_id += 1
        return id_val
    
    def peek_next_id(self) -> int:
        """Peek at next ID without consuming it."""
        return self.current_id
    
    def reset(self, start_id: int = 0) -> None:
        """Reset ID counter."""
        self.current_id = start_id
    
    def get_state(self) -> dict:
        """Get current state for serialization."""
        return {
            "current_id": self.current_id,
            "seed": self.seed,
            "rng_state": self.rng.bit_generator.state if self.rng is not None else None,
        }
    
    def set_state(self, state: dict) -> None:
        """Restore state from serialization."""
        self.current_id = state["current_id"]
        self.seed = state["seed"]
        if state["rng_state"] is not None:
            self.rng = np.random.default_rng(self.seed)
            self.rng.bit_generator.state = state["rng_state"]
