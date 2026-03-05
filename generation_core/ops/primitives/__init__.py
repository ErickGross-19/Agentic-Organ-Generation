"""
Primitive shape operations for vascular network generation.

This module provides low-level primitives for creating channel geometries,
domain meshes, and other basic shapes used in network construction.
"""

from .channels import (
    create_straight_channel,
    create_tapered_channel,
    create_fang_hook,
    create_channels_from_ports,
    ChannelPolicy,
)

__all__ = [
    "create_straight_channel",
    "create_tapered_channel",
    "create_fang_hook",
    "create_channels_from_ports",
    "ChannelPolicy",
]
