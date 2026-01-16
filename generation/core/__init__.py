"""Core data structures for vascular networks."""

from .types import Point3D, Direction3D, TubeGeometry
from .network import Node, VesselSegment, VascularNetwork
from .domain import DomainSpec, EllipsoidDomain, BoxDomain, CylinderDomain, MeshDomain
from .result import OperationResult, Delta
from .ids import IDGenerator

from .domain_transform import TransformDomain
from .domain_primitives import SphereDomain, CapsuleDomain, FrustumDomain
from .domain_composite import CompositeDomain
from .domain_implicit import ImplicitDomain

__all__ = [
    "Point3D",
    "Direction3D",
    "TubeGeometry",
    "Node",
    "VesselSegment",
    "VascularNetwork",
    "DomainSpec",
    "EllipsoidDomain",
    "BoxDomain",
    "CylinderDomain",
    "MeshDomain",
    "TransformDomain",
    "SphereDomain",
    "CapsuleDomain",
    "FrustumDomain",
    "CompositeDomain",
    "ImplicitDomain",
    "OperationResult",
    "Delta",
    "IDGenerator",
]
