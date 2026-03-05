"""
Patch Generator for DesignSpec

This module generates RFC 6902 JSON Patches from structured errors.
It takes StructuredError objects and creates targeted patches that
fix the underlying issues.
"""

import logging
from typing import Any, Dict, List, Optional

from .error_taxonomy import StructuredError, ErrorType

logger = logging.getLogger(__name__)


class PatchGenerator:
    """
    Generates JSON Patches from structured errors.

    Creates RFC 6902 compliant patches that can be applied to
    fix errors identified by the error parser.
    """

    def __init__(self):
        """Initialize the patch generator."""
        pass

    def generate_fix_patches(
        self,
        errors: List[StructuredError],
        spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate JSON Patches to fix errors.

        Parameters
        ----------
        errors : list of StructuredError
            List of structured errors to fix
        spec : dict
            The current spec (for context)

        Returns
        -------
        list of dict
            List of RFC 6902 JSON patches
        """
        patches = []

        for error in errors:
            # Only generate patches for high-confidence errors
            if error.confidence < 0.7:
                continue

            # Generate patch based on error type
            error_patches = self._generate_patches_for_error(error, spec)
            patches.extend(error_patches)

        return patches

    def _generate_patches_for_error(
        self,
        error: StructuredError,
        spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate patches for a specific error.

        Parameters
        ----------
        error : StructuredError
            The structured error
        spec : dict
            The current spec

        Returns
        -------
        list of dict
            List of patches
        """
        patches = []

        # If error has a direct patch (json_path and suggested_value)
        if error.json_path and error.suggested_value is not None:
            patches.append({
                "op": "replace",
                "path": error.json_path,
                "value": error.suggested_value,
            })
            return patches

        # Generate patches based on error type
        if error.error_type == ErrorType.PITCH_TOO_LARGE:
            patches.extend(self._fix_pitch_too_large(error, spec))

        elif error.error_type == ErrorType.PITCH_TOO_SMALL:
            patches.extend(self._fix_pitch_too_small(error, spec))

        elif error.error_type == ErrorType.NO_DOMAIN_DEFINED:
            patches.extend(self._fix_no_domain(error, spec))

        elif error.error_type == ErrorType.DOMAIN_TOO_SMALL:
            patches.extend(self._fix_domain_too_small(error, spec))

        elif error.error_type == ErrorType.PORT_DIRECTION_WRONG:
            patches.extend(self._fix_port_direction(error, spec))

        elif error.error_type == ErrorType.COMPONENT_OUTSIDE_DOMAIN:
            patches.extend(self._fix_component_outside_domain(error, spec))

        # Add more error type handlers as needed

        return patches

    def _fix_pitch_too_large(
        self,
        error: StructuredError,
        spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate patches to fix pitch too large error."""
        patches = []

        # If we have a suggested value, use it
        if error.suggested_value is not None:
            patches.append({
                "op": "replace",
                "path": "/policies/mesh_merge/voxel_pitch",
                "value": error.suggested_value,
            })
        else:
            # Calculate from domain scale if available
            domains = spec.get("domains", [])
            if domains:
                # Get first domain scale
                domain = domains[0]
                if "box" in domain:
                    box = domain["box"]
                    scale = max(box.get("width", 0), box.get("height", 0), box.get("depth", 0))
                    suggested_pitch = scale / 100.0

                    patches.append({
                        "op": "replace",
                        "path": "/policies/mesh_merge/voxel_pitch",
                        "value": suggested_pitch,
                    })

        return patches

    def _fix_pitch_too_small(
        self,
        error: StructuredError,
        spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate patches to fix pitch too small error."""
        patches = []

        # Get current pitch and increase it
        current_pitch = spec.get("policies", {}).get("mesh_merge", {}).get("voxel_pitch")
        if current_pitch:
            # Increase by 50%
            new_pitch = current_pitch * 1.5
            patches.append({
                "op": "replace",
                "path": "/policies/mesh_merge/voxel_pitch",
                "value": new_pitch,
            })

        return patches

    def _fix_no_domain(
        self,
        error: StructuredError,
        spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate patches to add a domain."""
        patches = []

        # Add a default box domain
        default_domain = {
            "type": "box",
            "box": {
                "width": 0.02,   # 20mm
                "height": 0.02,  # 20mm
                "depth": 0.02,   # 20mm
            },
        }

        domains = spec.get("domains", [])
        if not domains:
            # Add domains array
            patches.append({
                "op": "add",
                "path": "/domains",
                "value": [default_domain],
            })
        else:
            # Append to existing domains
            patches.append({
                "op": "add",
                "path": f"/domains/{len(domains)}",
                "value": default_domain,
            })

        return patches

    def _fix_domain_too_small(
        self,
        error: StructuredError,
        spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate patches to increase domain size."""
        patches = []

        domains = spec.get("domains", [])
        if domains:
            # Increase first domain by 50%
            domain = domains[0]
            if "box" in domain:
                box = domain["box"]
                width = box.get("width", 0.02) * 1.5
                height = box.get("height", 0.02) * 1.5
                depth = box.get("depth", 0.02) * 1.5

                patches.append({
                    "op": "replace",
                    "path": "/domains/0/box/width",
                    "value": width,
                })
                patches.append({
                    "op": "replace",
                    "path": "/domains/0/box/height",
                    "value": height,
                })
                patches.append({
                    "op": "replace",
                    "path": "/domains/0/box/depth",
                    "value": depth,
                })

        return patches

    def _fix_port_direction(
        self,
        error: StructuredError,
        spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate patches to fix port direction."""
        patches = []

        # Find components with inlets/outlets
        components = spec.get("components", [])
        for i, component in enumerate(components):
            # Check inlets
            inlets = component.get("inlets", [])
            for j, inlet in enumerate(inlets):
                direction = inlet.get("direction")
                if direction and not self._is_valid_direction(direction):
                    # Set to default direction [1, 0, 0]
                    patches.append({
                        "op": "replace",
                        "path": f"/components/{i}/inlets/{j}/direction",
                        "value": [1, 0, 0],
                    })

            # Check outlets
            outlets = component.get("outlets", [])
            for j, outlet in enumerate(outlets):
                direction = outlet.get("direction")
                if direction and not self._is_valid_direction(direction):
                    # Set to default direction [-1, 0, 0]
                    patches.append({
                        "op": "replace",
                        "path": f"/components/{i}/outlets/{j}/direction",
                        "value": [-1, 0, 0],
                    })

        return patches

    def _fix_component_outside_domain(
        self,
        error: StructuredError,
        spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate patches to move component inside domain."""
        patches = []

        # Move components to center of domain
        domains = spec.get("domains", [])
        components = spec.get("components", [])

        if domains and components:
            # Assume center is at origin [0, 0, 0]
            for i, component in enumerate(components):
                position = component.get("position")
                if position:
                    # Move to origin
                    patches.append({
                        "op": "replace",
                        "path": f"/components/{i}/position",
                        "value": [0.0, 0.0, 0.0],
                    })

        return patches

    def _is_valid_direction(self, direction: List[float]) -> bool:
        """Check if direction is a valid face normal."""
        if len(direction) != 3:
            return False

        # Check if it's a unit vector along one axis
        valid_directions = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ]

        return direction in valid_directions

    def generate_explanation(
        self,
        errors: List[StructuredError],
        patches: List[Dict[str, Any]],
    ) -> str:
        """
        Generate human-readable explanation of patches.

        Parameters
        ----------
        errors : list of StructuredError
            The errors being fixed
        patches : list of dict
            The generated patches

        Returns
        -------
        str
            Explanation text
        """
        lines = []
        lines.append("## Proposed Fixes\n")

        for error in errors:
            if error.confidence >= 0.7:
                lines.append(f"**{error.error_type.value}**")
                lines.append(f"- Current issue: {error.suggested_fix}")
                if error.json_path:
                    lines.append(f"- Affected: {error.json_path}")
                if error.current_value is not None:
                    lines.append(f"- Current value: {error.current_value}")
                if error.suggested_value is not None:
                    lines.append(f"- Suggested value: {error.suggested_value}")
                lines.append("")

        if patches:
            lines.append(f"Generated {len(patches)} patch(es) to fix these issues.")
        else:
            lines.append("No automatic patches could be generated. Manual intervention may be required.")

        return "\n".join(lines)
