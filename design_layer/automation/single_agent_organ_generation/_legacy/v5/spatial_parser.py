"""
Spatial Parser Module for Text-Based Spatial Descriptions.

This module provides a structured spatial language parser that can handle
text-based commands for positioning and connecting vascular network elements.

Supported command patterns:
- "place inlet at [coordinates]" or "place inlet at [face]"
- "connect [source] to [target]"
- "parallel to [axis]"
- "distance from [reference]"
- Relative positioning: "2cm above inlet", "parallel to x-axis"

The parser converts natural language spatial descriptions into structured
coordinate and constraint specifications.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class SpatialCommandType(Enum):
    """Types of spatial commands that can be parsed."""
    PLACE = "place"
    CONNECT = "connect"
    PARALLEL = "parallel"
    DISTANCE = "distance"
    RELATIVE = "relative"
    CONSTRAINT = "constraint"
    UNKNOWN = "unknown"


class CoordinateSystem(Enum):
    """Coordinate system types."""
    CARTESIAN = "cartesian"
    FACE_BASED = "face_based"
    RELATIVE = "relative"


class Face(Enum):
    """Domain face identifiers."""
    LEFT = "left"      # x_min
    RIGHT = "right"    # x_max
    FRONT = "front"    # y_min
    BACK = "back"      # y_max
    BOTTOM = "bottom"  # z_min
    TOP = "top"        # z_max


@dataclass
class Position:
    """Represents a 3D position, either absolute or relative."""
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    face: Optional[Face] = None
    relative_to: Optional[str] = None
    offset: Optional[Tuple[float, float, float]] = None
    unit: str = "mm"
    
    def is_absolute(self) -> bool:
        """Check if this is an absolute position."""
        return self.x is not None and self.y is not None and self.z is not None
    
    def is_face_based(self) -> bool:
        """Check if this is a face-based position."""
        return self.face is not None
    
    def is_relative(self) -> bool:
        """Check if this is a relative position."""
        return self.relative_to is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {"unit": self.unit}
        if self.is_absolute():
            result["coordinates"] = [self.x, self.y, self.z]
        if self.face:
            result["face"] = self.face.value
        if self.relative_to:
            result["relative_to"] = self.relative_to
        if self.offset:
            result["offset"] = list(self.offset)
        return result


@dataclass
class SpatialConstraint:
    """Represents a spatial constraint."""
    constraint_type: str
    target: str
    value: Optional[float] = None
    axis: Optional[str] = None
    unit: str = "mm"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": self.constraint_type,
            "target": self.target,
            "value": self.value,
            "axis": self.axis,
            "unit": self.unit,
        }


@dataclass
class SpatialCommand:
    """Represents a parsed spatial command."""
    command_type: SpatialCommandType
    element: str
    position: Optional[Position] = None
    target: Optional[str] = None
    constraints: List[SpatialConstraint] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "command_type": self.command_type.value,
            "element": self.element,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
        }
        if self.position:
            result["position"] = self.position.to_dict()
        if self.target:
            result["target"] = self.target
        if self.constraints:
            result["constraints"] = [c.to_dict() for c in self.constraints]
        return result


@dataclass
class ParseResult:
    """Result of parsing spatial text."""
    success: bool
    commands: List[SpatialCommand] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "commands": [c.to_dict() for c in self.commands],
            "errors": self.errors,
            "warnings": self.warnings,
        }


class SpatialParser:
    """
    Parser for text-based spatial descriptions.
    
    This parser handles natural language descriptions of spatial positioning
    and converts them into structured commands that can be used by the
    vascular network generation system.
    
    Examples
    --------
    >>> parser = SpatialParser()
    >>> result = parser.parse("place inlet at left face")
    >>> result.commands[0].position.face
    Face.LEFT
    
    >>> result = parser.parse("place outlet 2cm above inlet")
    >>> result.commands[0].position.relative_to
    'inlet'
    """
    
    # Unit conversion factors to meters
    UNIT_CONVERSIONS = {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
        "um": 1e-6,
        "μm": 1e-6,
    }
    
    # Face name mappings
    FACE_MAPPINGS = {
        "left": Face.LEFT,
        "right": Face.RIGHT,
        "front": Face.FRONT,
        "back": Face.BACK,
        "bottom": Face.BOTTOM,
        "top": Face.TOP,
        "x_min": Face.LEFT,
        "x_max": Face.RIGHT,
        "y_min": Face.FRONT,
        "y_max": Face.BACK,
        "z_min": Face.BOTTOM,
        "z_max": Face.TOP,
        "-x": Face.LEFT,
        "+x": Face.RIGHT,
        "-y": Face.FRONT,
        "+y": Face.BACK,
        "-z": Face.BOTTOM,
        "+z": Face.TOP,
    }
    
    # Direction mappings for relative positioning
    DIRECTION_MAPPINGS = {
        "above": (0, 0, 1),
        "below": (0, 0, -1),
        "left of": (-1, 0, 0),
        "right of": (1, 0, 0),
        "in front of": (0, -1, 0),
        "behind": (0, 1, 0),
        "north": (0, 1, 0),
        "south": (0, -1, 0),
        "east": (1, 0, 0),
        "west": (-1, 0, 0),
        "up": (0, 0, 1),
        "down": (0, 0, -1),
    }
    
    # Regex patterns for parsing
    PATTERNS = {
        "coordinates": re.compile(
            r"\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?"
            r"(?:\s*(m|cm|mm|um|μm))?"
        ),
        "single_value": re.compile(
            r"(-?\d+(?:\.\d+)?)\s*(m|cm|mm|um|μm)?"
        ),
        "face": re.compile(
            r"(?:on\s+)?(?:the\s+)?(left|right|front|back|bottom|top|x_min|x_max|y_min|y_max|z_min|z_max)"
            r"(?:\s+face|\s+side)?",
            re.IGNORECASE
        ),
        "relative_position": re.compile(
            r"(\d+(?:\.\d+)?)\s*(m|cm|mm|um|μm)?\s+"
            r"(above|below|left of|right of|in front of|behind|north|south|east|west|up|down)\s+"
            r"(?:the\s+)?(\w+)",
            re.IGNORECASE
        ),
        "place_command": re.compile(
            r"place\s+(?:the\s+)?(\w+)\s+(?:at|on)\s+(.+)",
            re.IGNORECASE
        ),
        "connect_command": re.compile(
            r"connect\s+(?:the\s+)?(\w+)\s+to\s+(?:the\s+)?(\w+)",
            re.IGNORECASE
        ),
        "parallel_command": re.compile(
            r"parallel\s+to\s+(?:the\s+)?([xyz])\s*(?:-?\s*axis)?",
            re.IGNORECASE
        ),
        "distance_command": re.compile(
            r"(?:at\s+)?(?:a\s+)?distance\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(m|cm|mm|um|μm)?\s+"
            r"from\s+(?:the\s+)?(\w+)",
            re.IGNORECASE
        ),
        "axis": re.compile(r"([xyz])\s*(?:-?\s*axis)?", re.IGNORECASE),
    }
    
    def __init__(self, default_unit: str = "mm"):
        """
        Initialize the spatial parser.
        
        Parameters
        ----------
        default_unit : str
            Default unit for measurements (default: "mm")
        """
        self.default_unit = default_unit
    
    def parse(self, text: str) -> ParseResult:
        """
        Parse spatial description text into structured commands.
        
        Parameters
        ----------
        text : str
            Natural language spatial description
            
        Returns
        -------
        ParseResult
            Parsed commands with success status and any errors/warnings
        """
        commands = []
        errors = []
        warnings = []
        
        # Split into sentences/commands
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            try:
                command = self._parse_sentence(sentence)
                if command:
                    commands.append(command)
                else:
                    warnings.append(f"Could not parse: '{sentence}'")
            except Exception as e:
                errors.append(f"Error parsing '{sentence}': {str(e)}")
        
        success = len(commands) > 0 or len(errors) == 0
        return ParseResult(
            success=success,
            commands=commands,
            errors=errors,
            warnings=warnings,
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into individual command sentences."""
        # Split on periods, semicolons, or newlines
        parts = re.split(r'[.;\n]+', text)
        # Also split on "and" when it separates commands
        result = []
        for part in parts:
            # Check if "and" separates two commands
            if " and " in part.lower():
                subparts = re.split(r'\s+and\s+', part, flags=re.IGNORECASE)
                result.extend(subparts)
            else:
                result.append(part)
        return [p.strip() for p in result if p.strip()]
    
    def _parse_sentence(self, sentence: str) -> Optional[SpatialCommand]:
        """Parse a single sentence into a spatial command."""
        sentence_lower = sentence.lower()
        
        # Try each command pattern
        if "place" in sentence_lower:
            return self._parse_place_command(sentence)
        elif "connect" in sentence_lower:
            return self._parse_connect_command(sentence)
        elif "parallel" in sentence_lower:
            return self._parse_parallel_command(sentence)
        elif "distance" in sentence_lower:
            return self._parse_distance_command(sentence)
        else:
            # Try to parse as a relative position description
            return self._parse_relative_position(sentence)
    
    def _parse_place_command(self, sentence: str) -> Optional[SpatialCommand]:
        """Parse a 'place X at Y' command."""
        match = self.PATTERNS["place_command"].search(sentence)
        if not match:
            return None
        
        element = match.group(1).lower()
        location_text = match.group(2)
        
        position = self._parse_position(location_text)
        if not position:
            return None
        
        return SpatialCommand(
            command_type=SpatialCommandType.PLACE,
            element=element,
            position=position,
            raw_text=sentence,
        )
    
    def _parse_connect_command(self, sentence: str) -> Optional[SpatialCommand]:
        """Parse a 'connect X to Y' command."""
        match = self.PATTERNS["connect_command"].search(sentence)
        if not match:
            return None
        
        source = match.group(1).lower()
        target = match.group(2).lower()
        
        return SpatialCommand(
            command_type=SpatialCommandType.CONNECT,
            element=source,
            target=target,
            raw_text=sentence,
        )
    
    def _parse_parallel_command(self, sentence: str) -> Optional[SpatialCommand]:
        """Parse a 'parallel to X-axis' command."""
        match = self.PATTERNS["parallel_command"].search(sentence)
        if not match:
            return None
        
        axis = match.group(1).lower()
        
        return SpatialCommand(
            command_type=SpatialCommandType.PARALLEL,
            element="orientation",
            constraints=[SpatialConstraint(
                constraint_type="parallel",
                target=f"{axis}-axis",
                axis=axis,
            )],
            raw_text=sentence,
        )
    
    def _parse_distance_command(self, sentence: str) -> Optional[SpatialCommand]:
        """Parse a 'distance from X' command."""
        match = self.PATTERNS["distance_command"].search(sentence)
        if not match:
            return None
        
        value = float(match.group(1))
        unit = match.group(2) or self.default_unit
        reference = match.group(3).lower()
        
        return SpatialCommand(
            command_type=SpatialCommandType.DISTANCE,
            element="distance_constraint",
            constraints=[SpatialConstraint(
                constraint_type="distance",
                target=reference,
                value=value,
                unit=unit,
            )],
            raw_text=sentence,
        )
    
    def _parse_relative_position(self, sentence: str) -> Optional[SpatialCommand]:
        """Parse a relative position description like '2cm above inlet'."""
        match = self.PATTERNS["relative_position"].search(sentence)
        if not match:
            return None
        
        value = float(match.group(1))
        unit = match.group(2) or self.default_unit
        direction = match.group(3).lower()
        reference = match.group(4).lower()
        
        # Get direction vector
        direction_vector = self.DIRECTION_MAPPINGS.get(direction, (0, 0, 0))
        
        # Calculate offset
        offset = tuple(v * value for v in direction_vector)
        
        position = Position(
            relative_to=reference,
            offset=offset,
            unit=unit,
        )
        
        return SpatialCommand(
            command_type=SpatialCommandType.RELATIVE,
            element="position",
            position=position,
            raw_text=sentence,
        )
    
    def _parse_position(self, text: str) -> Optional[Position]:
        """Parse position from text (coordinates, face, or relative)."""
        text_lower = text.lower().strip()
        
        # Try to parse as coordinates [x, y, z]
        coord_match = self.PATTERNS["coordinates"].search(text)
        if coord_match:
            x = float(coord_match.group(1))
            y = float(coord_match.group(2))
            z = float(coord_match.group(3))
            unit = coord_match.group(4) or self.default_unit
            return Position(x=x, y=y, z=z, unit=unit)
        
        # Try to parse as face
        face_match = self.PATTERNS["face"].search(text)
        if face_match:
            face_name = face_match.group(1).lower()
            face = self.FACE_MAPPINGS.get(face_name)
            if face:
                return Position(face=face)
        
        # Try to parse as relative position
        rel_match = self.PATTERNS["relative_position"].search(text)
        if rel_match:
            value = float(rel_match.group(1))
            unit = rel_match.group(2) or self.default_unit
            direction = rel_match.group(3).lower()
            reference = rel_match.group(4).lower()
            
            direction_vector = self.DIRECTION_MAPPINGS.get(direction, (0, 0, 0))
            offset = tuple(v * value for v in direction_vector)
            
            return Position(
                relative_to=reference,
                offset=offset,
                unit=unit,
            )
        
        return None
    
    def convert_to_meters(self, value: float, unit: str) -> float:
        """Convert a value to meters."""
        factor = self.UNIT_CONVERSIONS.get(unit.lower(), 1.0)
        return value * factor
    
    def validate_commands(
        self,
        commands: List[SpatialCommand],
        known_elements: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Validate parsed commands for consistency.
        
        Parameters
        ----------
        commands : List[SpatialCommand]
            Commands to validate
        known_elements : List[str], optional
            List of known element names (inlet, outlet, etc.)
            
        Returns
        -------
        List[str]
            List of validation error messages
        """
        errors = []
        known_elements = known_elements or ["inlet", "outlet", "terminal", "branch"]
        
        placed_elements = set()
        
        for cmd in commands:
            # Check element names
            if cmd.element not in known_elements and cmd.element not in placed_elements:
                errors.append(f"Unknown element: '{cmd.element}'")
            
            # Track placed elements
            if cmd.command_type == SpatialCommandType.PLACE:
                placed_elements.add(cmd.element)
            
            # Check relative references
            if cmd.position and cmd.position.relative_to:
                ref = cmd.position.relative_to
                if ref not in known_elements and ref not in placed_elements:
                    errors.append(f"Reference to unknown element: '{ref}'")
            
            # Check connect targets
            if cmd.target:
                if cmd.target not in known_elements and cmd.target not in placed_elements:
                    errors.append(f"Connect target unknown: '{cmd.target}'")
        
        return errors


def parse_spatial_description(text: str, default_unit: str = "mm") -> ParseResult:
    """
    Convenience function to parse spatial descriptions.
    
    Parameters
    ----------
    text : str
        Natural language spatial description
    default_unit : str
        Default unit for measurements
        
    Returns
    -------
    ParseResult
        Parsed commands with success status
    """
    parser = SpatialParser(default_unit=default_unit)
    return parser.parse(text)


def format_position_as_text(position: Position, domain_size: Optional[Tuple[float, float, float]] = None) -> str:
    """
    Format a Position object as human-readable text.
    
    Parameters
    ----------
    position : Position
        Position to format
    domain_size : tuple, optional
        Domain size for face-based positions
        
    Returns
    -------
    str
        Human-readable position description
    """
    if position.is_absolute():
        return f"[{position.x}, {position.y}, {position.z}] {position.unit}"
    elif position.is_face_based():
        return f"on the {position.face.value} face"
    elif position.is_relative():
        if position.offset:
            ox, oy, oz = position.offset
            direction = ""
            if oz > 0:
                direction = f"{abs(oz)} {position.unit} above"
            elif oz < 0:
                direction = f"{abs(oz)} {position.unit} below"
            elif ox > 0:
                direction = f"{abs(ox)} {position.unit} right of"
            elif ox < 0:
                direction = f"{abs(ox)} {position.unit} left of"
            elif oy > 0:
                direction = f"{abs(oy)} {position.unit} behind"
            elif oy < 0:
                direction = f"{abs(oy)} {position.unit} in front of"
            return f"{direction} {position.relative_to}"
        return f"relative to {position.relative_to}"
    return "unspecified position"


def get_coordinate_system_description() -> str:
    """
    Get a text description of the coordinate system used.
    
    Returns
    -------
    str
        Description of the coordinate system
    """
    return """
COORDINATE SYSTEM DESCRIPTION
=============================

The vascular network generation system uses a right-handed Cartesian coordinate system:

Axes:
  - X-axis: Left (-) to Right (+)
  - Y-axis: Front (-) to Back (+)
  - Z-axis: Bottom (-) to Top (+)

Domain Faces:
  - LEFT face:   x = x_min (negative X boundary)
  - RIGHT face:  x = x_max (positive X boundary)
  - FRONT face:  y = y_min (negative Y boundary)
  - BACK face:   y = y_max (positive Y boundary)
  - BOTTOM face: z = z_min (negative Z boundary)
  - TOP face:    z = z_max (positive Z boundary)

Units:
  - Default: millimeters (mm)
  - Supported: m, cm, mm, um (micrometers)

Position Specification Methods:
  1. Absolute coordinates: [x, y, z] mm
  2. Face-based: "on the left face"
  3. Relative: "2cm above inlet"

Examples:
  - "place inlet at [0, 0, -15] mm"
  - "place inlet on the bottom face"
  - "place outlet 30mm above inlet"
  - "connect inlet to outlet"
  - "parallel to z-axis"
"""
