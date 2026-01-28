"""
Text Feedback Module for Enhanced GUI Communication.

This module provides utilities for generating rich text-based feedback
for the GUI chat interface, including:
- Structured input templates for common commands
- Rich text formatting for code blocks and tables
- Text-based visualization panels
- Exportable text summaries of designs
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class FeedbackLevel(Enum):
    """Feedback verbosity levels."""
    MINIMAL = "minimal"
    NORMAL = "normal"
    VERBOSE = "verbose"
    DEBUG = "debug"


@dataclass
class TextReport:
    """A structured text report."""
    title: str
    sections: List[Tuple[str, str]] = field(default_factory=list)
    summary: str = ""
    
    def add_section(self, heading: str, content: str) -> None:
        """Add a section to the report."""
        self.sections.append((heading, content))
    
    def to_text(self, include_borders: bool = True) -> str:
        """Convert to formatted text."""
        lines = []
        
        if include_borders:
            border = "=" * max(len(self.title), 60)
            lines.append(border)
            lines.append(self.title.center(len(border)))
            lines.append(border)
        else:
            lines.append(f"# {self.title}")
        
        lines.append("")
        
        if self.summary:
            lines.append(self.summary)
            lines.append("")
        
        for heading, content in self.sections:
            if include_borders:
                lines.append(f"--- {heading} ---")
            else:
                lines.append(f"## {heading}")
            lines.append(content)
            lines.append("")
        
        return "\n".join(lines)


class InputTemplate:
    """Templates for structured user input."""
    
    TOPOLOGY_SELECTION = """
Select a vascular network topology:

  [1] tree        - Single inlet branching to multiple terminals
  [2] dual_trees  - Two interleaved trees (liver, kidney)
  [3] path        - Simple inlet-to-outlet channel
  [4] backbone    - Main trunk with side branches
  [5] loop        - Circular/recirculating network

Enter number or name: """

    DOMAIN_TYPE = """
Select domain shape:

  [1] box       - Rectangular box
  [2] ellipsoid - Ellipsoid/spheroid
  [3] cylinder  - Cylindrical

Enter number or name: """

    FACE_SELECTION = """
Select a face:

  [1] left   (x_min)    [2] right  (x_max)
  [3] front  (y_min)    [4] back   (y_max)
  [5] bottom (z_min)    [6] top    (z_max)

Enter number or name: """

    DIMENSION_INPUT = """
Enter dimensions (width x depth x height):

  Format: 20x60x30 mm  or  2x6x3 cm
  Example: 20x60x30 mm

Dimensions: """

    RADIUS_INPUT = """
Enter radius:

  Format: 2 mm  or  0.002 m
  Example: 2 mm

Radius: """

    @classmethod
    def get_template(cls, field_name: str) -> Optional[str]:
        """Get input template for a field."""
        templates = {
            "topology.kind": cls.TOPOLOGY_SELECTION,
            "domain.type": cls.DOMAIN_TYPE,
            "inlet.face": cls.FACE_SELECTION,
            "outlet.face": cls.FACE_SELECTION,
            "domain.size": cls.DIMENSION_INPUT,
            "inlet.radius": cls.RADIUS_INPUT,
            "outlet.radius": cls.RADIUS_INPUT,
        }
        return templates.get(field_name)


class TextFormatter:
    """Utilities for formatting text output."""
    
    @staticmethod
    def format_table(headers: List[str], rows: List[List[str]], align: str = "left") -> str:
        """Format data as a text table."""
        if not rows:
            return "(empty table)"
        
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        def format_row(cells: List[str]) -> str:
            formatted = []
            for i, cell in enumerate(cells):
                width = col_widths[i] if i < len(col_widths) else len(str(cell))
                if align == "right":
                    formatted.append(str(cell).rjust(width))
                elif align == "center":
                    formatted.append(str(cell).center(width))
                else:
                    formatted.append(str(cell).ljust(width))
            return "| " + " | ".join(formatted) + " |"
        
        separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        
        lines = [separator]
        lines.append(format_row(headers))
        lines.append(separator)
        for row in rows:
            lines.append(format_row(row))
        lines.append(separator)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_key_value(data: Dict[str, Any], indent: int = 0) -> str:
        """Format key-value pairs."""
        lines = []
        prefix = "  " * indent
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(TextFormatter.format_key_value(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}: [{', '.join(str(v) for v in value)}]")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)
    
    @staticmethod
    def format_code_block(code: str, language: str = "") -> str:
        """Format code in a code block."""
        return f"```{language}\n{code}\n```"
    
    @staticmethod
    def format_progress_bar(current: int, total: int, width: int = 40) -> str:
        """Format a text progress bar."""
        if total == 0:
            return "[" + "?" * width + "] ?%"
        
        percent = current / total
        filled = int(width * percent)
        bar = "=" * filled + "-" * (width - filled)
        return f"[{bar}] {percent*100:.1f}%"
    
    @staticmethod
    def format_status_indicator(status: str) -> str:
        """Format a status indicator."""
        indicators = {
            "success": "[OK]",
            "error": "[ERROR]",
            "warning": "[WARN]",
            "info": "[INFO]",
            "pending": "[...]",
            "running": "[>>>]",
        }
        return indicators.get(status.lower(), f"[{status.upper()}]")


class DesignSummaryGenerator:
    """Generate text summaries of vascular network designs."""
    
    @staticmethod
    def generate_spec_summary(spec: Dict[str, Any]) -> str:
        """Generate a text summary of a design specification."""
        report = TextReport(title="DESIGN SPECIFICATION SUMMARY")
        
        domain = spec.get("domain", {})
        domain_text = f"""
Type: {domain.get('type', 'not specified')}
Size: {domain.get('size', 'not specified')}
"""
        report.add_section("Domain", domain_text.strip())
        
        topology = spec.get("topology", {})
        topology_text = f"""
Kind: {topology.get('kind', 'not specified')}
Target Terminals: {topology.get('target_terminals', 'not specified')}
"""
        report.add_section("Topology", topology_text.strip())
        
        inlet = spec.get("inlet", {})
        inlet_text = f"""
Face: {inlet.get('face', 'not specified')}
Position: {inlet.get('position', 'derived from face')}
Radius: {inlet.get('radius', 'not specified')}
"""
        report.add_section("Inlet", inlet_text.strip())
        
        outlet = spec.get("outlet", {})
        outlet_text = f"""
Face: {outlet.get('face', 'not specified')}
Position: {outlet.get('position', 'derived from face')}
Radius: {outlet.get('radius', 'not specified')}
"""
        report.add_section("Outlet", outlet_text.strip())
        
        return report.to_text()
    
    @staticmethod
    def generate_validation_summary(validation_result: Dict[str, Any]) -> str:
        """Generate a text summary of validation results."""
        report = TextReport(title="VALIDATION REPORT")
        
        passed = validation_result.get("passed", False)
        status = "PASSED" if passed else "FAILED"
        report.summary = f"Overall Status: {status}"
        
        checks = validation_result.get("checks", [])
        if checks:
            rows = []
            for check in checks:
                name = check.get("name", "Unknown")
                result = "Pass" if check.get("passed", False) else "Fail"
                message = check.get("message", "")[:40]
                rows.append([name, result, message])
            
            table = TextFormatter.format_table(
                ["Check", "Result", "Message"],
                rows
            )
            report.add_section("Validation Checks", table)
        
        errors = validation_result.get("errors", [])
        if errors:
            error_text = "\n".join(f"  - {e}" for e in errors)
            report.add_section("Errors", error_text)
        
        warnings = validation_result.get("warnings", [])
        if warnings:
            warning_text = "\n".join(f"  - {w}" for w in warnings)
            report.add_section("Warnings", warning_text)
        
        return report.to_text()
    
    @staticmethod
    def generate_generation_summary(result: Dict[str, Any]) -> str:
        """Generate a text summary of generation results."""
        report = TextReport(title="GENERATION RESULTS")
        
        stats = result.get("statistics", {})
        if stats:
            stats_text = TextFormatter.format_key_value(stats)
            report.add_section("Statistics", stats_text)
        
        outputs = result.get("outputs", [])
        if outputs:
            output_text = "\n".join(f"  - {o}" for o in outputs)
            report.add_section("Output Files", output_text)
        
        timing = result.get("timing", {})
        if timing:
            timing_text = TextFormatter.format_key_value(timing)
            report.add_section("Timing", timing_text)
        
        return report.to_text()


class CommandHelp:
    """Help text for available commands."""
    
    COMMANDS = {
        "place": {
            "syntax": "place <element> at <location>",
            "description": "Position an element at a specific location",
            "examples": [
                "place inlet at left face",
                "place outlet at [10, 30, 15] mm",
                "place inlet 2cm above center",
            ],
        },
        "connect": {
            "syntax": "connect <source> to <target>",
            "description": "Connect two elements",
            "examples": [
                "connect inlet to outlet",
            ],
        },
        "set": {
            "syntax": "set <parameter> to <value>",
            "description": "Set a parameter value",
            "examples": [
                "set topology to tree",
                "set domain size to 20x60x30 mm",
                "set inlet radius to 2 mm",
            ],
        },
        "undo": {
            "syntax": "undo [to <entry_id>]",
            "description": "Undo the last change or revert to a specific entry",
            "examples": [
                "undo",
                "undo to entry_5",
            ],
        },
        "status": {
            "syntax": "status",
            "description": "Show current specification status",
            "examples": ["status", "show status", "what's missing"],
        },
        "help": {
            "syntax": "help [command]",
            "description": "Show help for commands",
            "examples": ["help", "help place"],
        },
    }
    
    @classmethod
    def get_help(cls, command: Optional[str] = None) -> str:
        """Get help text for a command or all commands."""
        if command and command in cls.COMMANDS:
            cmd = cls.COMMANDS[command]
            return f"""
Command: {command}
Syntax: {cmd['syntax']}
Description: {cmd['description']}

Examples:
{chr(10).join('  ' + e for e in cmd['examples'])}
"""
        
        lines = ["Available Commands:", ""]
        for name, cmd in cls.COMMANDS.items():
            lines.append(f"  {name:10} - {cmd['description']}")
        lines.append("")
        lines.append("Type 'help <command>' for detailed help on a specific command.")
        
        return "\n".join(lines)


def format_workflow_state(state: Dict[str, Any]) -> str:
    """Format workflow state as readable text."""
    report = TextReport(title="WORKFLOW STATE")
    
    status = state.get("status", "unknown")
    report.summary = f"Status: {TextFormatter.format_status_indicator(status)} {status}"
    
    facts = state.get("facts", {})
    if facts:
        fact_rows = []
        for key, value in facts.items():
            fact_rows.append([key, str(value)[:50]])
        table = TextFormatter.format_table(["Field", "Value"], fact_rows)
        report.add_section("Current Facts", table)
    
    missing = state.get("missing_fields", [])
    if missing:
        missing_text = "\n".join(f"  - {f}" for f in missing)
        report.add_section("Missing Fields", missing_text)
    
    return report.to_text()


def format_error_message(error: str, context: Optional[Dict[str, Any]] = None, verbose: bool = False) -> str:
    """Format an error message with optional context."""
    lines = [f"ERROR: {error}"]
    
    if context and verbose:
        lines.append("")
        lines.append("Context:")
        for key, value in context.items():
            lines.append(f"  {key}: {value}")
    
    lines.append("")
    lines.append("Suggestions:")
    lines.append("  - Check your input for typos")
    lines.append("  - Type 'help' for available commands")
    lines.append("  - Type 'status' to see current state")
    
    return "\n".join(lines)


def get_coordinate_system_help() -> str:
    """Get help text for the coordinate system."""
    return """
COORDINATE SYSTEM
=================

The system uses a right-handed Cartesian coordinate system:

  +Z (top)
    |
    |
    +---- +X (right)
   /
  /
 +Y (back)

Domain Faces:
  LEFT   = x_min (-X)    RIGHT  = x_max (+X)
  FRONT  = y_min (-Y)    BACK   = y_max (+Y)
  BOTTOM = z_min (-Z)    TOP    = z_max (+Z)

Units: millimeters (mm) by default
       Also supported: m, cm, um

Position Formats:
  - Face name: "left", "right", "top", etc.
  - Coordinates: [x, y, z] mm
  - Relative: "2cm above inlet"
"""
