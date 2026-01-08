"""
Agent Dialogue Module

Implements the "Interpret -> Plan -> Ask" flow that makes the agent feel like
a real designer rather than a form/prompt machine.

The dialogue system:
1. Understands: Paraphrases user intent, identifies assumptions and ambiguities
2. Plans: Proposes 2-3 viable generation strategies
3. Asks: Generates minimal questions to unblock the chosen plan
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import json
import re


# =============================================================================
# Data Classes for Understanding
# =============================================================================

@dataclass
class Assumption:
    """An assumption made about the user's intent."""
    field: str
    value: Any
    confidence: float  # 0.0 to 1.0
    reason: str


@dataclass
class Ambiguity:
    """An ambiguity detected in the user's description."""
    field: str
    description: str
    options: List[str]
    impact: str  # "high", "medium", "low"


@dataclass
class Risk:
    """A potential risk identified in the design."""
    category: str  # "printability", "collision", "units", "complexity", "feasibility"
    description: str
    severity: str  # "critical", "warning", "info"
    mitigation: Optional[str] = None


@dataclass
class InitialRequirementsDraft:
    """Initial requirements extracted from user intent with confidence scores."""
    domain_type: Optional[str] = None
    domain_size: Optional[Tuple[float, float, float]] = None
    num_inlets: Optional[int] = None
    num_outlets: Optional[int] = None
    inlet_radius: Optional[float] = None
    outlet_radius: Optional[float] = None
    target_terminals: Optional[int] = None
    min_radius: Optional[float] = None
    min_clearance: Optional[float] = None
    topology_style: Optional[str] = None
    
    # Confidence scores for each field (0.0 to 1.0)
    confidence: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_type": self.domain_type,
            "domain_size": list(self.domain_size) if self.domain_size else None,
            "num_inlets": self.num_inlets,
            "num_outlets": self.num_outlets,
            "inlet_radius": self.inlet_radius,
            "outlet_radius": self.outlet_radius,
            "target_terminals": self.target_terminals,
            "min_radius": self.min_radius,
            "min_clearance": self.min_clearance,
            "topology_style": self.topology_style,
            "confidence": self.confidence,
        }


@dataclass
class UnderstandingReport:
    """
    Complete understanding of the user's intent.
    
    This is produced by the understand_object() function and contains:
    - A summary paragraph of what the user wants
    - Assumptions made (with confidence levels)
    - Ambiguities detected
    - Risks identified
    - Initial requirements draft
    """
    summary: str
    assumptions: List[Assumption]
    ambiguities: List[Ambiguity]
    risks: List[Risk]
    requirements_draft: InitialRequirementsDraft
    organ_type: str = "generic"
    raw_intent: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "assumptions": [
                {"field": a.field, "value": a.value, "confidence": a.confidence, "reason": a.reason}
                for a in self.assumptions
            ],
            "ambiguities": [
                {"field": a.field, "description": a.description, "options": a.options, "impact": a.impact}
                for a in self.ambiguities
            ],
            "risks": [
                {"category": r.category, "description": r.description, "severity": r.severity, "mitigation": r.mitigation}
                for r in self.risks
            ],
            "requirements_draft": self.requirements_draft.to_dict(),
            "organ_type": self.organ_type,
        }


# =============================================================================
# Data Classes for Planning
# =============================================================================

class GeneratorStrategy(Enum):
    """Available generation strategies."""
    TREE_BACKBONE = "tree_backbone"
    SPACE_COLONIZATION = "space_colonization"
    REGION_WEIGHTED = "region_weighted"
    MULTI_ROOT = "multi_root"
    DUAL_TREE = "dual_tree"


@dataclass
class PlanOption:
    """
    A proposed generation plan.
    
    Each plan describes:
    - The generation strategy to use
    - Which repo utilities will be used
    - What parameters/knobs will be tuned
    - Which schema modules are required
    """
    id: str  # "A", "B", "C"
    name: str
    description: str
    strategy: GeneratorStrategy
    utilities: List[str]  # e.g., ["space_colonization", "embed_in_domain", "export_stl"]
    tunable_knobs: List[str]  # e.g., ["attraction_distance", "kill_distance", "step_size"]
    required_modules: List[str]  # e.g., ["TopologyModule", "GeometryStyleModule"]
    pros: List[str]
    cons: List[str]
    estimated_complexity: str  # "low", "medium", "high"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "strategy": self.strategy.value,
            "utilities": self.utilities,
            "tunable_knobs": self.tunable_knobs,
            "required_modules": self.required_modules,
            "pros": self.pros,
            "cons": self.cons,
            "estimated_complexity": self.estimated_complexity,
        }


# =============================================================================
# Understanding Functions
# =============================================================================

def detect_organ_type(intent: str) -> str:
    """
    Detect the organ type from the user's intent description.
    
    Returns the organ key (liver, kidney, lung, heart, generic) based on
    keywords found in the intent string.
    """
    intent_lower = intent.lower()
    
    organ_keywords = {
        "liver": ["liver", "hepatic", "hepato", "lobule", "sinusoid", "portal"],
        "kidney": ["kidney", "renal", "nephron", "glomerul", "cortex", "medulla", "ureter"],
        "lung": ["lung", "pulmonary", "bronch", "alveol", "respiratory", "airway"],
        "heart": ["heart", "coronary", "cardiac", "myocard", "ventricle", "atrium", "aortic"],
    }
    
    for organ, keywords in organ_keywords.items():
        for keyword in keywords:
            if keyword in intent_lower:
                return organ
    
    return "generic"


def extract_numeric_values(intent: str) -> Dict[str, Any]:
    """
    Extract numeric values from the user's intent.
    
    Looks for patterns like:
    - "20mm x 60mm x 30mm" -> domain size
    - "0.5mm radius" -> radius
    - "100 terminals" -> terminal count
    """
    extracted = {}
    
    # Domain size patterns
    size_patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:mm|cm|m)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)?",
        r"(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)",
    ]
    
    for pattern in size_patterns:
        match = re.search(pattern, intent)
        if match:
            groups = match.groups()
            unit = groups[3] if len(groups) > 3 and groups[3] else "mm"
            scale = {"mm": 0.001, "cm": 0.01, "m": 1.0}.get(unit, 0.001)
            extracted["domain_size"] = (
                float(groups[0]) * scale,
                float(groups[1]) * scale,
                float(groups[2]) * scale,
            )
            extracted["domain_size_unit"] = unit
            break
    
    # Radius patterns
    radius_patterns = [
        r"(\d+(?:\.\d+)?)\s*(mm|um|cm)?\s*(?:radius|diameter)",
        r"(?:radius|diameter)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(mm|um|cm)?",
    ]
    
    for pattern in radius_patterns:
        match = re.search(pattern, intent.lower())
        if match:
            value = float(match.group(1))
            unit = match.group(2) if match.group(2) else "mm"
            scale = {"mm": 0.001, "um": 0.000001, "cm": 0.01}.get(unit, 0.001)
            
            # Check if it's diameter (convert to radius)
            if "diameter" in intent.lower():
                value = value / 2
            
            extracted["radius"] = value * scale
            break
    
    # Terminal count patterns
    terminal_patterns = [
        r"(\d+)\s*(?:terminals?|tips?|endpoints?|branches?)",
        r"(?:terminals?|tips?|endpoints?)\s*(?:count\s*)?(?:of\s*)?(\d+)",
    ]
    
    for pattern in terminal_patterns:
        match = re.search(pattern, intent.lower())
        if match:
            extracted["target_terminals"] = int(match.group(1))
            break
    
    # Inlet/outlet count patterns
    io_patterns = [
        r"(\d+)\s*inlets?",
        r"(\d+)\s*outlets?",
    ]
    
    for pattern in io_patterns:
        match = re.search(pattern, intent.lower())
        if match:
            if "inlet" in pattern:
                extracted["num_inlets"] = int(match.group(1))
            else:
                extracted["num_outlets"] = int(match.group(1))
    
    return extracted


def detect_spatial_terms(intent: str) -> List[str]:
    """Detect spatial terms that require frame of reference."""
    spatial_terms = [
        "left", "right", "top", "bottom", "front", "back",
        "upper", "lower", "above", "below", "beside", "near",
        "center", "edge", "corner", "side", "face",
    ]
    
    intent_lower = intent.lower()
    found = []
    
    for term in spatial_terms:
        if re.search(rf"\b{term}\b", intent_lower):
            found.append(term)
    
    return found


def detect_vague_quantifiers(intent: str) -> List[str]:
    """Detect vague quantifiers that need clarification."""
    vague_terms = [
        "dense", "sparse", "many", "few", "some", "several",
        "thick", "thin", "large", "small", "big", "tiny",
        "highly", "moderately", "slightly", "very",
        "complex", "simple", "branched", "tortuous", "smooth",
    ]
    
    intent_lower = intent.lower()
    found = []
    
    for term in vague_terms:
        if re.search(rf"\b{term}\b", intent_lower):
            found.append(term)
    
    return found


def understand_object(
    user_text: str,
    context: Optional[Dict[str, Any]] = None
) -> UnderstandingReport:
    """
    Understand the user's intent and produce an UnderstandingReport.
    
    This function:
    1. Detects the organ type
    2. Extracts explicit numeric values
    3. Identifies assumptions that need to be made
    4. Detects ambiguities
    5. Identifies potential risks
    6. Creates an initial requirements draft
    
    Parameters
    ----------
    user_text : str
        The user's description of what they want to generate
    context : dict, optional
        Additional context (e.g., project defaults, previous objects)
        
    Returns
    -------
    UnderstandingReport
        Complete understanding of the user's intent
    """
    context = context or {}
    
    # Detect organ type
    organ_type = detect_organ_type(user_text)
    
    # Extract numeric values
    extracted = extract_numeric_values(user_text)
    
    # Detect spatial terms and vague quantifiers
    spatial_terms = detect_spatial_terms(user_text)
    vague_terms = detect_vague_quantifiers(user_text)
    
    # Build assumptions
    assumptions = []
    
    # Domain assumptions
    if "domain_size" not in extracted:
        default_size = context.get("default_embed_domain", (0.02, 0.06, 0.03))
        assumptions.append(Assumption(
            field="domain.size_m",
            value=default_size,
            confidence=0.5,
            reason="No explicit size specified; using default embed domain"
        ))
    
    # Topology assumptions based on organ type
    if organ_type != "generic":
        assumptions.append(Assumption(
            field="topology.style",
            value="tree",
            confidence=0.8,
            reason=f"Vascular networks for {organ_type} typically use tree topology"
        ))
    
    # I/O assumptions
    if "num_inlets" not in extracted:
        default_inlets = 1
        if organ_type == "liver":
            default_inlets = 2  # Hepatic artery + portal vein
        elif organ_type == "heart":
            default_inlets = 2  # LCA + RCA
        assumptions.append(Assumption(
            field="inlets_outlets.num_inlets",
            value=default_inlets,
            confidence=0.6,
            reason=f"Default inlet count for {organ_type}"
        ))
    
    # Build ambiguities
    ambiguities = []
    
    # Spatial ambiguity
    if spatial_terms:
        ambiguities.append(Ambiguity(
            field="frame_of_reference",
            description=f"Spatial terms used ({', '.join(spatial_terms)}) without coordinate convention",
            options=["Confirm default axes (X=width, Y=depth, Z=height)", "Specify custom axes"],
            impact="high"
        ))
    
    # Vague quantifier ambiguity
    for term in vague_terms:
        if term in ["dense", "sparse"]:
            ambiguities.append(Ambiguity(
                field="topology.target_terminals",
                description=f"'{term}' needs numeric mapping for terminal count",
                options=["50-100 (sparse)", "200-300 (medium)", "500-1000 (dense)"],
                impact="medium"
            ))
        elif term in ["thick", "thin"]:
            ambiguities.append(Ambiguity(
                field="constraints.min_radius_m",
                description=f"'{term}' needs numeric mapping for radius",
                options=["0.05-0.1mm (thin)", "0.1-0.2mm (medium)", "0.2-0.5mm (thick)"],
                impact="high"
            ))
    
    # Build risks
    risks = []
    
    # Printability risk
    if "radius" in extracted and extracted["radius"] < 0.0001:  # < 0.1mm
        risks.append(Risk(
            category="printability",
            description=f"Specified radius ({extracted['radius']*1000:.2f}mm) may be below printable threshold",
            severity="warning",
            mitigation="Verify printer capabilities or increase minimum radius"
        ))
    
    # Complexity risk
    if "target_terminals" in extracted and extracted["target_terminals"] > 1000:
        risks.append(Risk(
            category="complexity",
            description=f"High terminal count ({extracted['target_terminals']}) may cause long generation time",
            severity="warning",
            mitigation="Consider reducing terminal count or using complexity budget"
        ))
    
    # Unit risk
    if not any(unit in user_text.lower() for unit in ["mm", "cm", "m", "um"]):
        risks.append(Risk(
            category="units",
            description="No units specified in description",
            severity="info",
            mitigation="Confirm units before generation (default: mm for export)"
        ))
    
    # Build initial requirements draft
    requirements_draft = InitialRequirementsDraft(
        domain_type="box",
        domain_size=extracted.get("domain_size"),
        num_inlets=extracted.get("num_inlets"),
        num_outlets=extracted.get("num_outlets"),
        inlet_radius=extracted.get("radius"),
        target_terminals=extracted.get("target_terminals"),
        topology_style="tree" if organ_type != "generic" else None,
    )
    
    # Set confidence scores
    for field in ["domain_size", "num_inlets", "num_outlets", "inlet_radius", "target_terminals"]:
        if field in extracted:
            requirements_draft.confidence[field] = 0.9  # High confidence for explicit values
        else:
            requirements_draft.confidence[field] = 0.3  # Low confidence for defaults
    
    # Build summary
    summary_parts = [f"You want to generate a {organ_type} vascular network"]
    
    if "domain_size" in extracted:
        size = extracted["domain_size"]
        summary_parts.append(f"with domain size {size[0]*1000:.0f}mm x {size[1]*1000:.0f}mm x {size[2]*1000:.0f}mm")
    
    if "target_terminals" in extracted:
        summary_parts.append(f"targeting {extracted['target_terminals']} terminal branches")
    
    if vague_terms:
        summary_parts.append(f"with {', '.join(vague_terms)} characteristics")
    
    summary = ". ".join(summary_parts) + "."
    
    return UnderstandingReport(
        summary=summary,
        assumptions=assumptions,
        ambiguities=ambiguities,
        risks=risks,
        requirements_draft=requirements_draft,
        organ_type=organ_type,
        raw_intent=user_text,
    )


# =============================================================================
# Planning Functions
# =============================================================================

def propose_plans(
    understanding: UnderstandingReport,
    context: Optional[Dict[str, Any]] = None
) -> List[PlanOption]:
    """
    Propose 2-3 viable generation strategies based on the understanding.
    
    Parameters
    ----------
    understanding : UnderstandingReport
        The understanding report from understand_object()
    context : dict, optional
        Additional context
        
    Returns
    -------
    List[PlanOption]
        2-3 plan options for the user to choose from
    """
    context = context or {}
    plans = []
    
    organ_type = understanding.organ_type
    draft = understanding.requirements_draft
    
    # Plan A: Space Colonization (default for most cases)
    plans.append(PlanOption(
        id="A",
        name="Space Colonization Growth",
        description="Grow the network organically using space colonization algorithm. "
                   "Branches grow toward attractor points, creating natural-looking structures.",
        strategy=GeneratorStrategy.SPACE_COLONIZATION,
        utilities=["space_colonization", "bifurcate", "embed_in_domain", "export_stl"],
        tunable_knobs=["attraction_distance", "kill_distance", "step_size", "branching_angle"],
        required_modules=["TopologyModule", "GeometryStyleModule"],
        pros=[
            "Natural-looking organic growth",
            "Good coverage of domain",
            "Handles complex shapes well",
        ],
        cons=[
            "Less control over exact topology",
            "May need multiple iterations for desired density",
        ],
        estimated_complexity="medium",
    ))
    
    # Plan B: Tree Backbone (for structured networks)
    plans.append(PlanOption(
        id="B",
        name="Structured Tree Backbone",
        description="Build a hierarchical tree structure with controlled branching. "
                   "More predictable topology with explicit depth and branching control.",
        strategy=GeneratorStrategy.TREE_BACKBONE,
        utilities=["create_network", "bifurcate", "embed_in_domain", "export_stl"],
        tunable_knobs=["max_depth", "branching_factor", "length_ratio", "radius_ratio"],
        required_modules=["TopologyModule", "ConstraintsModule"],
        pros=[
            "Predictable topology",
            "Easy to control branching depth",
            "Faster generation",
        ],
        cons=[
            "Less organic appearance",
            "May not fill irregular domains well",
        ],
        estimated_complexity="low",
    ))
    
    # Plan C: Organ-specific or advanced (based on organ type)
    if organ_type == "liver":
        plans.append(PlanOption(
            id="C",
            name="Dual-Tree Hepatic Network",
            description="Generate both arterial and portal venous trees with meeting shell "
                       "for anastomosis. Specific to liver vasculature.",
            strategy=GeneratorStrategy.DUAL_TREE,
            utilities=["generate_liver_vasculature", "create_anastomosis", "embed_in_domain", "export_stl"],
            tunable_knobs=["arterial_segments", "venous_segments", "meeting_shell_radius"],
            required_modules=["TopologyModule", "GeometryStyleModule", "PerfusionZonesModule"],
            pros=[
                "Anatomically accurate for liver",
                "Includes both vascular systems",
                "Proper anastomosis handling",
            ],
            cons=[
                "More complex setup",
                "Longer generation time",
                "Specific to liver only",
            ],
            estimated_complexity="high",
        ))
    elif organ_type in ("kidney", "lung", "heart"):
        plans.append(PlanOption(
            id="C",
            name="Region-Weighted Growth",
            description="Use region-weighted space colonization with density multipliers "
                       "for different anatomical zones.",
            strategy=GeneratorStrategy.REGION_WEIGHTED,
            utilities=["space_colonization", "zone_density", "embed_in_domain", "export_stl"],
            tunable_knobs=["zone_densities", "attraction_distance", "kill_distance"],
            required_modules=["TopologyModule", "GeometryStyleModule", "PerfusionZonesModule"],
            pros=[
                "Anatomically appropriate density distribution",
                "Good for organs with distinct regions",
                "Flexible zone configuration",
            ],
            cons=[
                "Requires zone definition",
                "More parameters to tune",
            ],
            estimated_complexity="medium",
        ))
    else:
        plans.append(PlanOption(
            id="C",
            name="Multi-Root Network",
            description="Generate network from multiple inlet points simultaneously. "
                       "Good for distributed perfusion or multiple supply vessels.",
            strategy=GeneratorStrategy.MULTI_ROOT,
            utilities=["space_colonization", "multi_root_growth", "embed_in_domain", "export_stl"],
            tunable_knobs=["num_roots", "root_positions", "attraction_distance"],
            required_modules=["TopologyModule", "GeometryStyleModule"],
            pros=[
                "Good for multiple inlets",
                "Even coverage from multiple sources",
                "Flexible root placement",
            ],
            cons=[
                "More complex I/O setup",
                "May have overlapping territories",
            ],
            estimated_complexity="medium",
        ))
    
    return plans


# =============================================================================
# Formatting Functions
# =============================================================================

def format_user_summary(
    understanding: UnderstandingReport,
    plans: List[PlanOption]
) -> str:
    """
    Format the understanding and plans for display to the user.
    
    Parameters
    ----------
    understanding : UnderstandingReport
        The understanding report
    plans : List[PlanOption]
        The proposed plans
        
    Returns
    -------
    str
        Formatted text for display
    """
    lines = []
    
    # Header
    lines.append("=" * 60)
    lines.append("  Understanding Your Request")
    lines.append("=" * 60)
    lines.append("")
    
    # Summary
    lines.append("What I understood:")
    lines.append(f"  {understanding.summary}")
    lines.append("")
    
    # Assumptions
    if understanding.assumptions:
        lines.append("Assumptions I'm making:")
        for a in understanding.assumptions:
            confidence_str = f"({a.confidence*100:.0f}% confident)"
            lines.append(f"  - {a.field}: {a.value} {confidence_str}")
            lines.append(f"    Reason: {a.reason}")
        lines.append("")
    
    # Ambiguities
    if understanding.ambiguities:
        lines.append("Ambiguities to clarify:")
        for a in understanding.ambiguities:
            impact_marker = {"high": "[!]", "medium": "[?]", "low": "[i]"}.get(a.impact, "[-]")
            lines.append(f"  {impact_marker} {a.description}")
            lines.append(f"      Options: {', '.join(a.options)}")
        lines.append("")
    
    # Risks
    if understanding.risks:
        lines.append("Potential risks:")
        for r in understanding.risks:
            severity_marker = {"critical": "[!!]", "warning": "[!]", "info": "[i]"}.get(r.severity, "[-]")
            lines.append(f"  {severity_marker} {r.category}: {r.description}")
            if r.mitigation:
                lines.append(f"      Mitigation: {r.mitigation}")
        lines.append("")
    
    # Plans
    lines.append("=" * 60)
    lines.append("  Proposed Generation Strategies")
    lines.append("=" * 60)
    lines.append("")
    
    for plan in plans:
        lines.append(f"[{plan.id}] {plan.name}")
        lines.append(f"    {plan.description}")
        lines.append(f"    Complexity: {plan.estimated_complexity}")
        lines.append(f"    Pros: {', '.join(plan.pros[:2])}")
        lines.append(f"    Cons: {', '.join(plan.cons[:2])}")
        lines.append("")
    
    lines.append("Which plan would you like to use? (A/B/C)")
    
    return "\n".join(lines)


def format_living_spec(
    understanding: UnderstandingReport,
    chosen_plan: Optional[PlanOption] = None,
    active_modules: Optional[List[str]] = None,
    missing_fields: Optional[List[str]] = None
) -> str:
    """
    Format a "living spec" view showing current state of requirements.
    
    Parameters
    ----------
    understanding : UnderstandingReport
        The understanding report
    chosen_plan : PlanOption, optional
        The chosen plan (if selected)
    active_modules : list, optional
        Currently active schema modules
    missing_fields : list, optional
        Fields still missing
        
    Returns
    -------
    str
        Formatted living spec view
    """
    lines = []
    draft = understanding.requirements_draft
    
    lines.append("-" * 40)
    lines.append("  Current Specification")
    lines.append("-" * 40)
    
    # Domain
    if draft.domain_size:
        size = draft.domain_size
        lines.append(f"Domain: {draft.domain_type} {size[0]*1000:.0f}x{size[1]*1000:.0f}x{size[2]*1000:.0f}mm")
    else:
        lines.append("Domain: [not specified]")
    
    # I/O
    inlets = draft.num_inlets if draft.num_inlets else "[not specified]"
    outlets = draft.num_outlets if draft.num_outlets else "[not specified]"
    lines.append(f"Inlets: {inlets}, Outlets: {outlets}")
    
    # Constraints
    if draft.min_radius:
        lines.append(f"Min radius: {draft.min_radius*1000:.2f}mm")
    else:
        lines.append("Min radius: [not specified]")
    
    # Topology
    if draft.target_terminals:
        lines.append(f"Target terminals: {draft.target_terminals}")
    else:
        lines.append("Target terminals: [not specified]")
    
    # Plan
    if chosen_plan:
        lines.append(f"Strategy: {chosen_plan.name}")
    else:
        lines.append("Strategy: [not selected]")
    
    # Active modules
    if active_modules:
        lines.append(f"Active modules: {', '.join(active_modules)}")
    
    # Missing fields
    if missing_fields:
        lines.append("")
        lines.append("Still needed:")
        for field in missing_fields[:5]:  # Show max 5
            lines.append(f"  - {field}")
    
    lines.append("-" * 40)
    
    return "\n".join(lines)
