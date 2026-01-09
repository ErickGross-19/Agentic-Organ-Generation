"""
Concept & Spec Agent (CSA)

The CSA is responsible for:
- Conversing with the user to understand their intent
- Converting intent into a versioned, canonical specification
- Managing spec versions (only CSA can increment versions)
- Documenting assumptions, risks, and decisions
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from ..models import (
    SpecVersion,
    RiskFlag,
    VersionStatus,
    AgentType,
    get_timestamp,
)
from ..folder_manager import FolderManager


@dataclass
class CSASession:
    """
    A session with the Concept & Spec Agent.
    
    Tracks the conversation, decisions, and outputs for a single
    spec creation or refinement session.
    """
    session_id: str
    spec_version: int
    started_at: str
    conversation_log: List[Dict[str, str]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    risks_identified: List[RiskFlag] = field(default_factory=list)
    completed_at: Optional[str] = None
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation log."""
        self.conversation_log.append({
            "role": role,
            "content": content,
            "timestamp": get_timestamp(),
        })
    
    def add_decision(self, decision: str, rationale: str, field_affected: str) -> None:
        """Record a decision made during the session."""
        self.decisions.append({
            "decision": decision,
            "rationale": rationale,
            "field_affected": field_affected,
            "timestamp": get_timestamp(),
        })
    
    def add_assumption(self, assumption: str) -> None:
        """Record an assumption made during the session."""
        self.assumptions.append(assumption)
    
    def add_risk(self, risk: RiskFlag) -> None:
        """Record a risk identified during the session."""
        self.risks_identified.append(risk)
    
    def complete(self) -> None:
        """Mark the session as complete."""
        self.completed_at = get_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "spec_version": self.spec_version,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "conversation_log": self.conversation_log,
            "decisions": self.decisions,
            "assumptions": self.assumptions,
            "risks_identified": [r.to_dict() for r in self.risks_identified],
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CSASession":
        return cls(
            session_id=d["session_id"],
            spec_version=d["spec_version"],
            started_at=d["started_at"],
            completed_at=d.get("completed_at"),
            conversation_log=d.get("conversation_log", []),
            decisions=d.get("decisions", []),
            assumptions=d.get("assumptions", []),
            risks_identified=[RiskFlag.from_dict(r) for r in d.get("risks_identified", [])],
        )


@dataclass
class CSAOutput:
    """
    Output from a CSA session.
    
    Contains all the files that CSA must produce.
    """
    spec_version: SpecVersion
    session: CSASession
    summary_md: str
    risk_flags_json: List[RiskFlag]
    decisions_json: List[Dict[str, Any]]
    changelog_entry: str


class ConceptSpecAgent:
    """
    Concept & Spec Agent (CSA)
    
    Responsible for:
    - Converting user intent into canonical specifications
    - Managing spec versions (only agent that can increment versions)
    - Documenting assumptions, risks, and decisions
    - Producing required output files
    
    Required outputs per session:
    - spec_v###.json (canonical specification)
    - spec_v###_summary.md (plain English summary)
    - spec_v###_risk_flags.json (warnings + severity + mitigation)
    - CSA_session_v###.md (conversation and decisions)
    - CSA_decisions_v###.json (structured decisions)
    - Update to spec_changelog.md
    """
    
    def __init__(self, folder_manager: FolderManager, llm_client: Optional[Any] = None):
        """
        Initialize the CSA.
        
        Parameters
        ----------
        folder_manager : FolderManager
            Folder manager for the object
        llm_client : Any, optional
            LLM client for generating specs (if None, manual mode)
        """
        self.folder_manager = folder_manager
        self.llm_client = llm_client
        self.current_session: Optional[CSASession] = None
    
    def get_next_version(self) -> int:
        """Get the next available version number."""
        existing = self.folder_manager.get_existing_versions()
        if not existing:
            return 1
        return max(existing) + 1
    
    def start_session(self, parent_version: Optional[int] = None) -> CSASession:
        """
        Start a new CSA session.
        
        Parameters
        ----------
        parent_version : int, optional
            Version to base this session on (for refinements)
            
        Returns
        -------
        CSASession
            The new session
        """
        version = self.get_next_version()
        session_id = f"csa_session_{version:03d}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = CSASession(
            session_id=session_id,
            spec_version=version,
            started_at=get_timestamp(),
        )
        
        self.folder_manager.log_event(f"CSA session started: {session_id}")
        
        return self.current_session
    
    def create_spec_from_requirements(
        self,
        requirements: Dict[str, Any],
        user_description: str = "",
        parent_version: Optional[int] = None,
    ) -> CSAOutput:
        """
        Create a new spec version from requirements.
        
        Parameters
        ----------
        requirements : Dict[str, Any]
            Requirements dictionary (from workflow or user input)
        user_description : str
            Original user description/intent
        parent_version : int, optional
            Version this is based on (for refinements)
            
        Returns
        -------
        CSAOutput
            The complete CSA output
        """
        if self.current_session is None:
            self.start_session(parent_version)
        
        session = self.current_session
        
        # Record the user description
        if user_description:
            session.add_message("user", user_description)
        
        # Convert requirements to spec data
        spec_data = self._requirements_to_spec(requirements)
        
        # Identify risks
        risks = self._identify_risks(spec_data, requirements)
        for risk in risks:
            session.add_risk(risk)
        
        # Generate summary
        summary = self._generate_summary(spec_data, requirements)
        
        # Create spec version
        spec_version = SpecVersion(
            version=session.spec_version,
            spec_data=spec_data,
            status=VersionStatus.PENDING_APPROVAL,
            summary=summary,
            risk_flags=risks,
            assumptions=session.assumptions,
            changelog_entry=self._generate_changelog_entry(requirements, parent_version),
            parent_version=parent_version,
        )
        
        # Complete session
        session.complete()
        
        # Create output
        output = CSAOutput(
            spec_version=spec_version,
            session=session,
            summary_md=self._format_summary_md(spec_version),
            risk_flags_json=risks,
            decisions_json=session.decisions,
            changelog_entry=spec_version.changelog_entry,
        )
        
        return output
    
    def save_output(self, output: CSAOutput) -> Dict[str, str]:
        """
        Save all CSA output files.
        
        Parameters
        ----------
        output : CSAOutput
            The CSA output to save
            
        Returns
        -------
        Dict[str, str]
            Dictionary of file types to paths
        """
        version = output.spec_version.version
        paths = {}
        
        # Save spec JSON
        spec_path = self.folder_manager.get_spec_path(version)
        with open(spec_path, 'w') as f:
            json.dump(output.spec_version.to_dict(), f, indent=2)
        paths["spec"] = spec_path
        
        # Save summary MD
        summary_path = self.folder_manager.get_spec_summary_path(version)
        with open(summary_path, 'w') as f:
            f.write(output.summary_md)
        paths["summary"] = summary_path
        
        # Save risk flags JSON
        risk_flags_path = self.folder_manager.get_spec_risk_flags_path(version)
        with open(risk_flags_path, 'w') as f:
            json.dump([r.to_dict() for r in output.risk_flags_json], f, indent=2)
        paths["risk_flags"] = risk_flags_path
        
        # Save session MD
        csa_docs_dir = self.folder_manager.get_agent_docs_dir("CSA")
        os.makedirs(csa_docs_dir, exist_ok=True)
        
        session_path = os.path.join(csa_docs_dir, f"CSA_session_v{version:03d}.md")
        with open(session_path, 'w') as f:
            f.write(self._format_session_md(output.session))
        paths["session"] = session_path
        
        # Save decisions JSON
        decisions_path = os.path.join(csa_docs_dir, f"CSA_decisions_v{version:03d}.json")
        with open(decisions_path, 'w') as f:
            json.dump(output.decisions_json, f, indent=2)
        paths["decisions"] = decisions_path
        
        # Update changelog
        self.folder_manager.append_to_spec_changelog(
            version=version,
            summary=output.spec_version.summary,
            changes=[d["decision"] for d in output.decisions_json],
        )
        paths["changelog"] = self.folder_manager.spec_changelog_path
        
        # Update manifest
        manifest = self.folder_manager.load_manifest()
        manifest.active_spec_version = version
        manifest.total_versions = version
        self.folder_manager.save_manifest(manifest)
        
        self.folder_manager.log_event(f"CSA output saved for version {version}")
        
        return paths
    
    def _requirements_to_spec(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Convert requirements dictionary to spec data format."""
        spec_data = {
            "version": "1.0",
            "type": "organ_structure",
        }
        
        # Domain specification
        if "domain" in requirements:
            domain = requirements["domain"]
            spec_data["domain"] = {
                "type": domain.get("type", "box"),
                "size_m": domain.get("size_m", [0.02, 0.06, 0.03]),
                "center_m": domain.get("center_m", [0.0, 0.0, 0.0]),
                "margin_m": domain.get("margin_m", 0.001),
            }
        
        # Topology specification
        if "topology" in requirements:
            topology = requirements["topology"]
            spec_data["topology"] = {
                "kind": topology.get("topology_kind", "tree"),
                "style": topology.get("style", "tree"),
                "target_terminals": topology.get("target_terminals"),
                "max_depth": topology.get("max_depth"),
                "branching_factor_range": topology.get("branching_factor_range", [2, 2]),
            }
        
        # Ports specification
        if "ports" in requirements:
            ports = requirements["ports"]
            spec_data["ports"] = {
                "inlets": ports.get("inlets", []),
                "outlets": ports.get("outlets", []),
            }
        
        # Geometry specification
        if "geometry" in requirements:
            geometry = requirements["geometry"]
            spec_data["geometry"] = {
                "segment_length_m": geometry.get("segment_length_m", {"min": 0.0005, "max": 0.005}),
                "radius_profile": geometry.get("radius_profile", "murray"),
                "radius_bounds_m": geometry.get("radius_bounds_m", {"min": 0.0001, "max": None}),
                "branch_angle_deg": geometry.get("branch_angle_deg", {"min": 30.0, "max": 90.0}),
            }
        
        # Constraints
        if "constraints" in requirements:
            spec_data["constraints"] = requirements["constraints"]
        
        # Embedding/Export
        if "embedding" in requirements:
            spec_data["embedding"] = requirements["embedding"]
        
        # Acceptance criteria
        if "acceptance_criteria" in requirements:
            spec_data["acceptance_criteria"] = requirements["acceptance_criteria"]
        
        return spec_data
    
    def _identify_risks(
        self,
        spec_data: Dict[str, Any],
        requirements: Dict[str, Any],
    ) -> List[RiskFlag]:
        """Identify potential risks in the specification."""
        risks = []
        risk_id = 0
        
        # Check domain size
        if "domain" in spec_data:
            domain = spec_data["domain"]
            size = domain.get("size_m", [0.02, 0.06, 0.03])
            if any(s > 0.1 for s in size):
                risk_id += 1
                risks.append(RiskFlag(
                    id=f"risk_{risk_id:03d}",
                    severity="medium",
                    category="geometry",
                    description="Large domain size may result in long generation times",
                    mitigation="Consider reducing domain size or increasing voxel pitch",
                ))
        
        # Check topology complexity
        if "topology" in spec_data:
            topology = spec_data["topology"]
            terminals = topology.get("target_terminals")
            if terminals and terminals > 1000:
                risk_id += 1
                risks.append(RiskFlag(
                    id=f"risk_{risk_id:03d}",
                    severity="high",
                    category="performance",
                    description=f"High terminal count ({terminals}) may cause memory issues",
                    mitigation="Consider reducing target terminals or using progressive generation",
                ))
        
        # Check radius bounds
        if "geometry" in spec_data:
            geometry = spec_data["geometry"]
            radius_bounds = geometry.get("radius_bounds_m", {})
            min_radius = radius_bounds.get("min", 0.0001)
            if min_radius < 0.0001:
                risk_id += 1
                risks.append(RiskFlag(
                    id=f"risk_{risk_id:03d}",
                    severity="high",
                    category="manufacturing",
                    description=f"Minimum radius ({min_radius*1000:.3f}mm) may be below printable threshold",
                    mitigation="Increase minimum radius to at least 0.1mm for most printers",
                ))
        
        return risks
    
    def _generate_summary(
        self,
        spec_data: Dict[str, Any],
        requirements: Dict[str, Any],
    ) -> str:
        """Generate a plain English summary of the specification."""
        parts = []
        
        # Domain summary
        if "domain" in spec_data:
            domain = spec_data["domain"]
            size = domain.get("size_m", [0.02, 0.06, 0.03])
            size_mm = [s * 1000 for s in size]
            parts.append(
                f"Domain: {domain.get('type', 'box')} "
                f"({size_mm[0]:.1f} x {size_mm[1]:.1f} x {size_mm[2]:.1f} mm)"
            )
        
        # Topology summary
        if "topology" in spec_data:
            topology = spec_data["topology"]
            kind = topology.get("kind", "tree")
            terminals = topology.get("target_terminals", "unspecified")
            parts.append(f"Topology: {kind} with {terminals} target terminals")
        
        # Ports summary
        if "ports" in spec_data:
            ports = spec_data["ports"]
            n_inlets = len(ports.get("inlets", []))
            n_outlets = len(ports.get("outlets", []))
            parts.append(f"Ports: {n_inlets} inlet(s), {n_outlets} outlet(s)")
        
        return "; ".join(parts) if parts else "Specification created"
    
    def _generate_changelog_entry(
        self,
        requirements: Dict[str, Any],
        parent_version: Optional[int],
    ) -> str:
        """Generate a changelog entry for this version."""
        if parent_version is None:
            return "Initial specification created"
        else:
            return f"Refinement of version {parent_version:03d}"
    
    def _format_summary_md(self, spec_version: SpecVersion) -> str:
        """Format the summary as markdown."""
        md = f"""# Specification Summary - Version {spec_version.version:03d}

## Overview

{spec_version.summary}

## Status

- **Status**: {spec_version.status.value}
- **Created**: {spec_version.created_at}
- **Created By**: {spec_version.created_by}

## Assumptions

"""
        if spec_version.assumptions:
            for assumption in spec_version.assumptions:
                md += f"- {assumption}\n"
        else:
            md += "- No assumptions recorded\n"
        
        md += "\n## Risk Flags\n\n"
        if spec_version.risk_flags:
            for risk in spec_version.risk_flags:
                md += f"### {risk.id} ({risk.severity.upper()})\n\n"
                md += f"**Category**: {risk.category}\n\n"
                md += f"**Description**: {risk.description}\n\n"
                md += f"**Mitigation**: {risk.mitigation}\n\n"
        else:
            md += "No risks identified.\n"
        
        return md
    
    def _format_session_md(self, session: CSASession) -> str:
        """Format the session as markdown."""
        md = f"""# CSA Session - Version {session.spec_version:03d}

## Session Information

- **Session ID**: {session.session_id}
- **Started**: {session.started_at}
- **Completed**: {session.completed_at or "In progress"}

## Conversation Log

"""
        if session.conversation_log:
            for msg in session.conversation_log:
                md += f"### {msg['role'].title()} ({msg['timestamp']})\n\n"
                md += f"{msg['content']}\n\n"
        else:
            md += "No conversation recorded.\n"
        
        md += "\n## Decisions\n\n"
        if session.decisions:
            for i, decision in enumerate(session.decisions, 1):
                md += f"### Decision {i}\n\n"
                md += f"**Decision**: {decision['decision']}\n\n"
                md += f"**Rationale**: {decision['rationale']}\n\n"
                md += f"**Field Affected**: {decision['field_affected']}\n\n"
        else:
            md += "No decisions recorded.\n"
        
        md += "\n## Assumptions\n\n"
        if session.assumptions:
            for assumption in session.assumptions:
                md += f"- {assumption}\n"
        else:
            md += "No assumptions recorded.\n"
        
        return md
    
    def load_spec_version(self, version: int) -> Optional[SpecVersion]:
        """
        Load a spec version from disk.
        
        Parameters
        ----------
        version : int
            Version number to load
            
        Returns
        -------
        SpecVersion or None
            The loaded spec version, or None if not found
        """
        spec_path = self.folder_manager.get_spec_path(version)
        if not os.path.exists(spec_path):
            return None
        
        with open(spec_path, 'r') as f:
            data = json.load(f)
        
        return SpecVersion.from_dict(data)
    
    def approve_spec(self, version: int) -> bool:
        """
        Mark a spec version as approved.
        
        Parameters
        ----------
        version : int
            Version to approve
            
        Returns
        -------
        bool
            True if approved, False if not found
        """
        spec = self.load_spec_version(version)
        if spec is None:
            return False
        
        spec.status = VersionStatus.APPROVED
        
        spec_path = self.folder_manager.get_spec_path(version)
        with open(spec_path, 'w') as f:
            json.dump(spec.to_dict(), f, indent=2)
        
        self.folder_manager.log_event(f"Spec version {version} approved")
        
        return True
    
    def reject_spec(self, version: int, reason: str = "") -> bool:
        """
        Mark a spec version as rejected.
        
        Parameters
        ----------
        version : int
            Version to reject
        reason : str
            Reason for rejection
            
        Returns
        -------
        bool
            True if rejected, False if not found
        """
        spec = self.load_spec_version(version)
        if spec is None:
            return False
        
        spec.status = VersionStatus.REJECTED
        
        spec_path = self.folder_manager.get_spec_path(version)
        with open(spec_path, 'w') as f:
            json.dump(spec.to_dict(), f, indent=2)
        
        self.folder_manager.log_event(f"Spec version {version} rejected: {reason}")
        
        return True
