"""
CLI IO Adapter

Command-line interface implementation of the IO adapter.
"""

from typing import Any, Dict, List, Optional

from .base_io import BaseIOAdapter, IOMessageKind, TraceEvent


class CLIIOAdapter(BaseIOAdapter):
    """
    CLI implementation of the IO adapter.
    
    Uses standard input/output for all interactions.
    Approvals become Y/N prompts.
    """
    
    def __init__(
        self,
        verbose: bool = True,
        show_trace: bool = False,
    ):
        self.verbose = verbose
        self.show_trace = show_trace
    
    def say(
        self,
        message: str,
        kind: IOMessageKind = IOMessageKind.ASSISTANT,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Display a message to the user."""
        if kind == IOMessageKind.TRACE and not self.show_trace:
            return
        
        prefix = ""
        if kind == IOMessageKind.SYSTEM:
            prefix = "[SYSTEM] "
        elif kind == IOMessageKind.ERROR:
            prefix = "[ERROR] "
        elif kind == IOMessageKind.WARNING:
            prefix = "[WARNING] "
        elif kind == IOMessageKind.SUCCESS:
            prefix = "[SUCCESS] "
        elif kind == IOMessageKind.TRACE:
            prefix = "[TRACE] "
        
        print(f"{prefix}{message}")
    
    def ask_confirm(
        self,
        prompt: str,
        details: Optional[Dict[str, Any]] = None,
        modal: bool = True,
        runtime_estimate: Optional[str] = None,
        expected_outputs: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        risk_flags: Optional[List[str]] = None,
    ) -> bool:
        """Ask for yes/no confirmation."""
        print("\n" + "=" * 60)
        print(prompt)
        print("=" * 60)
        
        if runtime_estimate:
            print(f"Estimated runtime: {runtime_estimate}")
        
        if expected_outputs:
            print("\nExpected outputs:")
            for output in expected_outputs:
                print(f"  - {output}")
        
        if assumptions:
            print("\nCurrent assumptions:")
            for assumption in assumptions:
                print(f"  - {assumption}")
        
        if risk_flags:
            print("\nRisk flags:")
            for flag in risk_flags:
                print(f"  ! {flag}")
        
        if details:
            print("\nDetails:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        print()
        
        while True:
            response = input("Proceed? [y/n]: ").strip().lower()
            if response in ("y", "yes"):
                return True
            elif response in ("n", "no"):
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    def ask_text(
        self,
        prompt: str,
        suggestions: Optional[List[str]] = None,
        default: Optional[str] = None,
    ) -> str:
        """Ask for text input."""
        print()
        print(prompt)
        
        if suggestions:
            print("\nSuggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        if default:
            print(f"\n(Default: {default})")
        
        response = input("> ").strip()
        
        if not response and default:
            return default
        
        if suggestions and response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(suggestions):
                return suggestions[idx]
        
        return response
    
    def emit_trace(self, event: TraceEvent) -> None:
        """Emit a trace event."""
        if self.show_trace:
            print(f"[TRACE:{event.event_type}] {event.message}")
    
    def show_living_spec(self, spec_summary: Dict[str, Any]) -> None:
        """Show the current living spec summary."""
        print("\n" + "=" * 60)
        print("LIVING SPEC SUMMARY")
        print("=" * 60)
        
        confirmed = spec_summary.get("confirmed_facts", {})
        if confirmed:
            print("\nConfirmed facts:")
            for field, value in confirmed.items():
                print(f"  {field}: {value}")
        
        inferred = spec_summary.get("inferred_facts", {})
        if inferred:
            print("\nInferred/default facts (not yet confirmed):")
            for field, data in inferred.items():
                value = data.get("value") if isinstance(data, dict) else data
                reason = data.get("reason", "") if isinstance(data, dict) else ""
                print(f"  {field}: {value}")
                if reason:
                    print(f"    (reason: {reason})")
        
        questions = spec_summary.get("open_questions", [])
        if questions:
            print("\nOpen questions:")
            for q in questions:
                print(f"  - {q.get('question', q.get('field', 'Unknown'))}")
                if q.get("why"):
                    print(f"    Why: {q['why']}")
        
        changes = spec_summary.get("recent_changes", [])
        if changes:
            print("\nRecent changes:")
            for change in changes:
                print(f"  - {change.get('description', 'Unknown change')}")
        
        print(f"\nSpec hash: {spec_summary.get('spec_hash', 'N/A')}")
        print(f"Generation approved: {spec_summary.get('generation_approved', False)}")
        print(f"Postprocess approved: {spec_summary.get('postprocess_approved', False)}")
        print("=" * 60 + "\n")
    
    def show_plans(self, plans: List[Dict[str, Any]], recommended_id: Optional[str] = None) -> None:
        """Show proposed plans for user selection."""
        print("\n" + "=" * 60)
        print("PROPOSED PLANS")
        print("=" * 60)
        
        for plan in plans:
            plan_id = plan.get("plan_id", "?")
            name = plan.get("name", "Unnamed Plan")
            is_recommended = plan_id == recommended_id
            
            marker = " [RECOMMENDED]" if is_recommended else ""
            print(f"\n--- Plan {plan_id}: {name}{marker} ---")
            
            if plan.get("interpretation"):
                print(f"Interpretation: {plan['interpretation']}")
            
            if plan.get("geometry_strategy"):
                print(f"Strategy: {plan['geometry_strategy']}")
            
            if plan.get("parameter_draft"):
                print("Parameters:")
                for k, v in plan["parameter_draft"].items():
                    print(f"  {k}: {v}")
            
            if plan.get("risks"):
                print("Risks:")
                for risk in plan["risks"]:
                    print(f"  ! {risk}")
            
            if plan.get("cost_estimate"):
                print(f"Cost estimate: {plan['cost_estimate']}")
            
            if plan.get("what_needed_from_user"):
                print("What I need from you:")
                for item in plan["what_needed_from_user"]:
                    print(f"  ? {item}")
        
        print("=" * 60 + "\n")
    
    def prompt_plan_selection(self, plans: List[Dict[str, Any]]) -> Optional[str]:
        """Prompt user to select a plan."""
        print("\nSelect a plan (enter plan ID, or press Enter to use recommended):")
        
        plan_ids = [p.get("plan_id", "") for p in plans]
        for plan_id in plan_ids:
            print(f"  - {plan_id}")
        
        response = input("> ").strip()
        
        if not response:
            return None
        
        if response in plan_ids:
            return response
        
        print(f"Invalid plan ID. Please choose from: {', '.join(plan_ids)}")
        return self.prompt_plan_selection(plans)
    
    def show_safe_fix(
        self,
        field: str,
        before: Any,
        after: Any,
        reason: str,
    ) -> None:
        """Show a safe fix that was applied."""
        print("\n" + "-" * 40)
        print("SAFE FIX APPLIED")
        print("-" * 40)
        print(f"Field: {field}")
        print(f"Before: {before}")
        print(f"After: {after}")
        print(f"Reason: {reason}")
        print("-" * 40 + "\n")
    
    def show_generation_ready(
        self,
        runtime_estimate: str,
        expected_outputs: List[str],
        assumptions: List[str],
        risk_flags: List[str],
    ) -> None:
        """Show 'Ready to generate' card."""
        print("\n" + "=" * 60)
        print("READY TO GENERATE")
        print("=" * 60)
        print(f"Estimated runtime: {runtime_estimate}")
        
        print("\nExpected outputs:")
        for output in expected_outputs:
            print(f"  - {output}")
        
        print("\nAssumptions:")
        for assumption in assumptions:
            print(f"  - {assumption}")
        
        if risk_flags:
            print("\nRisk flags:")
            for flag in risk_flags:
                print(f"  ! {flag}")
        
        print("=" * 60 + "\n")
    
    def show_postprocess_ready(
        self,
        voxel_pitch: float,
        embedding_settings: Dict[str, Any],
        repair_steps: List[str],
        runtime_estimate: str,
        expected_outputs: List[str],
    ) -> None:
        """Show 'Ready to postprocess' card."""
        print("\n" + "=" * 60)
        print("READY TO POSTPROCESS")
        print("=" * 60)
        print(f"Voxel pitch: {voxel_pitch}")
        print(f"Estimated runtime: {runtime_estimate}")
        
        print("\nEmbedding settings:")
        for k, v in embedding_settings.items():
            print(f"  {k}: {v}")
        
        print("\nRepair steps:")
        for step in repair_steps:
            print(f"  - {step}")
        
        print("\nExpected outputs:")
        for output in expected_outputs:
            print(f"  - {output}")
        
        print("=" * 60 + "\n")
    
    def prompt_stl_viewer(self, stl_path: str) -> None:
        """Prompt user to view STL file."""
        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        print(f"STL file generated: {stl_path}")
        print("\nYou can view this file in an STL viewer.")
        print("=" * 60 + "\n")
