"""
LLM-First DesignSpec Agent

This module provides the LLM-first agent for DesignSpec workflows.
It replaces the rule-based agent with an LLM-driven approach that:
1. Passes user messages directly to the LLM with context
2. Parses structured directives from LLM responses
3. Falls back to rule-based logic when LLM is unavailable

The agent integrates with the existing DesignSpecWorkflow through
the AgentResponse interface.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .directive import (
    DesignSpecDirective,
    DirectiveParseError,
    Question as DirectiveQuestion,
    from_json,
    create_error_directive,
    create_fallback_directive,
)
from .context_builder import ContextBuilder, ContextPack
from .prompt_builder import PromptBuilder, get_system_prompt

if TYPE_CHECKING:
    from ..designspec_session import DesignSpecSession, PatchProposal, ValidationReport
    from ..llm_client import LLMClient

logger = logging.getLogger(__name__)


# Maximum retries for LLM parsing failures
MAX_PARSE_RETRIES = 2


@dataclass
class AgentTurnLog:
    """Log entry for a single agent turn."""
    timestamp: float
    user_message: str
    context_pack_hash: str
    directive_json: Optional[Dict[str, Any]] = None
    patch_applied: bool = False
    run_executed: bool = False
    run_result: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    context_request_fulfilled: bool = False
    second_directive_json: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "user_message": self.user_message,
            "context_pack_hash": self.context_pack_hash,
            "directive_json": self.directive_json,
            "patch_applied": self.patch_applied,
            "run_executed": self.run_executed,
            "run_result": self.run_result,
            "errors": self.errors,
            "context_request_fulfilled": self.context_request_fulfilled,
            "second_directive_json": self.second_directive_json,
        }


class DesignSpecLLMAgent:
    """
    LLM-first agent for DesignSpec workflows.
    
    This agent:
    1. Always attempts to use the LLM first
    2. Builds context packs from the session
    3. Parses LLM responses into structured directives
    4. Falls back to rule-based logic when LLM is unavailable
    
    Attributes
    ----------
    session : DesignSpecSession
        The active design spec session
    llm_client : LLMClient, optional
        The LLM client for making API calls
    """
    
    def __init__(
        self,
        session: "DesignSpecSession",
        llm_client: Optional["LLMClient"] = None,
        legacy_agent: Optional[Any] = None,
    ):
        """
        Initialize the LLM agent.
        
        Parameters
        ----------
        session : DesignSpecSession
            The active design spec session
        llm_client : LLMClient, optional
            LLM client for API calls. If None, falls back to legacy agent.
        legacy_agent : DesignSpecAgent, optional
            Legacy rule-based agent for fallback
        """
        self.session = session
        self.llm_client = llm_client
        self.legacy_agent = legacy_agent
        
        self.context_builder = ContextBuilder(session)
        self.prompt_builder = PromptBuilder()
        
        self._conversation_history: List[Dict[str, str]] = []
        self._turn_logs: List[AgentTurnLog] = []
        self._last_directive: Optional[DesignSpecDirective] = None
        self._pending_context_request: Optional[Dict[str, Any]] = None
    
    @property
    def has_llm(self) -> bool:
        """Check if LLM client is available."""
        return self.llm_client is not None
    
    def process_message(
        self,
        user_message: str,
        validation_report: Optional["ValidationReport"] = None,
        compile_report: Optional[Dict[str, Any]] = None,
        use_full_context: bool = False,
    ) -> "AgentResponse":
        """
        Process a user message and generate a response.
        
        This is the main entry point for the agent. It:
        1. Builds context from the session
        2. Calls the LLM with the user message and context
        3. Parses the response into a directive
        4. Converts the directive to an AgentResponse
        
        Parameters
        ----------
        user_message : str
            The user's message
        validation_report : ValidationReport, optional
            Current validation report
        compile_report : dict, optional
            Current compile report
        use_full_context : bool
            Whether to use full context (default: compact)
            
        Returns
        -------
        AgentResponse
            The agent's response
        """
        # Import here to avoid circular imports
        from ..designspec_agent import (
            AgentResponse,
            AgentResponseType,
            Question,
            RunRequest,
        )
        from ..designspec_session import PatchProposal
        
        # Add user message to history
        self._conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        
        # Build context pack
        context_pack = self.context_builder.build(full=use_full_context)
        context_hash = self._compute_context_hash(context_pack)
        
        # Create turn log
        turn_log = AgentTurnLog(
            timestamp=time.time(),
            user_message=user_message,
            context_pack_hash=context_hash,
        )
        
        # Try LLM first
        if self.has_llm:
            try:
                directive = self._call_llm(user_message, context_pack)
                turn_log.directive_json = directive.to_dict()
                
                # Check if context request fulfillment is needed (automatic second LLM call)
                if directive.context_requests and directive.context_requests.has_requests():
                    logger.info(f"Context request detected: {directive.context_requests.why}")
                    
                    # Build expanded context pack based on requests
                    expanded_context = self._build_expanded_context(
                        directive.context_requests,
                        validation_report,
                        compile_report,
                    )
                    
                    # Call LLM second time with expanded context
                    second_directive = self._call_llm(user_message, expanded_context)
                    turn_log.context_request_fulfilled = True
                    turn_log.second_directive_json = second_directive.to_dict()
                    
                    # Check if second call still requests more context
                    if second_directive.context_requests and second_directive.context_requests.has_requests():
                        logger.warning("Second LLM call still requests more context - returning as-is")
                        # Don't loop - return the second directive asking user to provide missing files
                    
                    # Use the second directive
                    directive = second_directive
                
                self._last_directive = directive
                
                # Convert directive to AgentResponse
                response = self._directive_to_response(directive)
                
                # Add assistant message to history
                self._conversation_history.append({
                    "role": "assistant",
                    "content": directive.assistant_message,
                })
                
                self._turn_logs.append(turn_log)
                self._save_turn_log(turn_log)
                
                return response
                
            except DirectiveParseError as e:
                logger.warning(f"LLM response parsing failed: {e}")
                turn_log.errors.append(f"Parse error: {e}")
                fallback_notice = self._build_fallback_notice("DirectiveParseError", str(e))
                # Fall through to fallback
                
            except Exception as e:
                logger.exception(f"LLM call failed: {e}")
                turn_log.errors.append(f"LLM error: {e}")
                fallback_notice = self._build_fallback_notice(type(e).__name__, str(e))
                # Fall through to fallback
        else:
            # No LLM client available
            fallback_notice = self._build_fallback_notice("NoLLMClient", "LLM client not initialized")
        
        # Fallback to legacy agent or error response
        if self.legacy_agent:
            logger.info("Falling back to legacy agent")
            
            # Sync conversation history to legacy agent so it can properly
            # detect question answers and maintain context
            self.legacy_agent._conversation_history = self._conversation_history.copy()
            
            spec = self.session.get_spec()
            response = self.legacy_agent.process_message(
                user_message=user_message,
                spec=spec,
                validation_report=validation_report,
                compile_report=compile_report,
            )
            
            # Prepend fallback notice to response message and patch explanation
            response = self._prepend_fallback_notice(response, fallback_notice)
            
            self._turn_logs.append(turn_log)
            self._save_turn_log(turn_log)  # Save turn log for fallback turns too
            return response
        
        # No LLM and no legacy agent - return error
        turn_log.errors.append("No LLM or legacy agent available")
        self._turn_logs.append(turn_log)
        
        return AgentResponse(
            response_type=AgentResponseType.ERROR,
            message="Unable to process message: LLM is unavailable and no fallback agent is configured.",
        )
    
    def _call_llm(
        self,
        user_message: str,
        context_pack: ContextPack,
    ) -> DesignSpecDirective:
        """
        Call the LLM and parse the response.
        
        Parameters
        ----------
        user_message : str
            The user's message
        context_pack : ContextPack
            The context pack
            
        Returns
        -------
        DesignSpecDirective
            The parsed directive
            
        Raises
        ------
        DirectiveParseError
            If parsing fails after retries
        """
        # Build prompts
        system_prompt = self.prompt_builder.system_prompt
        user_prompt = self.prompt_builder.build_user_prompt(
            user_message=user_message,
            context_pack=context_pack,
            conversation_history=self._conversation_history[:-1],  # Exclude current message
        )
        
        # Call LLM
        response = self._make_llm_call(system_prompt, user_prompt)
        
        # Try to parse
        try:
            return from_json(response)
        except DirectiveParseError as e:
            # Retry with error feedback
            for retry in range(MAX_PARSE_RETRIES):
                logger.info(f"Retrying LLM call (attempt {retry + 2})")
                
                retry_prompt = self.prompt_builder.build_retry_prompt(
                    original_response=response,
                    parse_errors=e.errors,
                )
                
                response = self._make_llm_call(system_prompt, retry_prompt)
                
                try:
                    return from_json(response)
                except DirectiveParseError as retry_error:
                    e = retry_error
                    continue
            
            # All retries failed
            raise e
    
    def _make_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """
        Make an LLM API call.
        
        Parameters
        ----------
        system_prompt : str
            The system prompt
        user_prompt : str
            The user prompt
            
        Returns
        -------
        str
            The LLM response text
        """
        # Use the LLM client's chat method
        # The client handles conversation history internally
        response = self.llm_client.chat(
            message=user_prompt,
            system_prompt=system_prompt,
            continue_conversation=False,  # We manage history ourselves
        )
        
        return response.content
    
    def _build_expanded_context(
        self,
        context_request: "ContextRequest",
        validation_report: Optional["ValidationReport"] = None,
        compile_report: Optional[Dict[str, Any]] = None,
    ) -> ContextPack:
        """
        Build an expanded context pack based on context requests.
        
        This method is called when the LLM requests additional context
        during the automatic second LLM call.
        
        Parameters
        ----------
        context_request : ContextRequest
            The context request from the LLM
        validation_report : ValidationReport, optional
            Current validation report
        compile_report : dict, optional
            Current compile report
            
        Returns
        -------
        ContextPack
            The expanded context pack
        """
        # Start with full context if need_full_spec is requested
        if context_request.need_full_spec:
            context_pack = self.context_builder.build_full()
        else:
            # Build compact context as base
            context_pack = self.context_builder.build_compact()
        
        # Add validation report if requested
        if context_request.need_validity_report and validation_report:
            context_pack = self.context_builder.add_validation_details(
                context_pack, validation_report
            )
        
        # Add last run report if requested
        if context_request.need_last_run_report:
            context_pack = self.context_builder.add_run_details(context_pack)
        
        # Add network artifact if requested
        if context_request.need_network_artifact:
            context_pack = self.context_builder.add_network_artifact(context_pack)
        
        # Add specific files if requested
        if context_request.need_specific_files:
            context_pack = self.context_builder.add_specific_files(
                context_pack, context_request.need_specific_files
            )
        
        # Add more conversation history if requested
        if context_request.need_more_history:
            context_pack = self.context_builder.add_extended_history(
                context_pack, self._conversation_history
            )
        
        return context_pack
    
    def _directive_to_response(
        self,
        directive: DesignSpecDirective,
    ) -> "AgentResponse":
        """
        Convert a DesignSpecDirective to an AgentResponse.
        
        Parameters
        ----------
        directive : DesignSpecDirective
            The directive from the LLM
            
        Returns
        -------
        AgentResponse
            The converted response
        """
        from ..designspec_agent import (
            AgentResponse,
            AgentResponseType,
            Question,
            RunRequest,
        )
        from ..designspec_session import PatchProposal
        
        # Determine response type based on directive content
        if directive.has_patches():
            # Patch proposal
            return AgentResponse(
                response_type=AgentResponseType.PATCH_PROPOSAL,
                message=directive.assistant_message,
                patch_proposal=PatchProposal(
                    explanation=directive.assistant_message,
                    patches=directive.proposed_patches,
                    confidence=directive.confidence,
                    requires_confirmation=directive.requires_approval,
                ),
            )
        
        elif directive.has_run_request():
            # Run request
            run_req = directive.run_request
            return AgentResponse(
                response_type=AgentResponseType.RUN_REQUEST,
                message=directive.assistant_message,
                run_request=RunRequest(
                    run_until=run_req.run_until,
                    full_run=(run_req.run_until == "full" or run_req.run_until is None),
                    reason=run_req.reason,
                ),
            )
        
        elif directive.has_questions():
            # Questions
            questions = [
                Question(
                    field_path=q.id,
                    question_text=q.question,
                    reason=q.why_needed,
                    default=q.default,
                )
                for q in directive.questions
            ]
            return AgentResponse(
                response_type=AgentResponseType.QUESTION,
                message=directive.assistant_message,
                questions=questions,
            )
        
        elif directive.needs_more_context():
            # Context request - store for next turn and ask user to continue
            self._pending_context_request = directive.context_requests.to_dict()
            return AgentResponse(
                response_type=AgentResponseType.MESSAGE,
                message=directive.assistant_message + "\n\n(Gathering additional context...)",
            )
        
        elif directive.stop:
            # Workflow complete
            return AgentResponse(
                response_type=AgentResponseType.MESSAGE,
                message=directive.assistant_message,
            )
        
        else:
            # General message
            return AgentResponse(
                response_type=AgentResponseType.MESSAGE,
                message=directive.assistant_message,
            )
    
    def _compute_context_hash(self, context_pack: ContextPack) -> str:
        """Compute a hash of the context pack for logging."""
        import hashlib
        context_json = json.dumps(context_pack.to_dict(), sort_keys=True)
        return hashlib.sha256(context_json.encode()).hexdigest()[:16]
    
    def _save_turn_log(self, turn_log: AgentTurnLog) -> None:
        """Save a turn log to the project directory."""
        logs_dir = self.session.project_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        log_file = logs_dir / "agent_turns.jsonl"
        
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(turn_log.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save turn log: {e}")
    
    def _build_fallback_notice(self, error_type: str, error_message: str) -> str:
        """
        Build a user-friendly fallback notice for LLM failures.
        
        Parameters
        ----------
        error_type : str
            The type/class name of the error
        error_message : str
            The error message (will be sanitized and truncated)
            
        Returns
        -------
        str
            A formatted fallback notice string
        """
        # Sanitize error message: remove potential secrets (API keys, tokens, etc.)
        # Look for common patterns that might contain secrets
        import re
        sanitized = error_message
        
        # Remove potential API keys (long alphanumeric strings)
        sanitized = re.sub(r'[A-Za-z0-9_-]{32,}', '[REDACTED]', sanitized)
        # Remove potential bearer tokens
        sanitized = re.sub(r'Bearer\s+\S+', 'Bearer [REDACTED]', sanitized, flags=re.IGNORECASE)
        # Remove potential URLs with credentials
        sanitized = re.sub(r'://[^@]+@', '://[REDACTED]@', sanitized)
        
        # Truncate to ~200 chars
        max_len = 200
        if len(sanitized) > max_len:
            sanitized = sanitized[:max_len - 3] + "..."
        
        return f"⚠️ LLM failed ({error_type}: {sanitized}). Falling back to legacy parser.\n\n"
    
    def _prepend_fallback_notice(
        self,
        response: "AgentResponse",
        fallback_notice: str,
    ) -> "AgentResponse":
        """
        Prepend a fallback notice to an AgentResponse.
        
        Modifies the response message, patch proposal explanation (if present),
        and question text (if present) to include the fallback notice.
        
        Parameters
        ----------
        response : AgentResponse
            The response from the legacy agent
        fallback_notice : str
            The fallback notice to prepend
            
        Returns
        -------
        AgentResponse
            The modified response with fallback notice prepended
        """
        from ..designspec_agent import AgentResponse, AgentResponseType
        
        # Prepend to main message
        new_message = fallback_notice + response.message
        
        # Prepend to patch proposal explanation if present
        new_patch_proposal = response.patch_proposal
        if new_patch_proposal and hasattr(new_patch_proposal, 'explanation'):
            new_patch_proposal.explanation = fallback_notice + (new_patch_proposal.explanation or "")
        
        # Prepend to first question if this is a QUESTION response
        new_questions = response.questions
        if response.response_type == AgentResponseType.QUESTION and new_questions:
            # Modify the first question's text to include the notice
            first_q = new_questions[0]
            if hasattr(first_q, 'question_text'):
                first_q.question_text = fallback_notice + first_q.question_text
        
        # Return a new response with the modified fields
        return AgentResponse(
            response_type=response.response_type,
            message=new_message,
            questions=new_questions,
            patch_proposal=new_patch_proposal,
            run_request=response.run_request,
        )
    
    def record_patch_applied(self, patch_id: str, success: bool) -> None:
        """
        Record that a patch was applied.
        
        Parameters
        ----------
        patch_id : str
            The patch ID
        success : bool
            Whether the patch was applied successfully
        """
        if self._turn_logs:
            self._turn_logs[-1].patch_applied = success
    
    def record_run_executed(self, run_result: Dict[str, Any]) -> None:
        """
        Record that a run was executed.
        
        Parameters
        ----------
        run_result : dict
            The run result
        """
        if self._turn_logs:
            self._turn_logs[-1].run_executed = True
            self._turn_logs[-1].run_result = run_result
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self._conversation_history.copy()
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()
    
    def add_assistant_message(self, message: str) -> None:
        """Add an assistant message to the history."""
        self._conversation_history.append({
            "role": "assistant",
            "content": message,
        })
    
    def get_last_directive(self) -> Optional[DesignSpecDirective]:
        """Get the last directive from the LLM."""
        return self._last_directive
    
    def get_turn_logs(self) -> List[AgentTurnLog]:
        """Get all turn logs."""
        return self._turn_logs.copy()
    
    def has_pending_context_request(self) -> bool:
        """Check if there's a pending context request."""
        return self._pending_context_request is not None
    
    def fulfill_context_request(self) -> Optional[ContextPack]:
        """
        Fulfill a pending context request.
        
        Returns
        -------
        ContextPack or None
            Full context pack if there was a pending request
        """
        if not self._pending_context_request:
            return None
        
        # Build full context
        context_pack = self.context_builder.build_full()
        self._pending_context_request = None
        
        return context_pack


def create_llm_agent(
    session: "DesignSpecSession",
    llm_client: Optional["LLMClient"] = None,
    use_legacy_fallback: bool = True,
) -> DesignSpecLLMAgent:
    """
    Create an LLM agent for a DesignSpec session.
    
    Parameters
    ----------
    session : DesignSpecSession
        The active session
    llm_client : LLMClient, optional
        LLM client for API calls
    use_legacy_fallback : bool
        Whether to use legacy agent as fallback
        
    Returns
    -------
    DesignSpecLLMAgent
        The configured agent
    """
    legacy_agent = None
    
    if use_legacy_fallback:
        from ..designspec_agent import DesignSpecAgent
        legacy_agent = DesignSpecAgent(llm_client=None)  # Rule-based only
    
    return DesignSpecLLMAgent(
        session=session,
        llm_client=llm_client,
        legacy_agent=legacy_agent,
    )
