"""CMO Multi-Agent Workflow Orchestration using LangGraph.

This module implements the orchestration layer using LangGraph for state management
and conditional routing, with CrewAI agents for specialized reasoning.

LangGraph handles:
- State management across agent interactions
- Conditional routing based on critique severity
- Iteration limits to prevent infinite loops
- Decision history tracking

Flow:
Problem Input → Analyst → Strategy → Execution → Critic
                                                    ↓
                                      ┌─────────────┴─────────────┐
                                      ↓                           ↓
                            If severity > 0.7          If severity < 0.3
                            → Reject & Replan          → Accept Decision
                                      ↓
                            If severity 0.3-0.7
                            → Refine Strategy
                                      ↓
                            (Max 3 iterations)
"""

import json
import logging
import operator
from typing import Any, Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from crewai import Crew, Process

from models import (
    AnalystOutput,
    StrategyOutput,
    ExecutionOutput,
    CriticOutput,
    DecisionStatus,
)
from agents import (
    create_analyst_agent,
    create_strategy_agent,
    create_execution_agent,
    create_critic_agent,
    create_analyst_task,
    create_strategy_task,
    create_execution_task,
    create_critic_task,
)
from config import (
    MAX_ITERATIONS,
    SEVERITY_ACCEPT_THRESHOLD,
    SEVERITY_REFINE_THRESHOLD,
    get_llm_for_agent,
    MEMORY_ENABLED,
    MEMORY_MIN_RELEVANCE,
    MEMORY_TOP_K,
)

logger = logging.getLogger(__name__)

# Import memory layer (optional, graceful if missing)
try:
    from memory import MemoryManager, format_memory_for_prompt
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    logger.warning("[Memory] Memory module not available")


# =============================================================================
# Helper Functions
# =============================================================================

def _normalize_success_metrics(metrics: list) -> list[str]:
    """Normalize success_metrics to a list of strings.
    
    LLMs sometimes return metrics as dicts like {"awareness": "10k views"}
    instead of strings. This handles both formats.
    """
    if not metrics:
        return []
    
    result = []
    for m in metrics:
        if isinstance(m, str):
            result.append(m)
        elif isinstance(m, dict):
            # Convert dict to string like "awareness: 10k views"
            for k, v in m.items():
                result.append(f"{k}: {v}")
        else:
            result.append(str(m))
    return result


# =============================================================================
# LangGraph State Definition
# =============================================================================

class CMOState(TypedDict):
    """State maintained across the LangGraph workflow."""
    # Input
    problem: str
    context: dict[str, Any]
    
    # Agent outputs (stored as dicts for serialization)
    analyst_output: dict[str, Any] | None
    strategy_output: dict[str, Any] | None
    execution_output: dict[str, Any] | None
    critic_output: dict[str, Any] | None
    
    # Workflow control
    iteration: Annotated[int, operator.add]  # Accumulates across iterations
    max_iterations: int
    status: str  # "accept", "refine", "reject"
    
    # History tracking
    decision_history: Annotated[list[dict], operator.add]
    
    # Final output
    final_decision: dict[str, Any] | None
    final_confidence: float
    
    # Error handling
    error: str | None


def create_initial_state(problem: str, context: dict[str, Any]) -> CMOState:
    """Create the initial state for the workflow."""
    return CMOState(
        problem=problem,
        context=context,
        analyst_output=None,
        strategy_output=None,
        execution_output=None,
        critic_output=None,
        iteration=1,  # Start at 1 (first iteration), incremented by router when looping
        max_iterations=MAX_ITERATIONS,
        status="pending",
        decision_history=[],
        final_decision=None,
        final_confidence=0.0,
        error=None,
    )


# =============================================================================
# JSON Parsing Helper
# =============================================================================

def parse_json_output(text: str) -> dict[str, Any]:
    """Extract JSON from agent output text with robust error handling."""
    text = text.strip()
    
    # Try to find JSON block in markdown
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
    
    # Find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Attempted to parse: {text[start:end][:500]}")
            # Return minimal valid structure instead of crashing
            return {
                "error": "Failed to parse JSON output",
                "raw_text": text[:500],
                "confidence": 0.3
            }
    
    # No JSON found - return error structure instead of raising
    logger.error(f"No JSON found in response: {text[:200]}")
    return {
        "error": "No JSON in response",
        "raw_text": text[:500],
        "confidence": 0.3
    }


# =============================================================================
# LangGraph Node Functions (invoke CrewAI agents)
# =============================================================================

def analyst_node(state: CMOState) -> dict:
    """Run the Analyst agent to diagnose the problem.
    
    Uses: Gemini 2.5 Flash (primary) → gpt-oss 20B (fallback)
    Optionally retrieves relevant past campaigns from memory.
    """
    logger.info("[Analyst] Diagnosing problem...")
    logger.info("[Analyst] LLM: Gemini 2.5 Flash → gpt-oss 20B fallback")
    
    # Memory retrieval (optional, graceful degradation)
    memory_context = None
    if MEMORY_ENABLED and MEMORY_AVAILABLE:
        try:
            memory = MemoryManager()
            relevant_campaigns = memory.get_relevant_campaigns(
                problem=state["problem"],
                context=state["context"],
                min_relevance=MEMORY_MIN_RELEVANCE,
                top_k=MEMORY_TOP_K
            )
            if relevant_campaigns:
                memory_context = format_memory_for_prompt(relevant_campaigns)
                logger.info(f"[Memory] Found {len(relevant_campaigns)} relevant campaigns")
            else:
                logger.info("[Memory] No relevant campaigns found")
        except Exception as e:
            logger.warning(f"[Memory] Retrieval failed: {e}, proceeding without memory")
            memory_context = None
    elif MEMORY_ENABLED and not MEMORY_AVAILABLE:
        logger.warning("[Memory] Enabled but memory module not available")
    
    llm = get_llm_for_agent("analyst")
    
    agent = create_analyst_agent(llm)
    task = create_analyst_task(
        agent, 
        state["problem"], 
        state["context"],
        memory_context=memory_context
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    
    try:
        result = crew.kickoff()
        output = parse_json_output(str(result))
        logger.info(f"[Analyst] Diagnosis: {output.get('primary_diagnosis', 'Unknown')}")
        logger.info(f"[Analyst] Confidence: {output.get('confidence', 0)}")
        return {"analyst_output": output}
    except Exception as e:
        logger.error(f"[Analyst] Error: {e}")
        return {
            "analyst_output": {
                "root_cause_hypotheses": [],
                "primary_diagnosis": f"Analysis failed: {str(e)}",
                "confidence": 0.3,
                "supporting_signals": [],
            },
            "error": str(e),
        }


def strategy_node(state: CMOState) -> dict:
    """Run the Strategy agent to generate options.
    
    Uses: Gemini 2.5 Flash (primary) → gpt-oss 20B (fallback)
    """
    logger.info("[Strategy] Generating strategy options...")
    logger.info("[Strategy] LLM: Gemini 2.5 Flash → gpt-oss 20B fallback")
    
    if not state.get("analyst_output"):
        return {"error": "No analyst output available"}
    
    llm = get_llm_for_agent("strategy")
    
    agent = create_strategy_agent(llm)
    task = create_strategy_task(agent, state["analyst_output"], state["context"])
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    
    try:
        result = crew.kickoff()
        output = parse_json_output(str(result))
        logger.info(f"[Strategy] Generated {len(output.get('options', []))} options")
        logger.info(f"[Strategy] Recommended: {output.get('recommended_option', 'None')}")
        return {"strategy_output": output}
    except Exception as e:
        logger.error(f"[Strategy] Error: {e}")
        return {
            "strategy_output": {
                "options": [],
                "recommended_option": "Unable to generate",
                "trade_off_analysis": "",
                "confidence": 0.3,
            },
            "error": str(e),
        }


def execution_node(state: CMOState) -> dict:
    """Run the Execution agent to create action plan.
    
    Uses: Gemini 2.5 Flash (primary) → Llama 3.1 8B (fallback)
    """
    iteration = state.get("iteration", 0)
    logger.info(f"[Execution] Creating action plan (iteration {iteration + 1})...")
    logger.info("[Execution] LLM: Gemini 2.5 Flash → Llama 3.1 8B fallback")
    
    if not state.get("analyst_output") or not state.get("strategy_output"):
        return {"error": "Missing required inputs for execution"}
    
    llm = get_llm_for_agent("execution")
    
    # Get critic feedback if this is a refinement iteration
    critic_feedback = None
    if state.get("critic_output") and state.get("status") == "refine":
        critic_data = state["critic_output"]
        feedback_parts = []
        if critic_data.get("feedback"):
            feedback_parts.append(f"ISSUES: {critic_data['feedback']}")
        if critic_data.get("refinement_suggestions"):
            suggestions = "\n".join([f"  - {s}" for s in critic_data["refinement_suggestions"]])
            feedback_parts.append(f"REQUIRED REFINEMENTS:\n{suggestions}")
        if critic_data.get("issues"):
            issues = "\n".join([f"  - [{i.get('severity', 'medium')}] {i.get('issue', '')}" for i in critic_data["issues"]])
            feedback_parts.append(f"SPECIFIC ISSUES:\n{issues}")
        critic_feedback = "\n\n".join(feedback_parts)
    
    agent = create_execution_agent(llm)
    task = create_execution_task(
        agent,
        state["strategy_output"],
        state["analyst_output"],
        state["context"],
        critic_feedback,
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    
    try:
        result = crew.kickoff()
        output = parse_json_output(str(result))
        logger.info(f"[Execution] Strategy: {output.get('selected_strategy', 'Unknown')}")
        logger.info(f"[Execution] Actions: {len(output.get('action_items', []))}")
        return {"execution_output": output}  # Don't increment iteration here
    except Exception as e:
        logger.error(f"[Execution] Error: {e}")
        return {
            "execution_output": {
                "selected_strategy": "Unable to create plan",
                "justification": str(e),
                "action_items": [],
                "budget_allocation": "",
                "success_metrics": [],
                "checkpoints": [],
                "fallback_plan": None,
                "confidence": 0.3,
            },
            "error": str(e),
        }


def critic_node(state: CMOState) -> dict:
    """Run the Critic agent to review the plan.
    
    Uses: Gemini 2.5 Flash (primary) → Llama 3.1 8B (fallback)
    """
    iteration = state.get("iteration", 0)
    logger.info(f"[Critic] Reviewing plan (iteration {iteration})...")
    logger.info("[Critic] LLM: Gemini 2.5 Flash → Llama 3.1 8B fallback")
    
    if not all([state.get("analyst_output"), state.get("strategy_output"), state.get("execution_output")]):
        return {"error": "Missing required inputs for critique"}
    
    llm = get_llm_for_agent("critic")
    
    agent = create_critic_agent(llm)
    task = create_critic_task(
        agent,
        state["execution_output"],
        state["strategy_output"],
        state["analyst_output"],
        state["context"],
        iteration - 1,  # 0-indexed for display
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    
    try:
        result = crew.kickoff()
        output = parse_json_output(str(result))
        severity = output.get("severity_score", 0.5)
        
        logger.info(f"[Critic] Severity: {severity}")
        logger.info(f"[Critic] Status: {output.get('approval_status', 'unknown')}")
        
        # Track in decision history
        history_entry = {
            "iteration": iteration,
            "strategy": state["execution_output"].get("selected_strategy", "Unknown"),
            "severity": severity,
            "status": output.get("approval_status", "unknown"),
        }
        
        return {
            "critic_output": output,
            "decision_history": [history_entry],
        }
    except Exception as e:
        logger.error(f"[Critic] Error: {e}")
        # Default to approve on error to prevent infinite loops
        return {
            "critic_output": {
                "severity_score": 0.2,
                "issues": [],
                "feedback": f"Critique failed: {str(e)}",
                "refinement_suggestions": [],
                "approval_status": "approve",
                "reasoning": "Auto-approved due to critique error",
            },
            "decision_history": [{
                "iteration": iteration,
                "strategy": state["execution_output"].get("selected_strategy", "Unknown"),
                "severity": 0.2,
                "status": "approve (error fallback)",
            }],
        }


def router_node(state: CMOState) -> dict:
    """Route based on critique severity and iteration count.
    
    Iteration is incremented here (only when looping) to prevent
    premature max iteration hits from execution retries.
    """
    critic_output = state.get("critic_output", {})
    severity = critic_output.get("severity_score", 0.5)
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", MAX_ITERATIONS)
    
    logger.info(f"[Router] Iteration {iteration}, Severity: {severity}")
    
    # Check iteration limit
    if iteration >= max_iter:
        logger.warning(f"[Router] Max iterations ({max_iter}) reached, forcing accept")
        return {"status": "accept"}
    
    # Route based on severity - increment iteration only when looping
    if severity < SEVERITY_ACCEPT_THRESHOLD:
        logger.info("[Router] → ACCEPT (low severity)")
        return {"status": "accept"}
    elif severity < SEVERITY_REFINE_THRESHOLD:
        logger.info("[Router] → REFINE (medium severity)")
        return {"status": "refine", "iteration": 1}  # Increment when looping
    else:
        logger.info("[Router] → REJECT (high severity)")
        return {"status": "reject", "iteration": 1}  # Increment when looping


def finalize_node(state: CMOState) -> dict:
    """Create the final decision output."""
    logger.info("[Finalize] Creating final decision...")
    
    execution = state.get("execution_output", {})
    analyst = state.get("analyst_output", {})
    strategy = state.get("strategy_output", {})
    critic = state.get("critic_output", {})
    
    # Calculate confidence
    analyst_conf = analyst.get("confidence", 0) * 0.3
    strategy_conf = strategy.get("confidence", 0) * 0.4
    critique_factor = (1 - critic.get("severity_score", 0.5)) * 0.3
    final_confidence = round(analyst_conf + strategy_conf + critique_factor, 2)
    
    # Build reasoning chain
    reasoning_parts = []
    if analyst:
        reasoning_parts.append(f"- Root cause: {analyst.get('primary_diagnosis', 'Unknown')}")
        reasoning_parts.append(f"  Confidence: {analyst.get('confidence', 0)}")
    if strategy:
        reasoning_parts.append(f"- Strategy: {strategy.get('recommended_option', 'Unknown')}")
        reasoning_parts.append(f"  Trade-offs: {strategy.get('trade_off_analysis', '')[:100]}")
    if execution:
        reasoning_parts.append(f"- Execution: {execution.get('selected_strategy', 'Unknown')}")
        reasoning_parts.append(f"  Justification: {execution.get('justification', '')[:100]}")
    if critic:
        reasoning_parts.append(f"- Critic: {critic.get('approval_status', 'Unknown')}")
        reasoning_parts.append(f"  Severity: {critic.get('severity_score', 0)}")
    
    # Get rejected alternatives
    alternatives = []
    selected = strategy.get("recommended_option", "")
    for opt in strategy.get("options", []):
        if opt.get("name") != selected:
            alternatives.append(f"{opt.get('name', 'Unknown')}: {opt.get('description', '')[:80]}...")
    
    final_decision = {
        "decision": execution.get("selected_strategy", "No decision"),
        "strategy": execution.get("justification", ""),
        "budget": execution.get("budget_allocation", "N/A"),
        "timeline": execution.get("checkpoints", ["N/A"])[0] if execution.get("checkpoints") else "N/A",
        "action_items": [
            f"Step {a.get('step', '?')}: {a.get('action', '')} ({a.get('deadline', 'TBD')})"
            for a in execution.get("action_items", [])
        ],
        "success_metrics": _normalize_success_metrics(execution.get("success_metrics", [])),
        "confidence": final_confidence,
        "iterations": state.get("iteration", 0),
        "reasoning": "\n".join(reasoning_parts),
        "alternatives_rejected": alternatives,
        "decision_history": state.get("decision_history", []),
    }
    
    # Log metrics for observability
    logger.info(f"[Metrics] Final confidence: {final_confidence:.2f}")
    logger.info(f"[Metrics] Total iterations: {state.get('iteration', 0)}")
    logger.info(f"[Metrics] Refinements: {max(0, state.get('iteration', 1) - 1)}")
    logger.info(f"[Metrics] Action items: {len(execution.get('action_items', []))}")
    
    # Store campaign in memory for future reference
    if MEMORY_ENABLED and MEMORY_AVAILABLE:
        try:
            memory = MemoryManager()
            outcome = {
                "success": final_confidence >= 0.7,
                "result": f"Strategy: {execution.get('selected_strategy', 'Unknown')[:100]}",
                "confidence": final_confidence,
            }
            lessons = critic.get("feedback", "")[:200] if critic else ""
            
            memory.store_campaign(
                problem=state["problem"],
                context=state["context"],
                diagnosis=analyst.get("primary_diagnosis", "Unknown"),
                strategy=execution.get("selected_strategy", "Unknown"),
                outcome=outcome,
                lessons_learned=lessons,
            )
            logger.info("[Memory] Campaign stored for future reference")
        except Exception as e:
            logger.warning(f"[Memory] Storage failed: {e}")
    
    return {
        "final_decision": final_decision,
        "final_confidence": final_confidence,
    }


# =============================================================================
# Routing Logic (conditional edges)
# =============================================================================

def route_after_critic(state: CMOState) -> Literal["router", "finalize"]:
    """Determine next step after critic review."""
    # Always go through router for decision logic
    return "router"


def route_after_router(state: CMOState) -> Literal["execution", "strategy", "finalize"]:
    """Route based on the router's decision."""
    status = state.get("status", "accept")
    
    if status == "accept":
        return "finalize"
    elif status == "refine":
        return "execution"  # Loop back to execution with feedback
    else:  # reject
        return "strategy"  # Loop back to strategy for replanning


# =============================================================================
# Build LangGraph Workflow
# =============================================================================

def build_cmo_workflow():
    """Build the CMO multi-agent workflow using LangGraph.
    
    Graph structure:
    START → analyst → strategy → execution → critic → router
                         ↑          ↑                    ↓
                         |          └── REFINE ←─────────┤
                         └────────── REJECT ←────────────┘
                                                         ↓
                                                    ACCEPT → finalize → END
    """
    # Create the state graph
    workflow = StateGraph(CMOState)
    
    # Add nodes
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("strategy", strategy_node)
    workflow.add_node("execution", execution_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("router", router_node)
    workflow.add_node("finalize", finalize_node)
    
    # Add edges (main flow)
    workflow.set_entry_point("analyst")
    workflow.add_edge("analyst", "strategy")
    workflow.add_edge("strategy", "execution")
    workflow.add_edge("execution", "critic")
    workflow.add_edge("critic", "router")
    
    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "execution": "execution",  # REFINE loop
            "strategy": "strategy",    # REJECT loop  
            "finalize": "finalize",    # ACCEPT
        }
    )
    
    # Finalize ends the workflow
    workflow.add_edge("finalize", END)
    
    # Compile with memory for checkpointing
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# =============================================================================
# Main Entry Point
# =============================================================================

async def run_cmo_workflow(problem: str, context: dict[str, Any]) -> dict:
    """Run the CMO workflow with the given problem and context.
    
    Args:
        problem: The marketing problem to solve
        context: Business context (budget, channels, historical data, etc.)
    
    Returns:
        The final decision as a dictionary
    """
    # Build workflow
    app = build_cmo_workflow()
    
    # Create initial state
    initial_state = create_initial_state(problem, context)
    
    # Run workflow with unique thread_id for fresh state each run
    import uuid
    config = {"configurable": {"thread_id": f"cmo-workflow-{uuid.uuid4()}"}}
    
    try:
        final_state = await app.ainvoke(initial_state, config)
        return final_state.get("final_decision", {"error": "No decision produced"})
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return {"error": str(e)}


def run_cmo_workflow_sync(problem: str, context: dict[str, Any]) -> dict:
    """Synchronous version of run_cmo_workflow."""
    # Build workflow
    app = build_cmo_workflow()
    
    # Create initial state
    initial_state = create_initial_state(problem, context)
    
    # Run workflow with unique thread_id for fresh state each run
    import uuid
    config = {"configurable": {"thread_id": f"cmo-workflow-{uuid.uuid4()}"}}
    
    try:
        final_state = app.invoke(initial_state, config)
        return final_state.get("final_decision", {"error": "No decision produced"})
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return {"error": str(e)}
