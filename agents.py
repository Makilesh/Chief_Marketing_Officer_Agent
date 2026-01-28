"""CrewAI Specialist Agents for AI CMO Multi-Agent System.

Each agent has a specific role in the marketing decision-making process:
- Analyst: Diagnoses the problem from context and signals
- Strategy: Generates 2-3 viable strategies with trade-offs  
- Execution: Selects best strategy and defines concrete steps
- Critic: Challenges assumptions and provides severity-based feedback

Uses CrewAI for role-based agent definitions.
For Ollama: Uses CrewAI's native LiteLLM format 'ollama/model_name'.
"""

from typing import Any, Union
from crewai import Agent, Task, Crew, Process
from langchain_core.language_models.chat_models import BaseChatModel


# =============================================================================
# Agent Backstories & Goals (CrewAI format)
# =============================================================================

ANALYST_BACKSTORY = """You are a seasoned marketing analyst with 15+ years of experience 
diagnosing marketing performance issues. You've worked with B2B SaaS companies, e-commerce 
brands, and enterprise clients. Your specialty is finding root causes behind metrics changes 
by correlating multiple signals. You're known for your data-driven approach and honest 
confidence assessments - you never overstate certainty when data is limited."""

ANALYST_GOAL = """Diagnose the marketing problem by identifying root causes with supporting 
evidence. Provide confidence scores based on data quality and signal strength."""

STRATEGY_BACKSTORY = """You are a marketing strategist who has developed go-to-market 
strategies for over 50 companies. You understand the trade-offs between speed, cost, and 
risk. You always present 2-3 options because you know there's rarely one perfect answer. 
Your strategies are grounded in resource constraints and historical performance data."""

STRATEGY_GOAL = """Generate 2-3 viable marketing strategies with clear trade-offs, costs, 
timelines, assumptions, and risks. Recommend the best option with justification."""

EXECUTION_BACKSTORY = """You are a marketing operations expert who turns strategies into 
action. You've managed campaigns across Google Ads, LinkedIn, content, and email for 
companies of all sizes. You're meticulous about deadlines, ownership, and success metrics. 
You always include fallback plans because you know marketing rarely goes exactly as planned."""

EXECUTION_GOAL = """Select the best strategy and create a concrete action plan with 
specific steps, owners, deadlines, budget allocation, success metrics, and fallback clauses."""

CRITIC_BACKSTORY = """You are a marketing advisor who has reviewed hundreds of campaign 
plans. Your job is to find the holes - unverified assumptions, missing fallbacks, budget 
violations, and strategic misalignments. You're not negative, but you're thorough. 
Companies trust you because your critiques prevent costly mistakes."""

CRITIC_GOAL = """Critically review the execution plan. Identify issues, assign severity 
scores, and provide specific refinement suggestions. Approve, refine, or reject the plan."""


# =============================================================================
# Agent Output Schemas (for structured outputs)
# =============================================================================

ANALYST_OUTPUT_SCHEMA = """
Respond in valid JSON format with this structure:
{
    "root_cause_hypotheses": [
        {"cause": "description of cause", "signals": ["signal1", "signal2"], "confidence": 0.0-1.0}
    ],
    "primary_diagnosis": "The main root cause identified",
    "confidence": 0.0-1.0,
    "supporting_signals": ["key signal 1", "key signal 2"]
}
"""

STRATEGY_OUTPUT_SCHEMA = """
Respond in valid JSON format with this structure:
{
    "options": [
        {
            "name": "Strategy Name",
            "description": "Strategy description",
            "cost": "$X amount",
            "timeline": "X days/weeks",
            "expected_outcome": "Expected results",
            "assumptions": ["assumption 1", "assumption 2"],
            "risks": ["risk 1", "risk 2"],
            "confidence": 0.0-1.0
        }
    ],
    "recommended_option": "Name of recommended strategy",
    "trade_off_analysis": "Analysis of trade-offs between options",
    "confidence": 0.0-1.0
}
"""

EXECUTION_OUTPUT_SCHEMA = """
Respond in valid JSON format with this structure:
{
    "selected_strategy": "Name of selected strategy",
    "justification": "Why this strategy was selected",
    "action_items": [
        {"step": 1, "action": "Action description", "owner": "Team/Person", "deadline": "Day X or Date"}
    ],
    "budget_allocation": "How budget is split",
    "success_metrics": ["KPI 1", "KPI 2"],
    "checkpoints": ["Checkpoint 1", "Checkpoint 2"],
    "fallback_plan": "What to do if primary fails",
    "confidence": 0.0-1.0
}
"""

CRITIC_OUTPUT_SCHEMA = """
Respond in valid JSON format with this structure:
{
    "severity_score": 0.0-1.0,
    "issues": [
        {"issue": "Issue description", "severity": "low/medium/high", "category": "assumption/risk/constraint/alignment"}
    ],
    "feedback": "Detailed feedback summary",
    "refinement_suggestions": ["Suggestion 1", "Suggestion 2"],
    "approval_status": "approve/refine/reject",
    "reasoning": "Reasoning behind the decision"
}
"""


# =============================================================================
# CrewAI Agent Factory Functions
# =============================================================================

def create_analyst_agent(llm: Union[str, BaseChatModel]) -> Agent:
    """Create the Analyst agent for problem diagnosis.
    
    Args:
        llm: Either a LangChain LLM or a string in CrewAI format (e.g., 'ollama/qwen2.5:14b')
    """
    return Agent(
        role="Marketing Analyst",
        goal=ANALYST_GOAL,
        backstory=ANALYST_BACKSTORY,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def create_strategy_agent(llm: Union[str, BaseChatModel]) -> Agent:
    """Create the Strategy agent for generating options.
    
    Args:
        llm: Either a LangChain LLM or a string in CrewAI format (e.g., 'ollama/qwen2.5:14b')
    """
    return Agent(
        role="Marketing Strategist",
        goal=STRATEGY_GOAL,
        backstory=STRATEGY_BACKSTORY,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def create_execution_agent(llm: Union[str, BaseChatModel]) -> Agent:
    """Create the Execution agent for action planning.
    
    Args:
        llm: Either a LangChain LLM or a string in CrewAI format (e.g., 'ollama/llama3.1:8b')
    """
    return Agent(
        role="Marketing Operations Expert",
        goal=EXECUTION_GOAL,
        backstory=EXECUTION_BACKSTORY,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def create_critic_agent(llm: Union[str, BaseChatModel]) -> Agent:
    """Create the Critic agent for plan review.
    
    Args:
        llm: Either a LangChain LLM or a string in CrewAI format (e.g., 'ollama/llama3.1:8b')
    """
    return Agent(
        role="Marketing Advisor & Critic",
        goal=CRITIC_GOAL,
        backstory=CRITIC_BACKSTORY,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


# =============================================================================
# CrewAI Task Factory Functions
# =============================================================================

def create_analyst_task(
    agent: Agent, 
    problem: str, 
    context: dict[str, Any],
    memory_context: str | None = None
) -> Task:
    """Create the analysis task.
    
    Args:
        agent: The analyst agent
        problem: Marketing problem description
        context: Business context dict
        memory_context: Optional memory context from past campaigns
    """
    context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
    
    # Include memory context if available
    memory_section = ""
    if memory_context:
        memory_section = f"\n{memory_context}\n"
    
    return Task(
        description=f"""
Analyze the following marketing problem and diagnose the root cause.
{memory_section}
MARKETING PROBLEM:
{problem}

BUSINESS CONTEXT:
{context_str}

Your task:
1. Identify 1-3 root cause hypotheses
2. Support each with specific signals/evidence from the context
3. Assign confidence scores based on data quality
4. Identify the primary (most likely) diagnosis
5. If past campaigns are provided above, consider their lessons learned

{ANALYST_OUTPUT_SCHEMA}
""",
        expected_output="A JSON object with root cause hypotheses, primary diagnosis, and confidence score",
        agent=agent,
    )


def create_strategy_task(
    agent: Agent, 
    analyst_output: dict[str, Any], 
    context: dict[str, Any]
) -> Task:
    """Create the strategy generation task."""
    context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
    
    return Task(
        description=f"""
Based on the analyst's diagnosis, generate 2-3 viable marketing strategies.

ANALYST DIAGNOSIS:
Primary Root Cause: {analyst_output.get('primary_diagnosis', 'Unknown')}
Confidence: {analyst_output.get('confidence', 0)}
Supporting Signals: {', '.join(analyst_output.get('supporting_signals', []))}

BUSINESS CONTEXT:
{context_str}

Your task:
1. Create 2-3 distinct strategy options addressing the root cause
2. For each: specify cost, timeline, expected outcome
3. Identify assumptions and risks for each option
4. Recommend the best option with clear justification

{STRATEGY_OUTPUT_SCHEMA}
""",
        expected_output="A JSON object with 2-3 strategy options and a recommendation",
        agent=agent,
    )


def create_execution_task(
    agent: Agent,
    strategy_output: dict[str, Any],
    analyst_output: dict[str, Any],
    context: dict[str, Any],
    critic_feedback: str | None = None
) -> Task:
    """Create the execution planning task."""
    context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
    options_str = "\n".join([
        f"- {opt.get('name', 'Unknown')}: {opt.get('description', '')}"
        for opt in strategy_output.get('options', [])
    ])
    
    feedback_section = ""
    if critic_feedback:
        feedback_section = f"""

CRITIC FEEDBACK (you must address these concerns):
{critic_feedback}

Refine your execution plan to address the critic's concerns.
"""
    
    return Task(
        description=f"""
Create a concrete execution plan for the marketing strategy.

ROOT CAUSE: {analyst_output.get('primary_diagnosis', 'Unknown')}

STRATEGY OPTIONS:
{options_str}

RECOMMENDED OPTION: {strategy_output.get('recommended_option', 'None')}

BUSINESS CONTEXT:
{context_str}
{feedback_section}

Your task:
1. Select the best strategy (consider critic feedback if provided)
2. Create step-by-step action items with owners and deadlines
3. Define budget allocation
4. Set measurable success metrics (KPIs)
5. Add checkpoints for progress review
6. Include a fallback plan

{EXECUTION_OUTPUT_SCHEMA}
""",
        expected_output="A JSON object with selected strategy, action items, budget, metrics, and fallback",
        agent=agent,
    )


def create_critic_task(
    agent: Agent,
    execution_output: dict[str, Any],
    strategy_output: dict[str, Any],
    analyst_output: dict[str, Any],
    context: dict[str, Any],
    iteration: int
) -> Task:
    """Create the critique task."""
    context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
    action_items_str = "\n".join([
        f"- Step {a.get('step', '?')}: {a.get('action', '')} ({a.get('deadline', 'TBD')})"
        for a in execution_output.get('action_items', [])
    ])
    
    # Extract constraints dynamically from context
    budget = context.get('monthly_budget', 'Not specified')
    timeline = context.get('timeline_constraint', 'Not specified')
    
    return Task(
        description=f"""
Critically review the proposed execution plan.

ITERATION: {iteration + 1}

PROPOSED EXECUTION PLAN:
Strategy: {execution_output.get('selected_strategy', 'Unknown')}
Justification: {execution_output.get('justification', '')}
Budget: {execution_output.get('budget_allocation', '')}
Success Metrics: {', '.join(execution_output.get('success_metrics', []))}
Fallback: {execution_output.get('fallback_plan', 'None specified')}

ACTION ITEMS:
{action_items_str}

ORIGINAL DIAGNOSIS:
{analyst_output.get('primary_diagnosis', 'Unknown')} (Confidence: {analyst_output.get('confidence', 0)})

BUSINESS CONSTRAINTS (from context):
{context_str}

KEY CONSTRAINTS TO VALIDATE:
- Budget Limit: {budget}
- Timeline: {timeline}

CRITICAL EVALUATION RULES:
1. BUDGET CHECK: If budget specified, total allocation must NOT exceed it. Within budget = VALID.
2. TIMELINE CHECK: If timeline specified, plan must complete within it. Within timeline = VALID.
3. CONFIDENCE CHECK: Confidence >= 0.7 is acceptable. Do NOT require higher.
4. FALLBACK CHECK: Having ANY fallback plan is acceptable.
5. DEFAULT TO APPROVE: If constraints are met, use severity 0.2-0.3 and approve.

SEVERITY SCORING - BE LENIENT:
- 0.0-0.3: APPROVE - Constraints met, reasonable plan (USE THIS for most valid plans)
- 0.3-0.5: APPROVE with minor notes
- 0.5-0.7: REFINE - Has fixable issues
- 0.7-1.0: REJECT - Critical failures ONLY (budget exceeded, timeline impossible, missing strategy)

BIAS TOWARD APPROVAL: Unless the plan has CRITICAL failures (budget exceeded, timeline impossible), 
you should approve with severity <= 0.3.

{CRITIC_OUTPUT_SCHEMA}
""",
        expected_output="A JSON object with severity score, issues, feedback, and approval status",
        agent=agent,
    )


# =============================================================================
# Helper Functions for Input Formatting (backward compatibility)
# =============================================================================

def format_problem_context(problem: str, context: dict[str, Any]) -> str:
    """Format the problem and context into a structured prompt."""
    context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
    return f"""
MARKETING PROBLEM:
{problem}

CONTEXT:
{context_str}
"""


def get_agent_instructions(agent_type: str) -> str:
    """Get agent backstory for a specific type (backward compatibility)."""
    instructions_map = {
        "analyst": ANALYST_BACKSTORY,
        "strategy": STRATEGY_BACKSTORY,
        "execution": EXECUTION_BACKSTORY,
        "critic": CRITIC_BACKSTORY,
    }
    return instructions_map.get(agent_type.lower(), "")
