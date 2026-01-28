"""Data models for AI CMO Multi-Agent System.

Defines structured outputs for each agent role following the specification.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class DecisionStatus(str, Enum):
    """Decision routing status based on critique severity."""
    ACCEPT = "accept"
    REFINE = "refine"
    REJECT = "reject"


# =============================================================================
# Agent Output Models (Pydantic for structured outputs)
# =============================================================================

class RootCauseHypothesis(BaseModel):
    """A single root cause hypothesis with supporting signals."""
    cause: str = Field(description="The identified root cause")
    signals: list[str] = Field(description="Supporting signals/evidence")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")


class AnalystOutput(BaseModel):
    """Output from the Analyst Agent - diagnoses the marketing problem."""
    root_cause_hypotheses: list[RootCauseHypothesis] = Field(
        description="List of root cause hypotheses ranked by confidence"
    )
    primary_diagnosis: str = Field(description="The most likely root cause")
    confidence: float = Field(ge=0, le=1, description="Overall analyst confidence 0-1")
    supporting_signals: list[str] = Field(description="Key signals supporting the diagnosis")


class StrategyOption(BaseModel):
    """A single strategy option with trade-offs."""
    name: str = Field(description="Strategy name")
    description: str = Field(description="Strategy description")
    cost: str = Field(description="Estimated cost")
    timeline: str = Field(description="Expected timeline")
    expected_outcome: str = Field(description="Expected outcome/metrics")
    assumptions: list[str] = Field(description="Key assumptions")
    risks: list[str] = Field(description="Potential risks")
    confidence: float = Field(ge=0, le=1, description="Confidence in this strategy")


class StrategyOutput(BaseModel):
    """Output from the Strategy Agent - generates viable strategies."""
    options: list[StrategyOption] = Field(
        description="2-3 viable strategy options"
    )
    recommended_option: str = Field(description="Name of recommended option")
    trade_off_analysis: str = Field(description="Analysis of trade-offs between options")
    confidence: float = Field(ge=0, le=1, description="Overall strategy confidence 0-1")
    
    @field_validator('options')
    @classmethod
    def validate_options_count(cls, v):
        if len(v) < 2 or len(v) > 3:
            raise ValueError(f"Must have 2-3 strategy options, got {len(v)}")
        return v


class ActionItem(BaseModel):
    """A concrete action item for execution."""
    step: int = Field(description="Step number")
    action: str = Field(description="Action to take")
    owner: str = Field(default="Marketing Team", description="Who owns this action")
    deadline: str = Field(description="When this should be completed")


class ExecutionOutput(BaseModel):
    """Output from the Execution Agent - concrete action plan."""
    selected_strategy: str = Field(description="Name of selected strategy")
    justification: str = Field(description="Why this strategy was selected")
    action_items: list[ActionItem] = Field(description="Concrete action steps")
    budget_allocation: str = Field(description="How budget is allocated")
    success_metrics: list[str] = Field(description="KPIs to measure success")
    checkpoints: list[str] = Field(description="Review checkpoints")
    fallback_plan: Optional[str] = Field(default=None, description="Fallback if primary fails")
    confidence: float = Field(ge=0, le=1, description="Execution confidence 0-1")


class CriticIssue(BaseModel):
    """A specific issue identified by the Critic."""
    issue: str = Field(description="The identified issue")
    severity: str = Field(description="low/medium/high")
    category: str = Field(description="assumption/risk/constraint/alignment")


class CriticOutput(BaseModel):
    """Output from the Critic Agent - challenges and refines decisions."""
    severity_score: float = Field(
        ge=0, le=1,
        description="Overall severity 0-1 (0=no issues, 1=critical issues)"
    )
    issues: list[CriticIssue] = Field(description="List of identified issues")
    feedback: str = Field(description="Detailed feedback summary")
    refinement_suggestions: list[str] = Field(description="Specific suggestions for improvement")
    approval_status: str = Field(description="approve/refine/reject")
    reasoning: str = Field(description="Reasoning behind the decision")


# =============================================================================
# Workflow State Model
# =============================================================================

@dataclass
class CMOWorkflowState:
    """Maintains state across the CMO workflow iterations."""
    problem: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    
    # Agent outputs
    analyst_output: Optional[AnalystOutput] = None
    strategy_output: Optional[StrategyOutput] = None
    execution_output: Optional[ExecutionOutput] = None
    critic_output: Optional[CriticOutput] = None
    
    # Iteration tracking
    iteration: int = 0
    max_iterations: int = 3
    decision_history: list[dict] = field(default_factory=list)
    
    # Final decision
    final_decision: Optional[dict] = None
    final_confidence: float = 0.0
    status: DecisionStatus = DecisionStatus.REFINE

    def calculate_confidence(self) -> float:
        """Calculate final confidence using weighted formula from spec."""
        if not all([self.analyst_output, self.strategy_output, self.critic_output]):
            return 0.0
        
        analyst_conf = self.analyst_output.confidence * 0.3
        strategy_conf = self.strategy_output.confidence * 0.4
        critique_factor = (1 - self.critic_output.severity_score) * 0.3
        
        return round(analyst_conf + strategy_conf + critique_factor, 2)


@dataclass
class FinalDecision(BaseModel):
    """The final output from the CMO workflow."""
    decision: str = Field(description="The final decision")
    strategy: str = Field(description="Selected strategy")
    budget: str = Field(description="Budget allocation")
    timeline: str = Field(description="Timeline")
    success_metrics: list[str] = Field(description="Success metrics")
    confidence: float = Field(description="Final confidence score")
    iterations: int = Field(description="Number of iterations taken")
    reasoning: str = Field(description="Full reasoning chain")
    alternatives_rejected: list[str] = Field(description="Alternatives that were rejected")

    class Config:
        """Pydantic config."""
        from_attributes = True
