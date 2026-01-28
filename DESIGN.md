# AI CMO Multi-Agent System — Design Document

## What This Is

A **multi-agent reasoning system** that simulates how an AI CMO makes marketing decisions through iterative critique and refinement. Built with LangGraph (orchestration) + CrewAI (agents) to demonstrate agentic AI architecture.

**Key Innovation:** Self-correcting decision loops—not sequential prompt chains.

---

## The Problem

Marketing decisions are complex:
- Multiple variables to consider (budget, timeline, channels, resources)
- Trade-offs between strategies
- Need for validation before execution
- Risk of costly mistakes

Traditional AI approaches use **linear prompt chains**: `Analyze → Recommend → Done`. This fails because:
- No self-correction mechanism
- Assumptions go unchallenged  
- Constraints can be violated
- No iterative refinement

---

## The Solution: Agentic Decision Loop

```
Problem → [Analyst] → [Strategy] → [Execution] → [Critic]
                                         ↑           ↓
                                         └─ REFINE ──┤
                                                     ↓
                                    [Strategy] ← REJECT
                                                     ↓
                                                  ACCEPT → Final Decision
```

**Core Insight:** The Critic agent can reject decisions and force replanning—mimicking how a human CMO pressure-tests strategies before executing.

### Example Flow

```
Execution: "Spend $5K on Google Ads"
    ↓
Critic: "Budget violation (max $3K)" → Severity 0.8 → REJECT
    ↓
Strategy: Regenerate with $3K constraint
    ↓
Execution: "Spend $2.5K on Google Ads + $500 LinkedIn fallback"
    ↓
Critic: "Acceptable" → Severity 0.2 → ACCEPT
```

This is **agentic reasoning**, not template filling.

---

## Architecture

### Technology Stack

| Layer | Technology | Why This Choice |
|-------|------------|-----------------|
| **Orchestration** | LangGraph | Explicit state management, conditional routing, iteration control |
| **Agents** | CrewAI | Role-based specialists with structured outputs, built-in delegation |
| **Primary LLM** | Gemini 2.5 Flash | Fast, capable, good at structured JSON output |
| **Fallback LLM** | Ollama (Llama 3.1 8B) | Local execution, no API costs, offline capability |

### Agent Roles

| Agent | Responsibility | Output |
|-------|----------------|--------|
| **Analyst** | Diagnose root causes from symptoms | `{root_cause, confidence, evidence}` |
| **Strategy** | Generate 2-3 viable options with trade-offs | `{options[], recommended, trade_offs}` |
| **Execution** | Create actionable plan from strategy | `{action_items[], budget, timeline, fallback}` |
| **Critic** | Challenge assumptions, validate constraints | `{severity_score, issues[], approval_status}` |

### State Management

LangGraph maintains workflow state across iterations:

```python
class CMOState(TypedDict):
    problem: str           # Input problem statement
    context: dict          # Business constraints
    analysis: dict         # Analyst output
    strategy: dict         # Strategy output  
    execution: dict        # Execution plan
    critic_feedback: dict  # Critic assessment
    iteration: int         # Current loop count
    decision_history: list # All iterations
```

### Routing Logic

The Router node implements the decision loop:

```python
if severity_score < 0.3:     → ACCEPT (finalize)
elif severity_score < 0.7:   → REFINE (back to Execution with feedback)
else:                        → REJECT (back to Strategy for new approach)
```

Maximum 3 iterations prevents infinite loops.

---

## Key Design Decisions

### 1. Structured Outputs (JSON + Pydantic)

**Why:** LLM outputs are unpredictable. Structured formats:
- Enable reliable parsing
- Allow validation before use
- Make debugging tractable

```python
class CriticOutput(BaseModel):
    severity_score: float = Field(ge=0.0, le=1.0)
    issues: List[dict]
    approval_status: Literal["approve", "refine", "reject"]
```

### 2. Dynamic Constraint Handling

**Why:** Hardcoded rules break when context changes.

The Critic reads constraints from context:
```python
budget_limit = context.get('monthly_budget', 'Not specified')
timeline = context.get('timeline_constraint', 'Not specified')
```

This makes the system generalizable across different problems.

### 3. Single LLM Factory

**Why:** CrewAI agents need consistent LLM types.

```python
def get_llm_for_agent(agent_type: str) -> LLM:
    # Returns properly configured LLM for any agent
    # Handles Gemini primary → Ollama fallback
```

### 4. Graceful Error Handling

**Why:** LLM failures shouldn't crash the workflow.

```python
def parse_json_output(text: str) -> dict:
    try:
        return json.loads(cleaned_text)
    except:
        return {"error": "Parse failed", "raw": text}
```

---

## Failure Modes & Mitigations

| Failure | Mitigation |
|---------|------------|
| Invalid JSON from LLM | `parse_json_output()` returns error dict, workflow continues |
| Missing context constraints | Critic falls back to lenient approval |
| Iteration exhaustion | Router stops at max iterations, finalizes best plan |
| LLM API failure | Automatic fallback to Ollama local models |
| Pydantic validation failure | Caught and logged, uses raw output |

---

## 🔮 Future Enhancements

### Short-term (Week 1-2)
- [ ] Vector DB integration (Pinecone/Weaviate) for semantic campaign retrieval
- [x] ~~Keyword-based campaign memory retrieval~~ ✅ Implemented in `memory.py`
- [x] ~~Retrieval of past campaign outcomes to inform strategy~~ ✅ Analyst receives memory context
- [x] ~~Parameterized critic thresholds via config~~ ✅ Available via environment variables

### Medium-term (Month 1)
- [ ] Evaluation framework: track decision quality over time
- [ ] Confidence calibration: tune severity thresholds based on outcomes
- [ ] Multi-campaign coordination: handle dependencies between campaigns
- [ ] A/B test result integration

### Long-term (Month 2-3)
- [ ] Tool-calling: integrate with ad platforms (Google Ads API, LinkedIn API)
- [ ] Real-time feedback loops: adjust strategies based on live campaign data
- [ ] Budget optimization agent: dynamically allocate across channels
- [ ] Autonomous execution with human-in-the-loop approval gates

---

## Running the System

```bash
# Interactive mode
python main.py

# Load from config file
python main.py --config examples/lead_quality.yaml

# Demo with built-in example
python main.py --demo

# Test without LLM calls
python main.py --demo --mock
```

---

## Implementation Notes

This design document describes the architecture. See the code for implementation:

- `main.py` — Entry point with CLI
- `workflow.py` — LangGraph state machine
- `agents.py` — CrewAI agent definitions
- `models.py` — Pydantic output schemas
- `config.py` — LLM configuration
- `memory.py` — Campaign memory layer

Test the system: `python main.py --demo --mock`