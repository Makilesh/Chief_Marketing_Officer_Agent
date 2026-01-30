# AI CMO Multi-Agent System

An intelligent **AI Chief Marketing Officer** that analyzes marketing problems, generates strategies, and makes decisions through a multi-agent critique and refinement loop.

## 🧠 Architecture

```
                    Problem Input
                         ↓
                      Analyst
                         ↓
         ┌──────────→ Strategy ←─────────────┐
         │               ↓                   │
         │           Execution ←────────┐    │
         │               ↓              │    │
         │             Critic           │    │
         │               ↓              │    │
         │            Router            │    │
         │               ↓              │    │
         │    ┌──────────┼──────────┐   │    │
         │    ↓          ↓          ↓   │    │
         │ ACCEPT    REFINE      REJECT │    │
         │    ↓     (0.3-0.7)    (>0.7) │    │
         │    ↓          │          │   │    │
         │  Final        └──────────┘   │    │
         │ Decision                     │    │
         │               └──────────────┘    │
         │                                   │
         └───────────────────────────────────┘
                   (Max 3 iterations)
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | LangGraph | State management, conditional routing, checkpointing |
| **Agents** | CrewAI | Role-based specialist agents with structured outputs |
| **Primary LLM** | Gemini 2.5 Flash | Fast, capable reasoning model |
| **Fallback LLM** | gpt-oss 20B (Ollama) | Local fallback for all agents |

### Agent Roles

| Agent | Role |
|-------|------|
| **Analyst** | Diagnoses problems, identifies root causes |
| **Strategy** | Generates 2-3 viable strategies with trade-offs |
| **Execution** | Selects strategy and creates action plan |
| **Critic** | Challenges assumptions, scores severity |

### Decision Logic

- **Severity < 0.3**: Accept decision ✅
- **Severity 0.3-0.7**: Refine strategy (loop to Execution) 🔄
- **Severity > 0.7**: Reject & replan (loop to Strategy) ❌

## 🎯 Design Philosophy

### Why LangGraph + CrewAI?

| Framework | Purpose |
|-----------|---------|
| **LangGraph** | Explicit state management, conditional routing, iteration control |
| **CrewAI** | Role-based agents with domain expertise, structured outputs |

### Why NOT Just Prompt Chaining?

| Approach | Limitation |
|----------|------------|
| ❌ Prompt chains | Fixed sequences, no self-correction, assumptions go unchallenged |
| ✅ This system | Dynamic routing, critique loops, state-driven decisions |

### Core Innovation

The Critic can **reject decisions and force replanning**—mimicking how a human CMO pressure-tests strategies before executing.

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

> 📄 See [DESIGN.md](DESIGN.md) for detailed architecture documentation.

## 📁 Project Structure

```
Chief_Marketing_Agent/
├── main.py            # Entry point with CLI (interactive/config/demo modes)
├── workflow.py        # LangGraph workflow with conditional routing
├── agents.py          # CrewAI agent definitions & task factories
├── models.py          # Pydantic data models for agent outputs
├── config.py          # LLM configuration & workflow thresholds
├── memory.py          # Memory layer for past campaign retrieval (optional)
├── requirements.txt   # Python dependencies
├── .env               # API keys and configuration
├── DESIGN.md          # System design document
├── README.md          # This file
├── memory/            # Campaign memory storage
│   ├── seed_data.json # Pre-seeded demo campaigns
│   └── campaigns/     # Stored campaign outcomes
└── examples/          # Sample problem configurations
    ├── cac_increase.yaml
    ├── lead_quality.yaml
    └── product_launch.yaml
```

## 🧠 Memory Feature (Optional)

The system can learn from past campaigns using keyword-based memory retrieval:

```bash
# Enable memory layer
set MEMORY_ENABLED=true  # Windows
export MEMORY_ENABLED=true  # Linux/Mac

# Run with memory
python main.py --config examples/cac_increase.yaml
# Output: "[Memory] Found 2 relevant campaigns"
```

When enabled, the Analyst agent receives context from relevant past campaigns:
- **Relevance**: Keyword matching on problem type, business type, channels
- **Priority**: Successful recent campaigns ranked higher
- **Graceful**: System works perfectly if memory disabled or unavailable

## 🚀 Quick Start

### 1. Setup Environment

**Requirements:** Python 3.10+

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Pull Ollama Models (for local fallback)

```bash
ollama pull gpt-oss:20b    # Fallback for all agents (local Ollama model)
```

### 3. Configure API Keys

Create a `.env` file:

```bash
# Primary LLM (required)
GOOGLE_API_KEY=your_google_api_key_here

# Optional cloud fallback
GROQ_API_KEY=your_groq_api_key_here

# Ollama (default: enabled)
USE_OLLAMA=true
OLLAMA_BASE_URL=http://localhost:11434
```

### 4. Run

```bash
# Interactive mode - prompts for your problem
python main.py

# Load from config file
python main.py --config examples/lead_quality.yaml

# Run built-in demo
python main.py --demo

# Test without LLM calls
python main.py --demo --mock
```

### Input Options

| Mode | Command | Description |
|------|---------|-------------|
| **Interactive** | `python main.py` | Prompts for problem and context |
| **Config File** | `python main.py -c input.yaml` | Loads from YAML/JSON file |
| **Demo** | `python main.py --demo` | Runs built-in CAC example |
| **Mock** | `python main.py --mock` | Simulates workflow without LLM |

### Config File Format

```yaml
# examples/my_problem.yaml
problem: |
  Describe your marketing problem here.
  Can be multiple lines.

context:
  business_type: "B2B SaaS"
  monthly_budget: "$10K/month"
  timeline_constraint: "4 weeks"
  # Add any relevant context fields...
```

## 📊 Example Output

```
=============================================================
FINAL DECISION
=============================================================

📋 Decision: Refresh Google Ads creative with LinkedIn fallback
💰 Budget: $2.5K (Google Ads) + $2K reserve (LinkedIn)
📊 Confidence: 78%
🔄 Iterations: 2

📌 Action Items:
   • Step 1: Confirm design availability (Day 0)
   • Step 2: Create 3 new ad variants (Day 1-2)
   • Step 3: A/B test over 5 days (Day 3-7)
   • Step 4: Checkpoint - If CTR < 2% by Day 7, shift to LinkedIn

🎯 Success Metrics:
   • CAC < $180
   • Google Ads CTR > 2%

📜 Decision History:
   Iteration 1: refine (severity: 0.60)
   Iteration 2: approve (severity: 0.20)
```

## 🔧 Configuration

Key settings in `config.py`:

```python
MAX_ITERATIONS = 3                  # Prevent infinite loops
SEVERITY_ACCEPT_THRESHOLD = 0.3     # Accept if severity below
SEVERITY_REFINE_THRESHOLD = 0.7     # Refine if between, Reject if above
```

## 🏗️ Design Principles

| Traditional Approach | This System |
|---------------------|-------------|
| Fixed prompt chain | Dynamic routing based on critique |
| No self-correction | Feedback loops with refinement |
| Hidden state | Explicit state management (LangGraph) |
| Single pass | Iterative improvement up to 3x |

### Key Properties

- **Self-Correcting**: Rejects poor decisions and replans
- **Transparent**: Full reasoning chain in output
- **Bounded**: Iteration limits prevent runaway loops
- **Resilient**: Automatic fallback to local models

## 📚 Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` | Workflow orchestration |
| `crewai` | Role-based agents |
| `langchain-google-genai` | Gemini integration |
| `langchain-ollama` | Local model fallback |
| `pydantic` | Structured outputs |

## � Troubleshooting

| Issue | Solution |
|-------|----------|
| `GOOGLE_API_KEY not set` | Create `.env` file with your Google AI API key |
| Ollama connection failed | Ensure Ollama is running: `ollama serve` |
| Model not found | Pull the model first: `ollama pull gpt-oss:20b` |
| JSON parse errors | Normal - system continues with graceful fallback |
| Infinite loop warnings | Already handled - max 3 iterations enforced |
| Memory errors | Set `MEMORY_ENABLED=false` to disable memory layer |

### Common Environment Variables

```bash
# Required
GOOGLE_API_KEY=your_key_here

# Optional overrides
USE_OLLAMA=true                    # Force Ollama fallback
MEMORY_ENABLED=true                # Enable campaign memory
CREW_VERBOSE=true                  # Show agent reasoning
OLLAMA_BASE_URL=http://localhost:11434  # Ollama endpoint
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and test with `python main.py --demo --mock`
4. Submit a pull request

## 📄 License

MIT
