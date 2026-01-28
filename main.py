"""AI CMO Multi-Agent System - Main Entry Point.

An AI-powered Chief Marketing Officer that diagnoses marketing problems,
develops strategies, and creates actionable execution plans.

Tech Stack:
- LangGraph: Workflow orchestration with conditional routing
- CrewAI: Role-based specialist agents
- Gemini 2.5 Flash: Primary LLM
- Ollama/Groq: Fallback LLM

Usage:
    # Interactive mode - prompts for problem and context
    python main.py
    
    # Load from config file
    python main.py --config problem.yaml
    
    # Run built-in demo example
    python main.py --demo

Environment Variables:
    GOOGLE_API_KEY: Google AI API key for Gemini
    GROQ_API_KEY: Groq API key for Llama (fallback)
    USE_OLLAMA: Set to 'true' for local Ollama models
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Demo Example (for --demo mode)
# =============================================================================

DEMO_PROBLEM = """
CAC (Customer Acquisition Cost) increased 40% in last 30 days.
We need to diagnose the root cause and develop an action plan to bring CAC back down.
"""

DEMO_CONTEXT = {
    "business_type": "B2B SaaS",
    "mrr": "$50K MRR",
    "current_channels": "Google Ads, LinkedIn, Content Marketing",
    "monthly_budget": "$5K/month",
    "historical_linkedin_roas": "2.1x",
    "historical_google_ads_roas": "1.4x",
    "google_ads_ctr_change": "-25% over last 30 days",
    "linkedin_ctr_change": "Stable",
    "content_traffic_change": "+10% but low conversion",
    "target_cac": "$150",
    "current_cac": "$210 (was $150)",
    "design_resources": "1 part-time designer available",
    "timeline_constraint": "Need results within 2 weeks",
}


# =============================================================================
# Input Handling
# =============================================================================

def load_from_config(config_path: str) -> tuple[str, dict]:
    """Load problem and context from a YAML or JSON config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif path.suffix == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml or .json")
    
    problem = data.get('problem', '')
    context = data.get('context', {})
    
    if not problem:
        raise ValueError("Config file must contain a 'problem' field")
    
    return problem, context


def get_interactive_input() -> tuple[str, dict]:
    """Interactively prompt user for problem and context."""
    print("\n" + "=" * 60)
    print("🤖 AI CMO - Marketing Problem Solver")
    print("=" * 60)
    
    print("\n📝 Describe your marketing problem:")
    print("   (Enter your problem, then press Enter twice to finish)")
    
    lines = []
    while True:
        line = input()
        if line == "":
            if lines:
                break
        else:
            lines.append(line)
    problem = "\n".join(lines)
    
    print("\n📊 Now let's gather some context about your business.")
    print("   (Press Enter to skip any field)\n")
    
    context = {}
    
    # Core business info
    prompts = [
        ("business_type", "Business type (e.g., B2B SaaS, E-commerce, Agency)"),
        ("mrr", "Monthly Recurring Revenue (e.g., $50K MRR)"),
        ("monthly_budget", "Marketing budget (e.g., $5K/month)"),
        ("current_channels", "Current marketing channels (e.g., Google Ads, LinkedIn, SEO)"),
        ("timeline_constraint", "Timeline constraint (e.g., Need results within 2 weeks)"),
    ]
    
    for key, prompt in prompts:
        value = input(f"   {prompt}: ").strip()
        if value:
            context[key] = value
    
    # Additional context
    print("\n   Add any additional context (key=value format, empty line to finish):")
    while True:
        line = input("   ").strip()
        if not line:
            break
        if "=" in line:
            key, value = line.split("=", 1)
            context[key.strip()] = value.strip()
    
    return problem, context


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI CMO Multi-Agent System - Marketing Problem Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Interactive mode
  python main.py --config input.yaml  # Load from config file
  python main.py --demo             # Run demo example
  python main.py --demo --mock      # Demo without LLM calls
        """
    )
    
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML/JSON config file with problem and context'
    )
    input_group.add_argument(
        '--demo',
        action='store_true',
        help='Run with built-in demo example (CAC increase scenario)'
    )
    
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Run in mock mode without actual LLM calls (for testing)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


async def run_workflow(problem: str, context: dict):
    """Run the AI CMO workflow with the given problem and context."""
    from config import (
        GOOGLE_API_KEY, 
        GROQ_API_KEY, 
        GEMINI_MODEL, 
        LLAMA_MODEL,
        USE_OLLAMA,
        OLLAMA_PLANNER_MODEL,
        OLLAMA_EXECUTION_MODEL,
        get_llm_for_agent,
    )
    from workflow import run_cmo_workflow
    
    # Check for API keys OR Ollama
    if not GOOGLE_API_KEY and not GROQ_API_KEY and not USE_OLLAMA:
        logger.error("No API keys found and Ollama is disabled!")
        logger.info("Set GOOGLE_API_KEY for Gemini 2.5 Flash (primary)")
        logger.info("Set GROQ_API_KEY for Llama 3.1 8B (cloud fallback)")
        logger.info("Or set USE_OLLAMA=true for local models")
        return None
    
    logger.info("=" * 60)
    logger.info("AI CMO Multi-Agent System")
    logger.info("=" * 60)
    logger.info("Tech Stack:")
    logger.info("  - LangGraph: Workflow orchestration")
    logger.info("  - CrewAI: Role-based agents")
    if GOOGLE_API_KEY:
        logger.info(f"  - Primary LLM: {GEMINI_MODEL}")
    if USE_OLLAMA:
        logger.info(f"  - Planner LLM (Ollama): {OLLAMA_PLANNER_MODEL}")
        logger.info(f"  - Execution LLM (Ollama): {OLLAMA_EXECUTION_MODEL}")
    elif GROQ_API_KEY:
        logger.info(f"  - Fallback LLM (Groq): {LLAMA_MODEL}")
    logger.info("-" * 60)
    
    # Verify LLM is working
    try:
        llm = get_llm_for_agent('analyst')
        logger.info("✓ LLM configured successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return None
    
    # Display the problem
    logger.info("\n" + "=" * 60)
    logger.info("PROBLEM:")
    logger.info(problem.strip())
    logger.info("-" * 60)
    logger.info("CONTEXT:")
    for key, value in context.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60 + "\n")
    
    try:
        result = await run_cmo_workflow(
            problem=problem,
            context=context,
        )
        return result
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_final_decision(result: dict):
    """Pretty print the final decision."""
    print("\n" + "=" * 60)
    print("FINAL DECISION")
    print("=" * 60)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"\n📋 Decision: {result.get('decision', 'N/A')}")
    print(f"💰 Budget: {result.get('budget', 'N/A')}")
    print(f"📊 Confidence: {result.get('confidence', 0):.0%}")
    print(f"🔄 Iterations: {result.get('iterations', 0)}")
    
    print("\n📌 Action Items:")
    for item in result.get('action_items', []):
        print(f"   • {item}")
    
    print("\n🎯 Success Metrics:")
    for metric in result.get('success_metrics', []):
        print(f"   • {metric}")
    
    print("\n💭 Reasoning Chain:")
    print(result.get('reasoning', 'N/A'))
    
    print("\n❌ Alternatives Rejected:")
    for alt in result.get('alternatives_rejected', []):
        print(f"   • {alt}")
    
    print("\n📜 Decision History:")
    for entry in result.get('decision_history', []):
        print(f"   Iteration {entry['iteration']}: {entry['status']} (severity: {entry['severity']:.2f})")
    
    print("\n" + "=" * 60)


async def run_mock_demo(problem: str, context: dict):
    """Run a mock demonstration without actual LLM calls."""
    logger.info("\n" + "=" * 60)
    logger.info("MOCK DEMONSTRATION (No LLM)")
    logger.info("=" * 60)
    
    logger.info("\nPROBLEM:")
    logger.info(problem.strip())
    logger.info("-" * 60)
    logger.info("CONTEXT:")
    for key, value in context.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    # Simulate the workflow
    mock_result = {
        "decision": "Refresh Google Ads creative with LinkedIn fallback",
        "strategy": "Addresses root cause (ad fatigue) directly while maintaining safety net",
        "budget": "$2.5K (Google Ads refresh) + $2K reserve (LinkedIn fallback)",
        "timeline": "7-14 days",
        "action_items": [
            "Step 1: Confirm design availability (Day 0)",
            "Step 2: Create 3 new ad variants (Day 1-2)",
            "Step 3: A/B test over 5 days (Day 3-7)",
            "Step 4: Checkpoint - If CTR < 2% by Day 7, shift $2K to LinkedIn",
            "Step 5: Monitor and report (Day 7-14)",
        ],
        "success_metrics": [
            "CAC < $180",
            "Google Ads CTR > 2%",
            "Overall ROAS improvement",
        ],
        "confidence": 0.78,
        "iterations": 2,
        "reasoning": """
- Root cause identified: Ad creative fatigue on Google Ads
  Confidence: 0.75
- Strategy selected: Refresh Google Ads with LinkedIn fallback
  Trade-offs: Faster results with safety net, respects budget
- Execution plan: Phased approach with clear checkpoints
  Justification: Balances speed with risk mitigation
- Critic review: approve
  Severity: 0.2
""",
        "alternatives_rejected": [
            "Full LinkedIn shift: Slower execution, leaves Google Ads underutilized",
            "Complete channel overhaul: Too risky and expensive for current constraints",
        ],
        "decision_history": [
            {"iteration": 1, "strategy": "Refresh Google Ads creative", "severity": 0.6, "status": "refine"},
            {"iteration": 2, "strategy": "Refresh Google Ads with LinkedIn fallback", "severity": 0.2, "status": "approve"},
        ],
    }
    
    # Simulate iteration progress
    logger.info("\n--- ITERATION 1 ---")
    logger.info("[Analyst] Diagnosing problem...")
    logger.info("[Analyst] Root cause: Ad creative fatigue on Google Ads (confidence: 0.75)")
    logger.info("[Strategy] Generating options...")
    logger.info("[Strategy] Recommended: Refresh Google Ads creative")
    logger.info("[Execution] Creating action plan...")
    logger.info("[Critic] Reviewing plan...")
    logger.info("[Critic] Severity: 0.6 (MEDIUM)")
    logger.info("[Critic] Issues:")
    logger.info("  - Assumes design resources available (unverified)")
    logger.info("  - No fallback if CTR doesn't improve")
    logger.info("[Router] → REFINE")
    
    await asyncio.sleep(0.5)
    
    logger.info("\n--- ITERATION 2 ---")
    logger.info("[Execution] Refining plan with critic feedback...")
    logger.info("[Execution] Added: Design availability check, LinkedIn fallback clause")
    logger.info("[Critic] Reviewing refined plan...")
    logger.info("[Critic] Severity: 0.2 (LOW)")
    logger.info("[Critic] Status: APPROVED")
    logger.info("[Router] → ACCEPT")
    
    await asyncio.sleep(0.5)
    
    print_final_decision(mock_result)


def main():
    """Main entry point."""
    load_dotenv()
    
    args = parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine input source
    try:
        if args.config:
            logger.info(f"Loading problem from config: {args.config}")
            problem, context = load_from_config(args.config)
        elif args.demo:
            logger.info("Running demo example...")
            problem, context = DEMO_PROBLEM, DEMO_CONTEXT
        else:
            # Interactive mode
            problem, context = get_interactive_input()
        
        # Validate we have a problem
        if not problem.strip():
            logger.error("No problem provided. Exiting.")
            sys.exit(1)
        
        # Run the appropriate mode
        if args.mock:
            asyncio.run(run_mock_demo(problem, context))
        else:
            result = asyncio.run(run_workflow(problem, context))
            if result:
                print_final_decision(result)
            else:
                logger.error("Workflow did not produce a result.")
                sys.exit(1)
                
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
