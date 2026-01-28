"""Configuration for AI CMO Multi-Agent System.

LLM Setup:
- Primary: Gemini 2.5 Flash (Google)
- Planner Fallback: Qwen 2.5 14B (Ollama) - for Analyst & Strategy agents
- Execution Fallback: Llama 3.1 8B (Ollama) - for Execution & Critic agents
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application.
    
    Args:
        verbose: If True, set DEBUG level. Otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Keep CrewAI at INFO level unless verbose
    if not verbose:
        logging.getLogger("crewai").setLevel(logging.INFO)


# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# LLM Configuration
# =============================================================================

# Primary LLM: Gemini 2.5 Flash
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")

# Cloud Fallback: Llama 3.1 8B via Groq (if Ollama not available)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama-3.1-8b-instant")

# Local Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"  # Default to true now

# Ollama Models for different agent types
# Using llama3.1:8b for all agents to avoid Chinese text mixing from Qwen
OLLAMA_PLANNER_MODEL = os.getenv("OLLAMA_PLANNER_MODEL", "llama3.1:8b")  # Analyst & Strategy
OLLAMA_EXECUTION_MODEL = os.getenv("OLLAMA_EXECUTION_MODEL", "llama3.1:8b")  # Execution & Critic

# =============================================================================
# Workflow Configuration
# =============================================================================

MAX_ITERATIONS = 3  # Prevent infinite refinement loops
SEVERITY_ACCEPT_THRESHOLD = 0.3  # Accept decision if severity < 0.3
SEVERITY_REFINE_THRESHOLD = 0.7  # Refine if severity 0.3-0.7, reject if > 0.7

# Confidence Weights (as per spec)
ANALYST_CONFIDENCE_WEIGHT = 0.3
STRATEGY_CONFIDENCE_WEIGHT = 0.4
CRITIQUE_WEIGHT = 0.3

# =============================================================================
# CrewAI Configuration
# =============================================================================

CREW_VERBOSE = os.getenv("CREW_VERBOSE", "true").lower() == "true"
CREW_MEMORY = os.getenv("CREW_MEMORY", "true").lower() == "true"

# =============================================================================
# Memory Layer Configuration
# =============================================================================

MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
MEMORY_MIN_RELEVANCE = float(os.getenv("MEMORY_MIN_RELEVANCE", "0.6"))
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "3"))


# =============================================================================
# LLM Factory Functions
# =============================================================================

def get_llm_for_agent(agent_type: str):
    """Get appropriate LLM for the specified agent type.
    
    Primary: Gemini 2.5 Flash (if GOOGLE_API_KEY set and USE_OLLAMA=false)
    Fallback: Ollama (llama3.1:8b via OpenAI-compatible endpoint)
    
    Args:
        agent_type: One of 'analyst', 'strategy', 'execution', 'critic'
    
    Returns:
        CrewAI LLM object
    """
    from crewai import LLM
    
    # Primary: Use Gemini if API key is available and not forcing Ollama
    if GOOGLE_API_KEY and not USE_OLLAMA:
        try:
            return LLM(
                model=GEMINI_MODEL,
                provider="google",
                api_key=GOOGLE_API_KEY,
                temperature=0.7,
            )
        except Exception as e:
            logger.warning(f"[Config] Gemini failed: {e}, falling back to Ollama...")
    
    # Fallback: Use Ollama with OpenAI-compatible endpoint
    if agent_type in ("analyst", "strategy"):
        model = OLLAMA_PLANNER_MODEL
    else:  # execution, critic
        model = OLLAMA_EXECUTION_MODEL
    
    return LLM(
        model=model,
        provider="openai",
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",  # Ollama doesn't need a real key
        temperature=0.7,
    )


# Legacy functions for backward compatibility (deprecated)
def get_primary_llm():
    """DEPRECATED: Use get_llm_for_agent() instead."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set.")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        convert_system_message_to_human=True,
    )


def get_fallback_llm():
    """DEPRECATED: Use get_llm_for_agent() instead."""
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=OLLAMA_EXECUTION_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
    )


def get_llm_with_fallback():
    """DEPRECATED: Use get_llm_for_agent() instead."""
    try:
        primary = get_primary_llm()
        fallback = get_fallback_llm()
        return primary.with_fallbacks([fallback])
    except ValueError:
        return get_fallback_llm()
