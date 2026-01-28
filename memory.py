"""Memory Layer for AI CMO Multi-Agent System.

Provides conditional retrieval with priority weighting for past campaigns.
Memory is optional and the system works perfectly without it.

Features:
- Keyword-based relevance detection (no embeddings)
- Numerical priority scoring
- File-based JSON storage
- Graceful degradation if memory fails
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# Problem Type Detection
# =============================================================================

PROBLEM_TYPE_KEYWORDS = {
    "cac_increase": [
        "cac", "customer acquisition cost", "cost per customer", "acquisition cost",
        "cost increased", "cac increased", "acquisition", "cost per acquisition"
    ],
    "lead_quality": [
        "lead quality", "sql", "mql", "qualified leads", "conversion rate",
        "lead scoring", "poor leads", "unqualified", "lead conversion"
    ],
    "engagement_drop": [
        "engagement", "ctr", "click rate", "interaction", "bounce rate",
        "engagement drop", "low engagement", "click through"
    ],
    "product_launch": [
        "launch", "new product", "feature release", "go-to-market", "gtm",
        "product launch", "new feature", "launching"
    ],
    "budget_optimization": [
        "budget", "spend", "roas", "roi", "cost efficiency",
        "budget allocation", "overspend", "underspend"
    ],
}


def extract_problem_type(problem_text: str) -> str:
    """Extract problem type from text using keyword matching.
    
    Args:
        problem_text: The problem description
        
    Returns:
        Problem type string (e.g., 'cac_increase') or 'general'
    """
    text_lower = problem_text.lower()
    
    for ptype, keywords in PROBLEM_TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return ptype
    
    return "general"


def extract_channels(context: dict[str, Any]) -> list[str]:
    """Extract marketing channels from context."""
    channels = []
    
    # Check common context fields
    for key in ["channels", "current_channels", "marketing_channels", "assets_available"]:
        value = context.get(key)
        if value:
            if isinstance(value, list):
                channels.extend(value)
            elif isinstance(value, str):
                # Parse comma-separated channels
                channels.extend([c.strip() for c in value.split(",")])
    
    return channels


def calculate_channel_overlap(current: list[str], past: list[str]) -> float:
    """Calculate channel overlap using Jaccard similarity.
    
    Args:
        current: Current campaign channels
        past: Past campaign channels
        
    Returns:
        Overlap score 0.0-1.0
    """
    if not current or not past:
        return 0.0
    
    current_set = set(c.lower().strip() for c in current)
    past_set = set(c.lower().strip() for c in past)
    
    intersection = len(current_set & past_set)
    union = len(current_set | past_set)
    
    return intersection / union if union > 0 else 0.0


def extract_business_type(context: dict[str, Any]) -> str:
    """Extract business type from context."""
    for key in ["business_type", "company_type", "industry"]:
        if key in context:
            return str(context[key]).lower()
    return ""


# =============================================================================
# Memory Manager Class
# =============================================================================

class MemoryManager:
    """Manages campaign memory storage and retrieval.
    
    Uses file-based JSON storage with keyword matching for relevance.
    """
    
    def __init__(self, memory_dir: str = "memory"):
        """Initialize memory manager.
        
        Args:
            memory_dir: Directory for memory storage (relative to project root)
        """
        self.memory_dir = Path(memory_dir)
        self.campaigns_dir = self.memory_dir / "campaigns"
        self.seed_file = self.memory_dir / "seed_data.json"
        
        # Ensure directories exist
        self.campaigns_dir.mkdir(parents=True, exist_ok=True)
        
        # Load seed data if campaigns dir is empty
        self._load_seed_data_if_needed()
    
    def _load_seed_data_if_needed(self):
        """Load seed data into campaigns directory if empty."""
        # Check if any campaign files exist
        existing_campaigns = list(self.campaigns_dir.glob("*.json"))
        
        if existing_campaigns:
            return  # Already have campaigns
        
        if not self.seed_file.exists():
            logger.warning("[Memory] No seed data found, starting with empty memory")
            return
        
        try:
            with open(self.seed_file, 'r', encoding='utf-8') as f:
                seed_campaigns = json.load(f)
            
            for campaign in seed_campaigns:
                campaign_id = campaign.get("campaign_id", str(uuid4())[:8])
                problem_type = campaign.get("problem_type", "general")
                timestamp = campaign.get("timestamp", datetime.now().isoformat())
                date_str = timestamp[:10]  # Extract YYYY-MM-DD
                
                filename = f"{date_str}_{problem_type}_{campaign_id}.json"
                filepath = self.campaigns_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(campaign, f, indent=2)
            
            logger.info(f"[Memory] Loaded {len(seed_campaigns)} seed campaigns")
            
        except Exception as e:
            logger.warning(f"[Memory] Failed to load seed data: {e}")
    
    def get_all_campaigns(self) -> list[dict]:
        """Load all campaigns from storage."""
        campaigns = []
        
        for filepath in self.campaigns_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    campaign = json.load(f)
                    campaigns.append(campaign)
            except Exception as e:
                logger.warning(f"[Memory] Failed to load {filepath}: {e}")
        
        return campaigns
    
    def calculate_relevance(
        self, 
        current_problem: str, 
        current_context: dict[str, Any],
        past_campaign: dict
    ) -> float:
        """Calculate relevance score between current problem and past campaign.
        
        Scoring logic:
        - Problem type match: +0.4
        - Business type match: +0.2  
        - Channel overlap: +0.2
        - Budget range similarity: +0.1
        - Timeline similarity: +0.1
        
        Args:
            current_problem: Current problem text
            current_context: Current business context
            past_campaign: Past campaign data
            
        Returns:
            Relevance score 0.0-1.0
        """
        relevance = 0.0
        
        # 1. Problem type match (+0.4)
        current_type = extract_problem_type(current_problem)
        past_type = past_campaign.get("problem_type", "general")
        
        if current_type == past_type and current_type != "general":
            relevance += 0.4
        elif current_type != "general" and past_type != "general":
            # Partial match for related types
            if any(kw in past_campaign.get("problem_text", "").lower() 
                   for kw in PROBLEM_TYPE_KEYWORDS.get(current_type, [])):
                relevance += 0.2
        
        # 2. Business type match (+0.2)
        current_biz = extract_business_type(current_context)
        past_biz = extract_business_type(past_campaign.get("context", {}))
        
        if current_biz and past_biz:
            # Check for overlap (B2B SaaS vs B2B)
            if current_biz in past_biz or past_biz in current_biz:
                relevance += 0.2
            elif any(word in past_biz for word in current_biz.split()):
                relevance += 0.1
        
        # 3. Channel overlap (+0.2)
        current_channels = extract_channels(current_context)
        past_channels = past_campaign.get("context", {}).get("channels", [])
        
        if isinstance(past_channels, str):
            past_channels = [c.strip() for c in past_channels.split(",")]
        
        channel_overlap = calculate_channel_overlap(current_channels, past_channels)
        relevance += channel_overlap * 0.2
        
        # 4. Budget range similarity (+0.1)
        current_budget = str(current_context.get("monthly_budget", 
                            current_context.get("available_budget", ""))).lower()
        past_budget = str(past_campaign.get("context", {}).get("budget", "")).lower()
        
        if current_budget and past_budget:
            # Simple check: both contain similar order of magnitude
            if any(amt in current_budget and amt in past_budget 
                   for amt in ["$1k", "$2k", "$3k", "$4k", "$5k", "$10k", "$20k", "$50k"]):
                relevance += 0.1
            elif "$" in current_budget and "$" in past_budget:
                relevance += 0.05  # Partial credit for having budget info
        
        # 5. Timeline similarity (+0.1)
        current_timeline = str(current_context.get("timeline_constraint", 
                              current_context.get("timeline", ""))).lower()
        past_timeline = str(past_campaign.get("context", {}).get("timeline", "")).lower()
        
        if current_timeline and past_timeline:
            # Check for similar timeframes
            if any(t in current_timeline and t in past_timeline 
                   for t in ["week", "2 week", "1 week", "month", "day"]):
                relevance += 0.1
        
        return round(min(relevance, 1.0), 2)
    
    def calculate_priority(self, campaign: dict) -> float:
        """Calculate priority score for a past campaign.
        
        Priority factors:
        - Outcome success: +0.5
        - Recency: +0.2 (last 30 days), +0.1 (30-90 days)
        - Confidence: +0.3 * confidence_score
        
        Args:
            campaign: Campaign data
            
        Returns:
            Priority score 0.0-1.0
        """
        priority = 0.0
        
        # 1. Outcome success (+0.5)
        outcome = campaign.get("outcome", {})
        if outcome.get("success", False):
            priority += 0.5
        
        # 2. Recency (+0.2 max)
        try:
            timestamp_str = campaign.get("timestamp", "")
            if timestamp_str:
                # Handle ISO format with or without timezone
                timestamp_str = timestamp_str.replace("Z", "+00:00")
                if "+" not in timestamp_str and "-" not in timestamp_str[10:]:
                    timestamp = datetime.fromisoformat(timestamp_str)
                else:
                    timestamp = datetime.fromisoformat(timestamp_str.split("+")[0])
                
                age_days = (datetime.now() - timestamp).days
                
                if age_days <= 30:
                    priority += 0.2
                elif age_days <= 90:
                    priority += 0.1
        except (ValueError, TypeError):
            pass  # Skip recency bonus if timestamp is invalid
        
        # 3. Confidence (+0.3 * confidence)
        confidence = outcome.get("confidence", 0.5)
        priority += confidence * 0.3
        
        return round(min(priority, 1.0), 2)
    
    def get_relevant_campaigns(
        self,
        problem: str,
        context: dict[str, Any],
        min_relevance: float = 0.6,
        top_k: int = 3
    ) -> list[dict]:
        """Get relevant past campaigns for the current problem.
        
        Args:
            problem: Current problem text
            context: Current business context
            min_relevance: Minimum relevance threshold (default 0.6)
            top_k: Maximum number of campaigns to return
            
        Returns:
            List of relevant campaigns sorted by priority (highest first)
        """
        all_campaigns = self.get_all_campaigns()
        
        if not all_campaigns:
            logger.info("[Memory] No campaigns in memory")
            return []
        
        # Calculate relevance and priority for each campaign
        scored_campaigns = []
        for campaign in all_campaigns:
            relevance = self.calculate_relevance(problem, context, campaign)
            
            if relevance >= min_relevance:
                priority = self.calculate_priority(campaign)
                campaign["_relevance"] = relevance
                campaign["_priority"] = priority
                scored_campaigns.append(campaign)
        
        if not scored_campaigns:
            logger.info("[Memory] No relevant campaigns found (threshold: {min_relevance})")
            return []
        
        # Sort by priority (descending), then by relevance (descending)
        scored_campaigns.sort(key=lambda x: (x["_priority"], x["_relevance"]), reverse=True)
        
        # Return top_k campaigns
        result = scored_campaigns[:top_k]
        
        logger.info(f"[Memory] Found {len(result)} relevant campaigns")
        for c in result:
            logger.debug(f"  - {c.get('problem_type')}: relevance={c['_relevance']}, priority={c['_priority']}")
        
        return result
    
    def store_campaign(
        self,
        problem: str,
        context: dict[str, Any],
        diagnosis: str,
        strategy: str,
        outcome: dict[str, Any],
        lessons_learned: str = ""
    ) -> str:
        """Store a new campaign in memory.
        
        Args:
            problem: Problem description
            context: Business context
            diagnosis: Root cause diagnosis
            strategy: Strategy chosen
            outcome: Outcome dict with success, result, confidence
            lessons_learned: Key lessons from this campaign
            
        Returns:
            Campaign ID
        """
        campaign_id = str(uuid4())[:8]
        problem_type = extract_problem_type(problem)
        timestamp = datetime.now().isoformat()
        
        campaign = {
            "campaign_id": campaign_id,
            "timestamp": timestamp,
            "problem_type": problem_type,
            "problem_text": problem[:500],  # Truncate if too long
            "context": {
                "business_type": context.get("business_type", ""),
                "channels": extract_channels(context),
                "budget": context.get("monthly_budget", context.get("available_budget", "")),
                "timeline": context.get("timeline_constraint", context.get("timeline", "")),
            },
            "diagnosis": diagnosis,
            "strategy_chosen": strategy,
            "outcome": outcome,
            "priority": 0.0,  # Will be calculated on retrieval
            "lessons_learned": lessons_learned,
        }
        
        # Save to file
        date_str = timestamp[:10]
        filename = f"{date_str}_{problem_type}_{campaign_id}.json"
        filepath = self.campaigns_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(campaign, f, indent=2)
        
        logger.info(f"[Memory] Stored campaign {campaign_id}")
        return campaign_id


# =============================================================================
# Prompt Formatting
# =============================================================================

def format_memory_for_prompt(campaigns: list[dict]) -> Optional[str]:
    """Format memory for inclusion in agent prompt.
    
    Keeps output concise (< 300 tokens).
    
    Args:
        campaigns: List of relevant campaigns with _priority scores
        
    Returns:
        Formatted string for prompt injection, or None if no campaigns
    """
    if not campaigns:
        return None
    
    sections = []
    for i, c in enumerate(campaigns, 1):
        priority = c.get("_priority", c.get("priority", 0.5))
        
        # Priority label
        if priority >= 0.7:
            priority_label = "🔥 HIGH"
        elif priority >= 0.4:
            priority_label = "⚠️ MEDIUM"
        else:
            priority_label = "❌ LOW"
        
        # Outcome label
        outcome = c.get("outcome", {})
        if outcome.get("success"):
            outcome_label = f"✅ Success - {outcome.get('result', 'Achieved goal')[:60]}"
        else:
            outcome_label = f"❌ Failed - {outcome.get('result', 'Did not achieve goal')[:60]}"
        
        # Build section (keep concise)
        section = f"""
{i}. [{priority_label}] {c.get('problem_type', 'general').upper()}
   Diagnosis: {c.get('diagnosis', 'Unknown')[:80]}
   Solution: {c.get('strategy_chosen', 'Unknown')[:80]}
   Outcome: {outcome_label}
   Lesson: {c.get('lessons_learned', 'N/A')[:100]}
"""
        sections.append(section)
    
    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RELEVANT PAST CAMPAIGNS (use to inform your analysis):
{''.join(sections)}
Note: Focus on HIGH priority campaigns. Be cautious with MEDIUM/LOW priority insights.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
