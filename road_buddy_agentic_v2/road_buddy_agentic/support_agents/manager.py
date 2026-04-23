from __future__ import annotations
from road_buddy_agentic.support_agents.providers.gemini_support_agent import GeminiSupportAgent
from road_buddy_agentic.support_agents.providers.llama_support_agent import LlamaSupportAgent
from road_buddy_agentic.support_agents.providers.noop_support_agent import NoopSupportAgent


def build_support_agent(provider: str, model_name: str, **kwargs):
    p = (provider or "none").strip().lower()
    if p == "none":
        return NoopSupportAgent(model_name=model_name or "none", **kwargs)
    if p == "gemini":
        return GeminiSupportAgent(model_name=model_name, **kwargs)
    if p == "groq":
        return LlamaSupportAgent(model_name=model_name, **kwargs)
    raise ValueError(f"Unsupported support provider: {provider}")
