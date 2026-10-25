"""All LLM prompts used by the agent, one module per service."""
from agent.prompt_templates.disease_prompt import disease_agent_prompt
from agent.prompt_templates.fertilize_prompt import fertilize_agent_prompt
from agent.prompt_templates.help_prompt import help_prompt
from agent.prompt_templates.maintenance_prompt import maintenance_prompt

__all__ = [
    "disease_agent_prompt",
    "fertilize_agent_prompt",
    "help_prompt",
    "maintenance_prompt",
]
