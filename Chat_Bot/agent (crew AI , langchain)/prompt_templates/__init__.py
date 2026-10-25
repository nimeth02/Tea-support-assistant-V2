"""All LLM prompts, one module per agent.

  orchestrator_prompt  LangChain supervisor (system prompt)
  disease_prompt       CrewAI disease expert (agent backstory)
  fertilize_prompt     LangChain fertilizer expert (system prompt)
  help_prompt          LangGraph help expert (answer step template)
"""
from agent.prompt_templates.disease_prompt import disease_agent_prompt
from agent.prompt_templates.fertilize_prompt import fertilize_agent_prompt
from agent.prompt_templates.help_prompt import help_prompt
from agent.prompt_templates.maintenance_prompt import maintenance_prompt
from agent.prompt_templates.orchestrator_prompt import orchestrator_system_prompt

__all__ = [
    "disease_agent_prompt",
    "fertilize_agent_prompt",
    "help_prompt",
    "maintenance_prompt",
    "orchestrator_system_prompt",
]
