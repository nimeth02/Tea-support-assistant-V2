"""Tools available to the MaintenanceAgent, one module per tool."""
from agent.agents.maintenance.tools.irrigation_planner import irrigation_planner_tool
from agent.agents.maintenance.tools.pruning_guide import pruning_guide_tool

MAINTENANCE_TOOLS = [pruning_guide_tool, irrigation_planner_tool]

__all__ = ["MAINTENANCE_TOOLS", "pruning_guide_tool", "irrigation_planner_tool"]
