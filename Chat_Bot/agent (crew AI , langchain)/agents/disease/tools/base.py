"""TeaTool: base class of the disease expert's CrewAI tools."""
from crewai.tools import BaseTool
from pydantic import Field

from agent.state.agent_state import DiseaseTurnState


class TeaTool(BaseTool):
    """A CrewAI tool built for one question that shares that question's DiseaseTurnState.

    Tools read the photo path from `turn` (the LLM never sees or writes it), and note their own
    name in `turn.tools_used`, which the A2A server reports back and counts in its metrics.
    `turn` is excluded from the schema, so it is never shown to the LLM.
    """

    turn: DiseaseTurnState = Field(default_factory=DiseaseTurnState, exclude=True)

    def note_use(self) -> None:
        if self.name not in self.turn.tools_used:
            self.turn.tools_used.append(self.name)
