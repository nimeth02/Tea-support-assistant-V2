"""TeaTool: the base class of every CrewAI tool in this project."""
from crewai.tools import BaseTool
from pydantic import Field

from agent.state.agent_state import TeaChatState


class TeaTool(BaseTool):
    """A CrewAI tool that is built for one chat turn and shares that turn's flow state.

    - `chat_state`: the flow's TeaChatState. Tools read the farmer's user_id / photo path from
      it, so those never appear in the tool's arguments and the LLM can't get them wrong.
    - `expert`: the intent label of the expert agent that owns the tool ("disease", ...).
      Each run notes it in chat_state.experts_used, which the flow turns into the intent.

    Neither field is shown to the LLM: the arguments it sees come from args_schema / _run.
    """

    chat_state: TeaChatState = Field(default_factory=TeaChatState, exclude=True)
    expert: str = Field(default="", exclude=True)

    def note_use(self) -> None:
        """Record that this tool's expert was used in this turn."""
        if self.expert and self.expert not in self.chat_state.experts_used:
            self.chat_state.experts_used.append(self.expert)
