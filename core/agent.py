# agent.py
from typing import Callable, Any

from langchain.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from core.tools import PythonExecTool

class StopRequested(Exception):
    """Raised to abort agent execution when stop_flag becomes True."""
    pass

class StopFlagCallback(BaseCallbackHandler):
    def __init__(self, stop_flag: Callable[[], bool]):
        self.stop_flag = stop_flag

    def on_agent_action(self, action: Any, **kwargs):
        if self.stop_flag and self.stop_flag():
            raise StopRequested()

def make_agent(df, sim_code: str, params: list, schema: list, backend: str = "local"):
    # 1) system prompt template
    system_template = SystemMessagePromptTemplate.from_template(
        """
        You are a scientific reasoning assistant. You cannot see `df` directly.
        You must call the `python_exec` tool to operate on it.

        ─── SIMULATION CODE ───
        ```python
        {sim_code}
        ```
        ─── SCHEMA ───
        Columns: {schema}
        Params: {params}
        """
    )
    human_template = HumanMessagePromptTemplate.from_template("{question}")

    if backend.lower() == "local":
        # Make sure you have `ollama` server running locally (default port 11434)
        llm = Ollama(
            model="deepseek-r1:14b",
            base_url="http://localhost:11434",
            temperature=0.0,
        )
    else:
        llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.0)

    # 3) wrap python_exec
    python_tool = PythonExecTool(df)

    # 4) initialize a “react”‑style agent
    agent = initialize_agent(
        tools=[python_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        system_message=system_template,
        human_message=human_template,
    )
    return agent
