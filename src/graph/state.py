from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    query: str
    messages: List[BaseMessage]
    response: str
    chart_path: Union[str, None]
    debug_log: str
