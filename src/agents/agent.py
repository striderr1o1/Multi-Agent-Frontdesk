import sys, os 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")) 
from .agent_config import get_chat_completion, get_chat_completion_system_prompt 
from KnowledgeBaseTool.kb_tools import ingest_documents, retrieve_documents 
from typing import Annotated 
from .state import orchestrator_output, graph_state

class agentic_workflow:
    def __init__(self, llm_client, kb_agent, bk_agent, setup_graph):
        self.kb_agent = kb_agent
        self.bk_agent = bk_agent
        self.llm_client = llm_client
        self.available_tools = [self.knowledge_base_agent, self.booking_agent]
        self.graph = setup_graph(self.orchestrator, self.knowledge_base_agent, self.booking_agent, self.tool_call_node)
        return
    
    def get_graph(self):
        return self.graph

    async def orchestrator(self, state: graph_state) -> graph_state:
        try:
            system_promptt = get_chat_completion_system_prompt(self.available_tools)
            response = await get_chat_completion(llm_client=self.llm_client, state=state, model="openai/gpt-oss-120b", response_model=orchestrator_output, system_prompt = system_promptt)
            json_response = response.model_dump()
            count = state["count"] + 1
            return {
                "messages": [{"role": "assistant", "content": f"""Reasoning: {json_response['reasoning']}...
            _                         ..agent/tool calls: {json_response['tool_calls']}...return to user decision: {json_response["return_to_user"]}"""}],
                "tool_calls": json_response["tool_calls"],
                "return_to_user_decision": json_response["return_to_user"] or count > 3,
                "response_to_user": json_response["summary_of_agents_response"],
                "count": count
            }
        except Exception as e:
            return {
                "return_to_user_decision": True,
                "response_to_user": f"An Unexpected Error Occured: {e}"
            }

    def tool_call_node(self, state: graph_state):
        try:
            if state["return_to_user_decision"] == True:
                return "end"
            if len(state["tool_calls"]) != 0:
                toolcall = state["tool_calls"][0]
                if toolcall["tool"] == "knowledge_base_agent":
                    return "knowledge_base_agent"
                if toolcall["tool"] == "booking_agent":
                    return "booking_agent"
            return "orchestrator"
        except Exception as e:
            state["return_to_user_decision"] = True
            state["response_to_user"] = f"An Unexpected Error Occured: {e}"
            return "end"
   
    async def knowledge_base_agent(self, state: graph_state)-> graph_state:
        query =""
        tool_calls = list(state["tool_calls"])
        toolcall = tool_calls[0]
        if toolcall["tool"] == "knowledge_base_agent":
            query = toolcall["argument"][0]
            tool_calls.remove(toolcall)
        response = await self.kb_agent.ainvoke({"messages": [{"role": "user", "content": f"Hi, heres your task from orchestrator: {query}"}]})
        summary = response["messages"][-1].content
        return {"knowledge_base_agent_output": summary, "tool_calls": tool_calls}

    async def booking_agent(self, state: graph_state)-> graph_state:
        query =""
        tool_calls = list(state["tool_calls"])
        if len(tool_calls) != 0:
            toolcall = tool_calls[0]
            if toolcall["tool"] == "booking_agent":
                query = toolcall["argument"][0]
                tool_calls.remove(toolcall)
        response = await self.bk_agent.ainvoke({"messages": [{"role": "user", "content": f"Hi, heres your task from orchestrator: {query}"}]})
        return {"booking_agent_output": response["messages"][-1].content, "tool_calls": tool_calls}

