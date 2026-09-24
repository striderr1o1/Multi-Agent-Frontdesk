import uuid
from agents.agent_config import get_kb_agent, get_booking_agent, get_orchestrator_client
from services.supabase_db_functions import save_customer_chat
import json
from agents.agent import agentic_workflow
from agents.graph import setup_graph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import os
from dotenv import load_dotenv
load_dotenv()

async def run_inference_with_stream(query: str, user: dict, supabase_client, customer_client_side_id=None, admin = False): #should get thread-id from links table
    # customer_client_side_id is only set on the published /c/ route - the dashboard
    # owner querying their own agent has no customers_data row, so there is nowhere
    # to save the pair and the save is skipped rather than raising "client id invalid"
    user_id = user["id"]
    business_id = user_id
    client = get_orchestrator_client()
    kb_agent = get_kb_agent(user_id, supabase_client)
    booking_agent = get_booking_agent(user_id, supabase_client)
    agent = agentic_workflow(llm_client=client, kb_agent=kb_agent, bk_agent=booking_agent, setup_graph=setup_graph)
    graph_builder = agent.get_graph()
    DB_URI = os.getenv("DATABASE_URL")
    # thread_id = customer_id + ":" + link_id
    # otherwise, if business owner running from the dashboard, thread_id = business_id
    if admin == True:
        thread_id = business_id

    thread_id = f"{customer_client_side_id}:{business_id}"
    
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        graph = graph_builder.compile(checkpointer=checkpointer)

        response_to_user = ""
        async for chunk in graph.astream(
            {
                "messages": [{"role": "user", "content": query}],
                "tool_calls": [],
                "knowledge_base_agent_output": "",
                "booking_agent_output": "",
                "return_to_user_decision": False,
                "response_to_user": "",
                "count": 0
            },
        {"configurable": {"thread_id": thread_id}},
            stream_mode="updates"
        ):
            for node_name, update in chunk.items():
                print("update: ", update)
                print(node_name, "\n")
                if node_name=="orchestrator":
                    yield f"data: {json.dumps({'event': 'agent calls', 'node': node_name, 'data': update['tool_calls'] if update['tool_calls'] else []})}\n\n"
                if node_name=="knowledge_base_agent":
                    yield f"data: {json.dumps({'event': 'knowledge base agent', 'data': update['knowledge_base_agent_output']})}\n\n"
                if node_name=="booking_agent":
                    yield f"data: {json.dumps({'event': 'booking agent', 'data': update['booking_agent_output']})}\n\n"
                if update.get("return_to_user_decision") == True and update.get("response_to_user"):
                    yield f"data: {json.dumps({'event': 'final response', 'data': update['response_to_user']})}\n\n"
                    # taken here rather than off the last update of the loop: the
                    # final chunk can come from a sub-agent node, whose state delta
                    # carries no response_to_user at all
                    response_to_user = update["response_to_user"]

        # nothing was returned to the user (every brake failed, or the graph errored
        # before the orchestrator's own except branch), so there is no pair to store
        if customer_client_side_id and response_to_user:
            user_ai_chat = {
                "user": query,
                "AI": response_to_user
                    }
            # the response has already been streamed by this point, so a failure here
            # can't become an HTTP error - raising would just truncate the SSE stream
            # on a request the customer has otherwise had answered correctly
            try:
                await save_customer_chat(supabase_client, user["id"], customer_client_side_id, user_ai_chat)
            except Exception as e:
                print(f"Could not save customer chat for {customer_client_side_id}: {e}")
