import asyncio
import agentops  # type: ignore
import os
from dotenv import load_dotenv

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig

# Load environment variables from .env file
load_dotenv()

# Initialize agentops with the provided API key
agentops.init(os.getenv("AGENTOPS_API_KEY"), default_tags=["llama-stack-client-agent"], auto_start_session=False)

LLAMA_STACK_HOST = "127.0.0.1"
LLAMA_STACK_PORT = 11437  # Updated to match new agent port (change to 11437)
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "llama3.2")  # Updated to match the new model

async def agent_test():
    # Initialize the LlamaStack client
    client = LlamaStackClient(base_url=f"http://{LLAMA_STACK_HOST}:{LLAMA_STACK_PORT}/v1")  # <-- Added `/v1`

    available_models = [model.identifier for model in client.models.list()]
    if not available_models:
        raise ValueError("No available models")
    else:
        selected_model = available_models[0]
        print(f"Using model: {selected_model}")

    # Configure the agent with the selected model
    agent_config = AgentConfig(
        model=selected_model,
        instructions="You are a helpful assistant.",
        sampling_params={
            "strategy": "greedy",
            "temperature": 1.0,
            "top_p": 0.9,
        },
        tools=[
            {
                "type": "brave_search",
                "engine": "brave",
                "api_key": os.getenv("BRAVE_SEARCH_API_KEY"),
            }
        ],
        tool_choice="auto",
        tool_prompt_format="json",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
    )

    # Create the agent instance
    agent = Agent(client, agent_config)
    user_prompts = [
        "Who won the NBA championship in 2020? Please use tools",
    ]

    session_id = agent.create_session("test-session")

    # Process the user prompts
    for prompt in user_prompts:
        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            session_id=session_id,
        )

        # Log the response
        for log in EventLogger().log(response):
            log.print()

# Start the agent session and run the agent test
agentops.start_session()
asyncio.run(agent_test())
agentops.end_session(end_state="Success")
