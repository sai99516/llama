import os
from dotenv import load_dotenv
import agentops  # type: ignore
import asyncio
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import UserMessage
from llama_stack_client.lib.inference.event_logger import EventLogger

# Load environment variables from .env file
load_dotenv()

# Initialize agentops with the provided API key
agentops.init(os.getenv("AGENTOPS_API_KEY"), default_tags=["llama-stack-client-inference"], auto_start_session=False)

LLAMA_STACK_HOST = "127.0.0.1"
LLAMA_STACK_PORT = 11436  # Updated to match the interface port (11436)
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "llama3.2")  # Use model from .env

# Build the host URL for the API
full_host = f"http://{LLAMA_STACK_HOST}:{LLAMA_STACK_PORT}/v1"  # <-- Added `/v1`

# Initialize the LlamaStack client
client = LlamaStackClient(base_url=full_host)

async def stream_test():
    try:
        # Send a test message to the inference API
        response = await client.inference.chat_completion(
            messages=[UserMessage(content="Write a 3 word poem about the moon", role="user")],
            model_id=INFERENCE_MODEL,  # Use model from .env
            stream=True,
        )

        # Log and print the inference response
        async for log in EventLogger().log(response):
            log.print()
    except Exception as e:
        print(f"Error during inference: {e}")

def main():
    # Start the session for tracking
    agentops.start_session()

    # Run the stream test
    asyncio.run(stream_test())

    # End the session once done
    agentops.end_session(end_state="Success")

# Run the main function
main()
