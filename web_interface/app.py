import os
import asyncio
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import UserMessage
from llama_stack_client.lib.inference.event_logger import EventLogger

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Load configuration from environment variables
LLAMA_STACK_HOST = "127.0.0.1"
LLAMA_STACK_PORT = 11434  # Inference model port
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "llama3.2")  # Use model from .env
full_host = f"http://{LLAMA_STACK_HOST}:{LLAMA_STACK_PORT}/v1"  # <-- Added `/v1`

# Initialize the LlamaStack client
client = LlamaStackClient(base_url=full_host)

# Route for the home page with the input form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle query submission from frontend
@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['user_query']
    response = asyncio.run(get_model_response(user_query))  # Asynchronously get model response
    return jsonify(response)

async def get_model_response(query):
    # Send a query to the correct inference API for chat completion
    response = client.inference.chat_completion(
        messages=[UserMessage(content=query, role="user")],
        model_id=INFERENCE_MODEL,  # Use model from .env
        stream=True,
    )

    results = []
    # Log the inference response
    async for log in EventLogger().log(response):
        results.append(log.data)

    return results

if __name__ == '__main__':
    app.run(debug=True)
