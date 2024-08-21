import asyncio
import logging
import os
from pathlib import Path
from collections import defaultdict
import json
from openai import OpenAI
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments


async def main():
    # Load the .env file
    load_dotenv()

    user_api_key = os.getenv('OPENAI_API_KEY')

    client = OpenAI(api_key=user_api_key)

    audio_file = open("./03_06.mp3", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )

    # Initialize the kernel
    kernel = Kernel()

    # Add Azure OpenAI chat completion
    kernel.add_service(OpenAIChatCompletion(
        ai_model_id="gpt-4",
        api_key=user_api_key,
    ))

    # Set the logging level for  semantic_kernel.kernel to DEBUG.
    logging.basicConfig(
        format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("kernel").setLevel(logging.DEBUG)

    chat_completion: OpenAIChatCompletion = kernel.get_service(type=ChatCompletionClientBase)

    # Enable planning
    execution_settings = OpenAIChatPromptExecutionSettings()
    execution_settings.function_call_behavior = FunctionCallBehavior.EnableFunctions(auto_invoke=True, filters={})

    # Create a history of the conversation
    history = ChatHistory()

    base_prompt = "You are a librarian." + \
                  f"Provide a recommendation to a book based on the following information. {transcription}." + \
                  "Explain your thinking step by step including a list of top books you selected and how you got to your final choice."

    # Add user input to the history
    history.add_user_message(base_prompt)

    # Get the response from the AI
    response_result = (await chat_completion.get_chat_message_contents(
        chat_history=history,
        settings=execution_settings,
        kernel=kernel,
        arguments=KernelArguments(),
    ))[0]

    # Summarize the list
    print("GPT-4 Count", response_result)


# Run the main function
if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
