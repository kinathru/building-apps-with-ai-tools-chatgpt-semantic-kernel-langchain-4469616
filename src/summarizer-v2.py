import asyncio
import logging
import os
from pathlib import Path
from collections import defaultdict
import json

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments


def _unidiff_output(expected, actual):
    """
    Helper function. Returns a string containing the unified diff of two multiline strings.
    """

    import difflib
    expected = expected.splitlines(1)
    actual = actual.splitlines(1)

    diff = difflib.unified_diff(expected, actual)

    return ''.join(diff)


async def main():
    user_api_key = os.getenv('OPENAI_API_KEY')

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

    # Add a plugin (the LightsPlugin class is defined below)
    # kernel.add_plugin(
    #     LightsPlugin(),
    #     plugin_name="Lights",
    # )

    chat_completion: OpenAIChatCompletion = kernel.get_service(type=ChatCompletionClientBase)

    # Enable planning
    execution_settings = OpenAIChatPromptExecutionSettings()
    execution_settings.function_call_behavior = FunctionCallBehavior.EnableFunctions(auto_invoke=True, filters={})

    # Create a history of the conversation
    history = ChatHistory()

    with open("order.txt") as f:
        order = f.read()
        f.seek(0)
        order_lines = f.readlines()
    item_count = defaultdict(int)

    # Loop over each line in the file
    for line in order_lines:
        # Split the line into quantity and item
        quantity, item = line.strip().split(" x ")

        # Update the count in the dictionary
        item_count[item] += int(quantity)

    # Initiate a back-and-forth chat
    user_input = order

    prompt_examples = """"List:
    1 x apple
    2 x oranges
    3 x fishes
    1 x apple
    3 x duck
    Output:{
        "apple": 2,
        "orange": 2,
        "fish": 2, 
        "duck": 3
    }
    
    List:"""

    prompt_prefix = """You are an order counting assistant. Summarize the list into product name and total quantity into a JSON document. Only output the JSON, do not give an explanation.\n"""
    prompt = f"{prompt_prefix}{prompt_examples}" + f"{user_input}" + "Output:\n"

    # Add user input to the history
    history.add_user_message(prompt)

    # Get the response from the AI
    summary_result = (await chat_completion.get_chat_message_contents(
        chat_history=history,
        settings=execution_settings,
        kernel=kernel,
        arguments=KernelArguments(),
    ))[0]

    # Summarize the list
    print("GPT-4 Count", summary_result)

    # Add the message from the agent to the chat history
    history.add_message(summary_result)

    formatted_actual = json.dumps(item_count, indent=4)
    print("Actual Count:")
    print(formatted_actual)

    print(_unidiff_output(summary_result.content, formatted_actual))

    # we'll use this data in some downstream process
    assert summary_result.content == formatted_actual


# Run the main function
if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
