import asyncio
import logging
import os

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments


async def main():
    user_api_key = os.getenv('OPENAI_API_KEY')

    # Initialize the kernel
    kernel = Kernel()

    # Add Azure OpenAI chat completion
    kernel.add_service(OpenAIChatCompletion(
        ai_model_id="gpt-4",
        api_key=user_api_key,
    ))

    chat_completion: OpenAIChatCompletion = kernel.get_service(type=ChatCompletionClientBase)

    # Enable planning
    execution_settings = OpenAIChatPromptExecutionSettings()
    execution_settings.function_call_behavior = FunctionCallBehavior.EnableFunctions(auto_invoke=True, filters={})

    # Create a history of the conversation
    history = ChatHistory()

    # Initiate a back-and-forth chat
    user_input = "There is a country that is surrounded with sandy beaches and coconut groves. Center of the island have highlands with lush green and misty mountain peaks"

    # Add user input to the history
    history.add_user_message(f"{user_input} \n\n Give me the TLDR in exactly 5 words")

    # Get the response from the AI
    result = (await chat_completion.get_chat_message_contents(
        chat_history=history,
        settings=execution_settings,
        kernel=kernel,
        arguments=KernelArguments(),
    ))[0]

    # Print the results
    print("Assistant > " + str(result))

    # Add the message from the agent to the chat history
    history.add_message(result)


# Run the main function
if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
