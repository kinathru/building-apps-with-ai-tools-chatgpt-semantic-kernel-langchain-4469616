import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser


def generate_travel_recommendations(in_travel_requests):
    """
    Generate travel recommendations based on user requests
    """
    # create templates
    system_template_travel_agent = """You are travel recommendation agent. Provide a short recommendation based on the user request."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template_travel_agent)

    human_template_travel_agent = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template_travel_agent)

    # create full prompt
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])

    chain = chat_prompt | ChatOpenAI(temperature=1)

    gen_recommendations = []
    for travel_request in in_travel_requests:
        gen_recommendations.append(chain.invoke(travel_request))

    return gen_recommendations


def generate_travel_requests(n=5) -> list[str]:
    """ Generate travel requests
    n: number of requests
    """
    system_template_travel_agent = """Generate one utterance for how someone would travel for a {text}"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template_travel_agent)

    # create full prompt
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt])

    chain = chat_prompt | ChatOpenAI(model='gpt-4')

    results = []
    for _ in range(0, n):
        results.append(chain.invoke("beach vacation"))

    return results


# Load the .env file
load_dotenv()

user_api_key = os.getenv('OPENAI_API_KEY')
# generate some requests
travel_requests = generate_travel_requests()
# travel_requests = ["I want a beach vacation"]
print("Travel Requests : \n")
print(travel_requests)
print("\n")

# get the recommendations
recommendations = generate_travel_recommendations(travel_requests)
print("Travel Recommendations : \n")
print(recommendations)
