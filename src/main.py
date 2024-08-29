import os
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.evaluation import QAEvalChain
from dotenv import load_dotenv


def generate_book_recommendations(book_requests):
    """
    Generate book recommendations based on user requests
    """
    # create templates
    system_template_book_agent = """You are book recommendation agent. Provide a short recommendation based on the user request."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template_book_agent)

    human_template_book_agent = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template_book_agent)

    # create full prompt
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])

    chain = chat_prompt | ChatOpenAI(temperature=1)

    recommendations = []
    for book_request in book_requests:
        invoke_res = chain.invoke(book_request)
        recommendations.append(invoke_res.content)

    return recommendations


def generate_book_requests(n=5) -> list[str]:
    """ Generate book requests
    n: number of requests
    """
    # create templates
    system_template_book_agent = """Generate one utterance for how someone may ask for a {text}. Include a genre and a year. """
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template_book_agent)

    # create full prompt
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt])

    chain = chat_prompt | ChatOpenAI(model='gpt-4')

    results = []

    for _ in range(0, n):
        invoke_result = chain.invoke("book")
        results.append(invoke_result.content)

    return results


# Load the .env file
load_dotenv()

user_api_key = os.getenv('OPENAI_API_KEY')

# generate some requests
book_requests = generate_book_requests()
print("The book requests are:\n")
print("\n".join(book_requests))
print("\n")
# get the recommendations
print("The book recommendations are:\n")
recommendations = generate_book_recommendations(book_requests)
print("\n\n".join(recommendations))
print("\n")

llm = ChatOpenAI(model="gpt-4")
predictions = []
question_answers = []

for i in range(0, len(recommendations)):
    q = book_requests[i]
    a = recommendations[i]
    question_answers.append({"question": q, "answer": a})
    response = llm.invoke(
        f"Generate the response to the question: {q}. Only print the answer.")
    predictions.append({"result": {response.content}})

print("\nGenerating Self eval:")

# Start your eval chain
eval_chain = QAEvalChain.from_llm(llm)

# compare the results of two prompts against themselves
graded_outputs = eval_chain.evaluate(question_answers,
                                     predictions,
                                     question_key="question",
                                     prediction_key="result",
                                     answer_key='answer')
print(graded_outputs)
