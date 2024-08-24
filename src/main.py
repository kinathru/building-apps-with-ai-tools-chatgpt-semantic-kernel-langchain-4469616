import os

import numpy as np
from dotenv import load_dotenv
from langchain.evaluation import QAEvalChain
from langchain_openai import ChatOpenAI
from openai import OpenAI

# Load the .env file
load_dotenv()

user_api_key = os.getenv('OPENAI_API_KEY')

question_answers = [
    {'question': "When was tea discovered?",
        'answer': "3rd century"},
    {'question': "I'd like a 1 line ice cream slogan",
        'answer': "It's the coolest thing around!"}
]
llm = ChatOpenAI(model="gpt-4")
predictions = []
responses = []
for pairs in question_answers:
    q = pairs["question"]
    response = llm.invoke(f"Generate the response to the question: {q}. Only print the answer.")
    responses.append(response.content)
    predictions.append({"result": {response.content}})

print("\nGenerating text matches:")

for i in range(0, len(responses)):
    print(question_answers[i]["answer"] == responses[i])

client = OpenAI()
resp = client.embeddings.create(
    input=[r["answer"] for r in question_answers] + responses,
    model="text-embedding-ada-002")

print("\nGenerating Similarity Score:!")
for i in range(0, len(question_answers)*2, 2):
    embedding_a = resp.data[i].embedding
    embedding_b = resp.data[len(question_answers)].embedding
    similarity_score = np.dot(embedding_a, embedding_b)
    print(similarity_score, similarity_score > 0.8)


print("\nGenerating Self eval:")

# Start your eval chain
eval_chain = QAEvalChain.from_llm(llm)

# Have it grade itself. The code below helps the eval_chain know where the different parts are
graded_outputs = eval_chain.evaluate(question_answers,
                                     predictions,
                                     question_key="question",
                                     prediction_key="result",
                                     answer_key='answer')
print(graded_outputs)
