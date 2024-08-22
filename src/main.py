from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv


class CommaSeparatedOutputParser(BaseOutputParser):

    @property
    def _type(self) -> str:
        pass

    def get_format_instructions(self) -> str:
        pass

    def parse(self, text: str):
        return text.strip().split(", ")


# Load the .env file
load_dotenv()

user_api_key = os.getenv('OPENAI_API_KEY')

chat_model = ChatOpenAI(openai_api_key=user_api_key)
print(chat_model.invoke("hi"))

system_template = "You are a helpful assistant who generates comma separated lists. " \
                  "A user will pass in a category, and you should generate 5 objects in that category in a comma seperated list. " \
                  "ONLY return a comma seperated list, and nothing more"
human_template = "{text}"

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# chain = LLMChain(llm=ChatOpenAI(),
#                  prompt=chat_prompt,
#                  output_parser=None)


chain = chat_prompt | ChatOpenAI() | CommaSeparatedOutputParser()

# blue, green, yellow, red, purple
print(chain.invoke({"text": "colors"}))
