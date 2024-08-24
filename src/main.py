import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import (
    PromptTemplate
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Person(BaseModel):
    first_name: str = Field(description="first name")
    last_name: str = Field(description="last name")
    dob: str = Field(description="date of birth")


class PeopleList(BaseModel):
    people: list[Person] = Field(description="A list of people")


# Load the .env file
load_dotenv()

user_api_key = os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(model="gpt-4")
people_data = model.invoke("Generate a list of 10 fake people information. Only return the list. Each person must have first name, last name and date of birth.")

parser = PydanticOutputParser(pydantic_object=PeopleList)

prompt = PromptTemplate(template="Answer the user query. \n {format_instructions} \n {query}",
                        input_variables=["query"],
                        partial_variables={"format_instructions": parser.get_format_instructions()})

parser_input = prompt.format_prompt(query=people_data)
model = ChatOpenAI()
output = model.invoke(parser_input.to_string())

people_data_parsed = parser.parse(output.content)
print(people_data_parsed)
