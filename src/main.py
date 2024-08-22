from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts.chat import (
    PromptTemplate
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from typing import Optional


# This logic is to develop an information extractor for furniture store

# First we create our Pydantic schema
class Furniture(BaseModel):
    type: Optional[str] = Field(description="the type of furniture")
    style: Optional[str] = Field(description="the style of furniture")
    color: Optional[str] = Field(description="color")


load_dotenv()
user_api_key = os.getenv('OPENAI_API_KEY')

furniture_request = "I'd like a blue mid century chair"

parser = PydanticOutputParser(pydantic_object=Furniture)

prompt = PromptTemplate(template="Answer the user query.\n{format_instructions}\n{query}\n",
                        input_variables=["query"],
                        partial_variables={"format_instructions": parser.get_format_instructions()})

_input = prompt.format_prompt(query=furniture_request)
model = ChatOpenAI()
output = model.invoke(_input.to_string())
parsed = parser.parse(output.content)
print(f"color : {parsed.color}")
print(f"style : {parsed.style}")
print(f"type : {parsed.type}")
