
import langchain

from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # To load the env variable into this session

#Creating object for the OpenAI using the model gpt-3.5-turbo-instruct
llm_model = OpenAI(model='gpt-3.5-turbo-instruct')

#Invoking the LLM for the response with user prompt
llm_response = llm_model.invoke('What is the capital of Andhra Pradesh State?')

print(llm_response)