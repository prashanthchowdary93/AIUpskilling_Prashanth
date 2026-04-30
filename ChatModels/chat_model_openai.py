from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # To load the env variable into this session

#Creating object for the ChatOpenAI model using model gpt-4
chat_model_openai = ChatOpenAI(model='gpt-4')

#Invoking the LLM for the response with user prompt
chat_model_response = chat_model_openai.invoke('When was the  united Andhra Pradesh divided into 2 states?')

print(f"Chat Model Entire  response is {chat_model_response}")
print(f"Chat Model fetched only response part is {chat_model_response.content}")
