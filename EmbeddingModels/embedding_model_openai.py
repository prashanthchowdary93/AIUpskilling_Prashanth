from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#Creating object for the model OpenAIEmbeddings using the embedding model text-embedding-3-large
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

#Generating embedding for the user query
embedding_response = embedding_model.embed_query('Amaravati is the capital of Andhra Pradesh')

print(f"Embedding results are : {embedding_response}")
print(f"EMbedding results as string are : {str(embedding_response)}")