from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

documents = [
    'ML is the algorithm where machines learns patters from data',
    'DL is the susbset of ML, where deep neural layers present to learn patters from data',
    'AI is the field where human intelligence is being replaced with machine intelligence'
]

#Creating object for the model OpenAIEmbeddings using the embedding model text-embedding-3-large
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

#Generating embedding for the user query
embedding_response = embedding_model.embed_documents(documents)

print(f"Embedding results of documents as string are : {str(embedding_response)}")