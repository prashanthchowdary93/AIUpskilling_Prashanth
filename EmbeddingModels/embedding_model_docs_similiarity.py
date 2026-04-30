from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#Loading the environment variables for API Secret Keys from env file
load_dotenv()

#Creating object for the OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=300)

#Multiple statements as a document
documents = [
    'Machine Learning is the algorithm where machines learns patters from data',
    'Deep Learning is the susbset of ML, where deep neural layers present to learn patters from data',
    'Artifical Intelligence is the field where human intelligence is being replaced with machine intelligence'
]

#Input Prompt
query = 'What is Intelligence'

#Create embeddings for documents in dimension of 300
doc_embeddings = embedding_model.embed_documents(documents)

#Create emebedding for the query
query_embedding = embedding_model.embed_query(query)

print(f"Documents embeddings are : {doc_embeddings}")
print(f"Query Embeddings are     : {query_embedding}")

#Call cosine similiarty by passing above 2 embeddings in 2D format for comparing query with documents
#output will be 2D, but here we need 1D only therefore appending [0]
scores = cosine_similarity([query_embedding],doc_embeddings)[0]
print(scores)

#Now applying enumerate function to have index for each score
enumerate_scores = enumerate(scores)
print(f"enumerate_scores are : {enumerate_scores}")
#sorting the enumearate_scores in ascending order based on cosine score

sorted_order = sorted(list(enumerate_scores),key=lambda x:x[1])

print(f"Final sorted order is : {sorted_order}")

#final result of our prompt is th eone having highest cosine score

index , similiairty_score = (sorted_order)[-1]

final_response = documents[index]

print(f"User Prompt is       : {query}")
print(f"Final response is    : {final_response}")
print(f"Similiarity score is : {similiairty_score}")


