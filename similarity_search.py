from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import hashlib
import chromadb

app = FastAPI()

embedding_model = SentenceTransformer("paraphrase-albert-small-v2")

def generate_id(text):
    return hashlib.md5(text.encode()).hexdigest()

#chroma db setup
def init_chroma():
    client = chromadb.Client()
    collection_occasion = client.get_or_create_collection(
        name= 'occasion',
        metadata = {"hnsw:space" : "cosine"}
    )
    collection_country = client.get_or_create_collection(
        name= 'country',
        metadata = {"hnsw:space" : "cosine"}
    )
    return collection_occasion, collection_country

def add_document(collection_occasion, collection_country):
    doc_occasion = ['casual_outing', 'picnic', 'graduation', 'beach_party', 'wedding', 'formal_dinner', 'business_meeting', 'religious_event', 'job_interview', 'nightclub', 'cultural_festival']
    doc_country = ['nigeria', 'france', 'uk', 'uae', 'usa', 'brazil', 'japan', 'germany', 'saudi_arabia', 'canada', 'australia', 'india', 'south_africa', 'china', 'mexico']

    embeddings_occasion = embedding_model.encode(doc_occasion).tolist()
# 
    ids_occasion = [generate_id(doc) for doc in doc_occasion]

    collection_occasion.add(
        ids = ids_occasion,
        documents = doc_occasion,
        embeddings = embeddings_occasion,
        metadatas = None
    )
    embeddings_country = embedding_model.encode(doc_country).tolist()

    ids_country = [generate_id(doc) for doc in doc_country]

    collection_country.add(
        ids = ids_country,
        documents = doc_country,
        embeddings = embeddings_country,
        metadatas = None
    )
    return ids_occasion, ids_country

def search_similar(collection, query, n_results=1):
    query_embeddings = embedding_model.encode([query]).tolist()
    result = collection.query(
        query_embeddings = query_embeddings,
        n_results = n_results
    )
    return result

#init chromadb
collection_occasion, collection_country = init_chroma()
ids_occasion, ids_country = add_document(collection_occasion, collection_country)

class SimilarityRequest(BaseModel):
    occasion: str
    country: str

@app.post('/similar')
def get_similar(request: SimilarityRequest):
    occasion_similar = search_similar(collection_occasion, query=request.occasion)
    country_similar = search_similar(collection_country, query=request.country)
    return {
        "occasion_similar": occasion_similar, 
        "country_similar" : country_similar 
        }