from langchain_community.embeddings import OllamaEmbeddings

def get_embedding_func():
    embeddings = OllamaEmbeddings()
    return embeddings
