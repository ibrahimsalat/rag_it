import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embeddings import get_embedding_func
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_doc()
    chunks = split_doc(documents)
    add_to_chroma(chunks)

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def load_doc():
    doc_loader = PyPDFDirectoryLoader(DATA_PATH)
    return doc_loader.load_and_split()


def split_doc(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_id(chunks):
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get('source')
        page =  chunk.metadata.get('page')
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata['id'] = chunk_id
    return chunk

def add_to_chroma(chunks: list[Document]):
    
    print('hello')
    db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=get_embedding_func()
        )
    print('bye')
    existing_items = db.get(include=[])
    existing_ids = set(existing_items['ids'])
    print(f"number of existing documents in DB: {len(existing_ids)}")

    chunks_with_ids = calculate_chunk_id(chunks)

    new_chunks = []
    for chunk in chunks:
        if chunk.metadata['id'] not in existing_ids: 
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist
    else:
        print("No new documents to add")


if __name__ == "__main__":
    main()
