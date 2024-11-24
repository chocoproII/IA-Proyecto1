from dotenv import load_dotenv
import os
import sys
import io
load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import WebBaseLoader
from const import INDEX_NAME

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    # List of URLs you want to process
    urls = [
        "https://docs.arduino.cc",
        "https://docs.arduino.cc/language-reference/en/functions/digital-io/digitalread/",
        "https://docs.arduino.cc/language-reference/en/functions/digital-io/digitalwrite/",
        "https://docs.arduino.cc/language-reference/en/functions/digital-io/pinMode/",
        "https://docs.arduino.cc/language-reference/en/functions/math/constrain/",
        "https://docs.arduino.cc/language-reference/en/functions/analog-io/analogRead/",
        # Add more URLs as needed
    ]

    loader = WebBaseLoader(urls)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(documents)} chunks")

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embedding=embeddings, index_name=INDEX_NAME
    )

if __name__ == "__main__":
    ingest_docs()