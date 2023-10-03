import os
from langchain.document_loaders import ReadTheDocsLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone
# import pinecone
from langchain.vectorstores import FAISS

from consts import INDEX_NAME

# pinecone.init(
#     api_key="4e4f1dd4-c83d-4f00-b0f3-6e0e9e14592d",
#     environment="us-west1-gcp-free",
# )


def ingest_docs() -> None:
    loader = PyPDFDirectoryLoader(
        path="/Users/rahuldacha/PersonalProjcets/chat-bot-langchain/documentation-helper/docs"
    )
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents) }documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    # for doc in documents:
    #     old_path = doc.metadata["source"]
    #     new_url = old_path.replace("langchain-docs", "https:/")
    #     doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    # Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    vectorstore = FAISS.from_documents(documents,embeddings)
    vectorstore.save_local("tabulera_index")
    print("****** Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()
