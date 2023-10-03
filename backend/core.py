import os
from typing import Any

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.vectorstores import FAISS
# from langchain.vectorstores import Pinecone
# import pinecone

from consts import INDEX_NAME

# pinecone.init(
#     api_key="4e4f1dd4-c83d-4f00-b0f3-6e0e9e14592d",
#     environment="us-west1-gcp-free",
# )


def run_llm(query: str, chat_histroy) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.load_local("tabulera_index",embeddings)
    # docsearch = Pinecone.from_existing_index(
    #     index_name=INDEX_NAME, embedding=embeddings
    # )

    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    # )

    return qa({"query": query})


if __name__ == "__main__":
    print(run_llm(query="Types of benefit Plans ?"))


"""
rumali roti is best they said 
koftha and panner they use same gravy so they said chenna is better 



"""
