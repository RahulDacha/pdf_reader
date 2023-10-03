from backend.core import run_llm
import streamlit as st
from streamlit_chat import message
from PIL import Image 


image = Image.open("/Users/rahuldacha/PersonalProjcets/chat-bot-langchain/documentation-helper/tabulera-dark.png")
new_image = image.resize((150,25))
st.image(new_image)
st.divider()


st.text("\n\n")




promt = st.chat_input("Enter your question here ....")

if "user_promt_histroy" not in st.session_state:
    st.session_state["user_promt_histroy"] = []
if "chat_answers_histroy" not in st.session_state:
    st.session_state["chat_answers_histroy"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if promt:
   
    with st.spinner("Generating Response"):
        generated_response = run_llm(
            query=promt, chat_histroy=st.session_state["chat_history"]
        )
        # sources = set(
        #     [doc.metadata["source"] for doc in generated_response["source_documents"]]
        #     )

        formatted_response = generated_response["result"]

        st.session_state["user_promt_histroy"].append(promt)

        st.session_state["chat_answers_histroy"].append(formatted_response)
        st.session_state["chat_history"].append((promt, formatted_response))

if st.session_state["chat_answers_histroy"]:
    for g_r, u_q in zip(
        st.session_state["chat_answers_histroy"], st.session_state["user_promt_histroy"]
    ):
        message(u_q, is_user=True)
        message(g_r)