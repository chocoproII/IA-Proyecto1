from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("Arduino - Documentation Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here")

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def create_sources_string(source_urls: set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "Sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt,
            chat_history=st.session_state["chat_history"]
        )

        sources = set([doc.metadata.get("source", "") for doc in generated_response["source"]])

        formatted_response = (
            f"{generated_response['result']}\n\n{create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        reversed(st.session_state["chat_answers_history"]),
        reversed(st.session_state["user_prompt_history"])
    ):
        message(user_query, is_user=True)
        message(generated_response)