import streamlit as st
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from streamlit_chat import message
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.output_parsers import StructuredOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


def chat_friendly():
    st.set_page_config(page_title='ChatBot', page_icon="🤖")
    st.title('Personal Chat Friendly AI 💬')

    # ✅ FIXED: removed space in env key
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.8
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful AI assistant.")
        ]

    # User input box
    user_input = st.chat_input('Enter your message here...')
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner('Thinking...'):
            # ✅ FIXED: use invoke() instead of direct call
            response = model.invoke(st.session_state.messages)

            # Append AI response
            st.session_state.messages.append(AIMessage(content=response.content))

    # Display chat messages
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):  # Skip system message
        if isinstance(msg, HumanMessage):
            message(msg.content, is_user=True, key=f"user_{i}")
        elif isinstance(msg, AIMessage):
            message(msg.content, is_user=False, key=f"ai_{i}")


if __name__ == "__main__":
    chat_friendly()
