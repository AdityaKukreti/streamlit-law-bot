import streamlit as st
from bot import LawBot

bot = LawBot()

st.title("Law Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# with "messages" not in st.session_state:
#     st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message["content"])

prompt = st.chat_input("Enter your query...")

if prompt:

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({'role':'user',"content":prompt})

    response = bot.getResponse(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role":"assistant","content":response})