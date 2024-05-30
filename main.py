import streamlit as st
from newbot import LawBot

bot = LawBot()

st.title("Law Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"Hi, I am your AI assistant. How can i help you?"}]


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message["content"])

prompt = st.chat_input("Enter your query...")

if prompt:

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({'role':'user',"content":prompt})

    response = bot.query(prompt)
    response = f'''{response}'''

    with st.chat_message("assistant"):
        st.markdown(response,unsafe_allow_html=True)

    st.session_state.messages.append({"role":"assistant","content":response})
