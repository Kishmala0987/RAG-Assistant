import streamlit as st

from src.pipeline import ask, build_hr_agent
from src.logger import get_logger

logger = get_logger(__name__)
st.set_page_config(page_title = "HR Assistant", page_icon="🤖")
st.title("🤖 Your AI HR Assistant")
st.caption("Ask me anything about company HR policites")

@st.cache_resource(show_spinner = "Setting up the assistant (only happens once)... ")
def get_agent():
    return build_hr_agent()

agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about HR Policy...")

if question:
    logger.info("====Streamlit Run: new question received ===")
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(agent, question)
        st.markdown(answer)

    st.session_state.messages.append({"role":"assistant", "content": answer})