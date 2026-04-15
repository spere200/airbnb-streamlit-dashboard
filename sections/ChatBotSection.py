import streamlit as st
import anthropic
from data.systemPrompt import SYSTEM_PROMPT as SP

def render():
    client = anthropic.Anthropic()

    SYSTEM_PROMPT = SP

    col1, col2 = st.columns([6, 1])
    with col1:
        st.subheader("Project Assistant")
        st.caption("Ask me anything about this project")
    with col2:
        if st.button("New Chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        opening = (
            "Hello! I'm the Project Assistant for the Broward County Airbnb Listings Dashboard. "
            "I'm here to answer questions about the dashboard's data, analysis, findings, and methodology."
        )
        st.session_state.messages.append({"role": "assistant", "content": opening})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if len(st.session_state.messages) >= 25:
        st.warning("Conversation limit reached. Please start a new conversation.")
    else:
        prompt = st.chat_input("Ask a question about the project...")

        if prompt and prompt.strip():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = "Sorry, something went wrong. Please try again."
                    try:
                        response = client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=2048,
                            system=[{"type": "text",
                                     "text": SYSTEM_PROMPT,
                                     "cache_control": {"type": "ephemeral"}}
                            ],
                            messages=st.session_state.messages
                        )

                        if response.content:
                            reply = response.content[0].text
                        else:
                            reply = "I wasn't able to generate a response. Please try rephrasing your question."
                    except Exception as e:
                        error_str = str(e)
                        if '401' in error_str or 'authentication' in error_str.lower():
                            reply = "Error: Invalid API key."
                        elif '429' in error_str:
                            reply = "Error: Rate limit reached. Please try again in a moment."
                        elif '529' in error_str or 'overloaded' in error_str.lower():
                            reply = "Error: The service is temporarily overloaded. Please try again."
                        else:
                            reply = "Sorry, something went wrong. Please try again."

                    reply = reply.replace('~~', '').replace('$', '&#36;')
                    st.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})