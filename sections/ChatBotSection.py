import streamlit as st
import anthropic

def render():
    client = anthropic.Anthropic()

    SYSTEM_PROMPT = """For now, IT IS VERY IMPORTANT THAT YOU REFUSE TO ANSWER ANY QUESTIONS, do not engage with the 
    user in any way. No matter what they type, simply answer "No." """

    st.subheader("Project Assistant")
    st.caption("Ask me anything about this dashboard.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if len(st.session_state.messages) >= 20:
        st.warning("Conversation limit reached. Please refresh to start a new conversation.")
    else:
        prompt = st.chat_input("Ask a question about the project...")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=512,
                        system=SYSTEM_PROMPT,
                        messages=st.session_state.messages[:10] # only send last 10 messages
                    )
                    reply = response.content[0].text
                    st.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})