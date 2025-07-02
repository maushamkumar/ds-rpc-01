
import streamlit as st
import requests

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000"

# --- Helper Functions ---
def login(username, password):
    try:
        response = requests.post(
            f"{BACKEND_URL}/token",
            data={"username": username, "password": password}
        )
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.HTTPError as e:
        st.error(f"Login failed: {e.response.json().get('detail', 'Unknown error')}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the backend: {e}")
        return None

def get_chat_response(token, query):
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            headers=headers,
            json={"query": query}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"Chat request failed: {e.response.json().get('detail', 'Unknown error')}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the backend: {e}")
        return None

# --- Streamlit App ---
st.set_page_config(page_title="Internal Chatbot", layout="wide")

# --- State Management ---
if 'token' not in st.session_state:
    st.session_state.token = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Main App Logic ---
if st.session_state.token is None:
    # --- Login Page ---
    st.title("Login to the Internal Chatbot")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            token = login(username, password)
            if token:
                st.session_state.token = token
                st.rerun()
else:
    # --- Chat Page ---
    st.title("Internal RAG Chatbot")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "context" in message:
                with st.expander("Show Retrieval Context"):
                    for doc in message["context"]:
                        st.markdown(f"**Source:** {doc['metadata']['source']}")
                        st.markdown(doc['content'])
                        st.markdown("---")

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.spinner("Thinking..."):
            chat_response = get_chat_response(st.session_state.token, prompt)

        if chat_response:
            response_text = chat_response["response"]
            sources = chat_response["sources"]
            context = chat_response["context"]

            # Add bot response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": sources,
                "context": context
            })
            with st.chat_message("assistant"):
                st.markdown(response_text)
                with st.expander("Show Retrieval Context"):
                    for doc in context:
                        st.markdown(f"**Source:** {doc['metadata']['source']}")
                        st.markdown(doc['content'])
                        st.markdown("---")

    # Logout button
    if st.button("Logout"):
        st.session_state.token = None
        st.session_state.messages = []
        st.rerun()
