import streamlit as st
import requests
from chatbot_ui.core.config import config
from chatbot_ui.core.constants import OPENAI, GROQ, GOOGLE, MODELS


def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}


if "used_context" not in st.session_state:
    st.session_state.used_context = []

## Lets create a sidebar with a dropdown for the model list and providers
with st.sidebar:
    st.title("Settings")

    # Dropdown for model
    provider = st.selectbox("Provider", [OPENAI, GROQ, GOOGLE])
    if provider == OPENAI:
        model_name = st.selectbox("Model", MODELS[OPENAI])
    elif provider == GROQ:
        model_name = st.selectbox("Model", MODELS[GROQ])
    else:
        model_name = st.selectbox("Model", MODELS[GOOGLE])
    # Save provider and model to session state
    st.session_state.provider = provider
    st.session_state.model_name = model_name

    # Advanced Options section
    st.subheader("Advanced Options")
    enable_reranking = st.checkbox("Enable Re-ranking", value=False)
    top_k = st.slider(
        "Number of Contexts to Retrieve (top_k)", min_value=1, max_value=20, value=5
    )
    # save to session state
    st.session_state.enable_reranking = enable_reranking
    st.session_state.top_k = top_k

    st.divider()

    # Create tabs in the sidebar
    (suggestions_tab,) = st.tabs(["Suggestions"])

    with suggestions_tab:
        if st.session_state.used_context:
            for idx, item in enumerate(st.session_state.used_context):
                st.caption(item.get("description", "No description available"))
                if "image_url" in item:
                    st.image(item["image_url"], width=250)
                st.caption(f"Price: {item.get('price', 'N/A')} USD")
                st.divider()
        else:
            st.info("No suggestions yet.")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        json = {
            "provider": st.session_state.provider,
            "model_name": st.session_state.model_name,
            "query": prompt,
            "extra_options": {
                "top_k": st.session_state.top_k,
                "enable_reranking": st.session_state.enable_reranking,
            },
        }
        status, output = api_call("post", f"{config.API_URL}/rag", json=json)
        answer = output["answer"]
        used_context = output["used_context"]

        # set the used_context into the state
        st.session_state.used_context = used_context
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    # rerun to show the used context updated (images)
    st.rerun()
