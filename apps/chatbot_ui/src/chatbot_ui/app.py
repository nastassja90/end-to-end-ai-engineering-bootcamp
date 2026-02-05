import streamlit as st
import requests
from chatbot_ui.core.config import config
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


session_id = get_session_id()


@st.cache_data(ttl=300)
def fetch_app_config():
    """Fetch configuration from the backend API."""
    try:
        response = requests.get(f"{config.API_URL}/config")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch configuration from API: {e}")
        return {"models": {}, "providers": []}


# Fetch configuration at startup
app_config = fetch_app_config()
MODELS = app_config.get("models", {})
PROVIDERS = app_config.get("providers", [])
TOP_K = app_config.get("top_k", {"default": 5, "max": 20})


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


def submit_feedback(feedback_type=None, feedback_text=""):
    """Submit feedback to the API endpoint"""

    def _feedback_score(feedback_type):
        if feedback_type == "positive":
            return 1
        elif feedback_type == "negative":
            return 0
        else:
            return None

    feedback_data = {
        "feedback_score": _feedback_score(feedback_type),
        "feedback_text": feedback_text,
        "trace_id": st.session_state.trace_id,
        "thread_id": session_id,
        "feedback_source_type": "api",
    }

    logger.info(f"Feedback data: {feedback_data}")

    status, response = api_call(
        "post", f"{config.API_URL}/feedback", json=feedback_data
    )
    return status, response


# Initialize feedback states (simplified)
if "latest_feedback" not in st.session_state:
    st.session_state.latest_feedback = None

if "show_feedback_box" not in st.session_state:
    st.session_state.show_feedback_box = False

if "feedback_submission_status" not in st.session_state:
    st.session_state.feedback_submission_status = None

if "trace_id" not in st.session_state:
    st.session_state.trace_id = None


if "used_context" not in st.session_state:
    st.session_state.used_context = []

## Lets create a sidebar with a dropdown for the model list and providers
with st.sidebar:
    st.title("Settings")

    # Dropdown for provider and model
    provider = st.selectbox("Provider", PROVIDERS)
    model_name = st.selectbox("Model", MODELS.get(provider, []))
    # Save provider and model to session state
    st.session_state.provider = provider
    st.session_state.model_name = model_name

    st.subheader("Execution Type")
    execution_type = st.radio(
        "Select RAG Execution Type",
        ("pipeline", "agent"),
        help="Choose 'pipeline' for a straightforward retrieval and generation process, or 'agent' for a more dynamic interaction using an agent-based approach.",
    )
    st.session_state.execution_type = execution_type

    # Advanced Options section
    st.subheader("Advanced Options")
    enable_reranking = st.checkbox("Enable Re-ranking", value=False)
    top_k = st.slider(
        "Number of Contexts to Retrieve (top_k)",
        min_value=1,
        max_value=TOP_K["max"],
        value=TOP_K["default"],
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


for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Add feedback buttons only for the latest assistant message (excluding the initial greeting)
        is_latest_assistant = (
            message["role"] == "assistant"
            and idx == len(st.session_state.messages) - 1
            and idx > 0
        )

        if is_latest_assistant:
            # Use Streamlit's built-in feedback component
            feedback_key = f"feedback_{len(st.session_state.messages)}"
            feedback_result = st.feedback("thumbs", key=feedback_key)

            # Handle feedback selection
            if feedback_result is not None:
                feedback_type = "positive" if feedback_result == 1 else "negative"

                # Only submit if this is a new/different feedback
                if st.session_state.latest_feedback != feedback_type:
                    with st.spinner("Submitting feedback..."):
                        status, response = submit_feedback(feedback_type=feedback_type)
                        if status:
                            st.session_state.latest_feedback = feedback_type
                            st.session_state.feedback_submission_status = "success"
                            st.session_state.show_feedback_box = (
                                feedback_type == "negative"
                            )
                        else:
                            st.session_state.feedback_submission_status = "error"
                            st.error("Failed to submit feedback. Please try again.")
                    st.rerun()

            # Show feedback status message
            if (
                st.session_state.latest_feedback
                and st.session_state.feedback_submission_status == "success"
            ):
                if st.session_state.latest_feedback == "positive":
                    st.success("✅ Thank you for your positive feedback!")
                elif (
                    st.session_state.latest_feedback == "negative"
                    and not st.session_state.show_feedback_box
                ):
                    st.success("✅ Thank you for your feedback!")
            elif st.session_state.feedback_submission_status == "error":
                st.error("❌ Failed to submit feedback. Please try again.")

            # Show feedback text box if thumbs down was pressed
            if st.session_state.show_feedback_box:
                st.markdown("**Want to tell us more? (Optional)**")
                st.caption(
                    "Your negative feedback has already been recorded. You can optionally provide additional details below."
                )

                # Text area for detailed feedback
                feedback_text = st.text_area(
                    "Additional feedback (optional)",
                    key=f"feedback_text_{len(st.session_state.messages)}",
                    placeholder="Please describe what was wrong with this response...",
                    height=100,
                )

                # Send additional feedback button
                col_send, col_spacer, col_close = st.columns([3, 5, 2])
                with col_send:
                    if st.button(
                        "Send Additional Details",
                        key=f"send_additional_{len(st.session_state.messages)}",
                    ):
                        if feedback_text.strip():  # Only send if there's actual text
                            with st.spinner("Submitting additional feedback..."):
                                status, response = submit_feedback(
                                    feedback_text=feedback_text
                                )
                                if status:
                                    st.success(
                                        "✅ Thank you! Your additional feedback has been recorded."
                                    )
                                    st.session_state.show_feedback_box = False
                                else:
                                    st.error(
                                        "❌ Failed to submit additional feedback. Please try again."
                                    )
                        else:
                            st.warning(
                                "Please enter some feedback text before submitting."
                            )
                        st.rerun()

                with col_close:
                    if st.button(
                        "Close", key=f"close_feedback_{len(st.session_state.messages)}"
                    ):
                        st.session_state.show_feedback_box = False
                        st.rerun()


if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        json = {
            "execution_type": st.session_state.execution_type,
            "provider": st.session_state.provider,
            "model_name": st.session_state.model_name,
            "query": prompt,
            "thread_id": session_id,
            "extra_options": {
                "top_k": st.session_state.top_k,
                "enable_reranking": st.session_state.enable_reranking,
            },
        }
        status, output = api_call("post", f"{config.API_URL}/rag", json=json)
        answer = output["answer"]
        used_context = output["used_context"]
        trace_id = output["trace_id"]
        st.session_state.trace_id = trace_id

        # set the used_context into the state
        st.session_state.used_context = used_context
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    # rerun to show the used context updated (images)
    st.rerun()
