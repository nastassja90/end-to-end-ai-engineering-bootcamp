def string_for_sse(message: str):
    """Format a message string for Server-Sent Events (SSE).

    Args:
        message (str): The message payload to send to the SSE client.

    Returns:
        str: The message formatted with the SSE "data:" prefix and double newline.
    """
    return f"data: {message}\n\n"
