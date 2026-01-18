def hide_sensitive_inputs(inputs: dict) -> dict:
    """Filter out sensitive inputs from tracing."""
    sensitive_keys = {"app_config", "qdrant_client"}
    filtered = {k: v for k, v in inputs.items() if k not in sensitive_keys}
    return filtered
