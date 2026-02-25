from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from uuid import uuid4
from api.utils.logs import logger


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that adds a unique request ID to each request."""

    async def dispatch(self, request: Request, call_next):

        request_id = str(uuid4())

        request.state.request_id = request_id
        logger.info(
            f"Request started: {request.method} {request.url.path} (request_id: {request_id})"
        )

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        logger.info(
            f"Request completed: {request.method} {request.url.path} (request_id: {request_id})"
        )

        return response
