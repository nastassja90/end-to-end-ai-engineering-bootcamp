from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.server.middleware import RequestIDMiddleware
from api.server.endpoints import api_router
from api.utils.logs import logger

app = FastAPI()

app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
