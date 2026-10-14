from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import chat
from observability import store
from observability.logging_config import setup_logging
from observability.middleware import add_observability_middleware
from observability.routes import router as observability_router


def create_app():
    setup_logging()
    store.init_db()

    app = FastAPI(title="Chat Bot API",
                description="A modular FastAPI application for chat interactions",
                version="1.0.0")

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8510", "http://localhost:8511"],  # Streamlit frontend URL
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    # Trace ids + HTTP metrics/logging for every request
    add_observability_middleware(app)

    # Include routers
    app.include_router(chat.router, tags=["chat"])
    app.include_router(observability_router, tags=["observability"])

    return app

app = create_app()
