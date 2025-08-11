from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat

def create_app():
    app = FastAPI(title="Chat Bot API",
                description="A modular FastAPI application for chat interactions",
                version="1.0.0")

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # Frontend URL
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    # Include routers
    app.include_router(chat.router, tags=["chat"])
    
    return app

app = create_app()