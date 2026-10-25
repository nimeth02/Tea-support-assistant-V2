from agent_loader import mount_agent_variant

# Must run before anything imports `agent`: picks which variant folder *is* `agent` in this
# process (AGENT_VARIANT env var, default "agent") and points LangSmith at its project.
VARIANT = mount_agent_variant()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from backend.routes import chat  # noqa: E402
from observability import store  # noqa: E402
from observability.logging_config import setup_logging  # noqa: E402
from observability.middleware import add_observability_middleware  # noqa: E402
from observability.routes import router as observability_router  # noqa: E402
from variants import VARIANTS  # noqa: E402


def create_app():
    setup_logging()
    store.init_db()

    if VARIANT.key == "crewai":
        # CrewAI doesn't use LangChain callbacks, so bridge its events to LangSmith.
        from observability import langsmith_crewai
        langsmith_crewai.install()

    app = FastAPI(title=f"Chat Bot API - {VARIANT.title}",
                description=f"Tea support chat API backed by the {VARIANT.framework}",
                version="1.0.0")

    # Configure CORS: every variant's Streamlit UI
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[f"http://localhost:{v.ui_port}" for v in VARIANTS.values()],
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
