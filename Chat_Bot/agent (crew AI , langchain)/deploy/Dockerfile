# One image for every service of the multi-framework A2A variant: the three experts, the
# backend (orchestrator) and the UI. docker-compose.yml picks what each container runs.
#
# Build context = the REPO ROOT (the folder holding Chat_Bot/ and "ML Models/"), because the
# experts need the trained model files from "ML Models/":
#   docker build -f "Chat_Bot/agent (crew AI , langchain)/deploy/Dockerfile" -t tea-agents .
#
# Inside the image the layout mirrors a local checkout (/app = repo root), so the same
# commands and paths work in both places.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    AGENT_VARIANT=hybrid \
    ENVIRONMENT=production \
    LOG_FORMAT=json \
    A2A_BIND_HOST=0.0.0.0

WORKDIR /app/Chat_Bot

# Dependencies first: this layer is cached until requirements.txt changes.
COPY Chat_Bot/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Trained models (only the ones this variant uses - not the 500 MB VGG16 challenger).
COPY ["ML Models/Tea Fertilizer Dosage/model/tea_fertilizer_model.pkl", "/app/ML Models/Tea Fertilizer Dosage/model/"]
COPY ["ML Models/Tea Disease Classification/model/tea_disease_resnet18.pkl", "/app/ML Models/Tea Disease Classification/model/"]

# Application code: the variant folder (under its real name) plus the shared layers.
COPY ["Chat_Bot/agent (crew AI , langchain)", "/app/Chat_Bot/agent (crew AI , langchain)"]
COPY Chat_Bot/backend ./backend
COPY Chat_Bot/observability ./observability
COPY Chat_Bot/frontend ./frontend
COPY Chat_Bot/agent_loader.py Chat_Bot/variants.py Chat_Bot/launch.py ./

# Never run as root in production.
RUN useradd --create-home --uid 10001 app && mkdir -p data && chown -R app:app /app
USER app

# Build the local vector knowledge bases at image build time, so every container starts with
# the same, reproducible data. Done as `app` (not root) because Chroma caches its embedding
# model in the user's home folder, and the running containers need to find it there.
RUN python "agent (crew AI , langchain)/run_hybrid.py" ingest --reset

# Default: one expert. docker-compose.yml sets the command per service.
CMD ["python", "agent (crew AI , langchain)/run_hybrid.py", "experts", "--agent", "help"]
