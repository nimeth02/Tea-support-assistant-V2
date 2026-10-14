from fastapi import APIRouter, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from observability import store

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/observability/stats")
async def get_stats():
    return store.stats()


@router.get("/observability/traces")
async def get_traces(limit: int = 100):
    return store.list_traces(limit)


@router.get("/observability/traces/{trace_id}")
async def get_trace(trace_id: str):
    trace = store.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@router.get("/observability/spans")
async def get_spans(kind: str | None = None, limit: int = 1000):
    return store.list_spans(kind, limit)
