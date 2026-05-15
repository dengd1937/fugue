"""src/fugue/server/__init__.py — Server 层（FastAPI + CLI）。"""

from fugue.server.app import create_app
from fugue.server.endpoints import IngestRequest, QueryRequest, create_endpoints

__all__ = [
    "IngestRequest",
    "QueryRequest",
    "create_app",
    "create_endpoints",
]
