from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.api import documents, ner, system


def create_app() -> FastAPI:
    app = FastAPI(
        title="InNOChem Intelligence Hub API",
        version="0.1.0",
        description=(
            "Backend API for the React migration. Heavy chemistry features remain "
            "step-based and user-triggered."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(system.router)
    app.include_router(documents.router)
    app.include_router(ner.router)
    return app


app = create_app()


