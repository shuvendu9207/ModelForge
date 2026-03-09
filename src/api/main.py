"""
ModelForge FastAPI Server
Run: uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="ModelForge API",
    description="End-to-End ML Platform REST API",
    version="1.0.0"
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "ModelForge is running", "docs": "/docs"}
