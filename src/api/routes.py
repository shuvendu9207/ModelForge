"""
ModelForge API Routes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from src.pipeline.pipeline  import Pipeline
from src.tracking.tracker   import ExperimentTracker

router = APIRouter()


class RunRequest(BaseModel):
    dataset:    str
    model_type: str = "random_forest"
    params:     Optional[Dict[str, Any]] = {}


class TrainRequest(BaseModel):
    dataset:   str
    algorithm: str = "random_forest"
    output:    str = "ml/models/model.pkl"


@router.post("/run")
def run_pipeline(req: RunRequest):
    try:
        pipeline = Pipeline(
            dataset_path=req.dataset,
            model_type=req.model_type,
            params=req.params,
        )
        metrics = pipeline.execute()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
def list_experiments():
    tracker = ExperimentTracker()
    return {"experiments": tracker.load_all()}


@router.get("/experiments/{exp_id}")
def get_experiment(exp_id: int):
    tracker = ExperimentTracker()
    try:
        return tracker.load(exp_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/compare")
def compare_experiments():
    tracker     = ExperimentTracker()
    experiments = tracker.load_all()
    if not experiments:
        return {"message": "No experiments found."}

    best = max(experiments,
               key=lambda e: e["metrics"].get("accuracy",
                              e["metrics"].get("f1", 0)))
    return {
        "total":       len(experiments),
        "best_model":  best["model"],
        "best_exp_id": best["id"],
        "best_metrics": best["metrics"],
        "all":         experiments,
    }
