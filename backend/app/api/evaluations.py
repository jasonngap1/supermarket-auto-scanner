from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_state
from app.schemas import EvaluationRequest

router = APIRouter()


@router.post("/evaluations", status_code=202)
def start_evaluation(payload: EvaluationRequest, state=Depends(get_state)):
    if state.evaluation is None:
        raise HTTPException(status_code=503, detail="Evaluation service unavailable")
    try:
        run_id = state.evaluation.start(payload.variants, payload.item_ids)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"run_id": run_id, "status": "queued"}


@router.get("/evaluations")
def list_evaluations(state=Depends(get_state)):
    return state.evaluation.list() if state.evaluation else []


@router.get("/evaluations/{run_id}")
def get_evaluation(run_id: str, state=Depends(get_state)):
    result = state.evaluation.get(run_id) if state.evaluation else None
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return result


@router.get("/evaluations/{run_id}/failures")
def get_failures(
    run_id: str, variant: str | None = None, limit: int = 50,
    state=Depends(get_state),
):
    result = state.evaluation.get(run_id) if state.evaluation else None
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    failures = result.get("failures", [])
    if variant:
        failures = [row for row in failures if row["variant"] == variant]
    return failures[:limit]
