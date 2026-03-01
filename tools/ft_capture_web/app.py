from __future__ import annotations

from pathlib import Path
from typing import Dict
from urllib.parse import urlencode

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from tools.ft_capture.catalog import ScenarioCatalog, load_catalog

from .service import (
    RecordFilters,
    build_record_views,
    default_expected_decision_json,
    find_record_view,
    record_by_token,
    scenario_choices,
    update_label_from_form,
)



def _build_context(
    *,
    request: Request,
    catalog: ScenarioCatalog,
    dataset_root: str,
    mount_prefix: str,
    selected_token: str,
    filters: RecordFilters,
    message: str,
    error: str,
) -> Dict[str, object]:
    views = build_record_views(
        dataset_root=dataset_root,
        catalog=catalog,
        mount_prefix=mount_prefix,
        filters=filters,
    )
    selected = find_record_view(views, selected_token)

    expected_decision_json = ""
    if selected is not None:
        expected_decision_json = default_expected_decision_json(selected.record)

    return {
        "request": request,
        "catalog": catalog,
        "dataset_root": dataset_root,
        "records": views,
        "selected": selected,
        "selected_token": selected.token if selected is not None else "",
        "expected_decision_json": expected_decision_json,
        "message": message,
        "error": error,
        "scenario_choices": scenario_choices(views),
        "filters": {
            "scenario": filters.scenario,
            "status": filters.status,
            "split": filters.split,
            "quality": filters.quality,
        },
    }



def create_app(*, dataset_root: str, catalog_path: str, mount_prefix: str = "/dataset-files") -> FastAPI:
    root = Path(dataset_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(catalog_path)

    module_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(module_dir / "templates"))

    app = FastAPI(title="PALA Fine-tune Capture Review", version="0.1.0")
    app.mount("/ft-capture-static", StaticFiles(directory=str(module_dir / "static")), name="ft_capture_static")
    app.mount(mount_prefix, StaticFiles(directory=str(root)), name="ft_capture_dataset")

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index(
        request: Request,
        take: str = "",
        scenario: str = "",
        status: str = "",
        split: str = "",
        quality: str = "",
        message: str = "",
        error: str = "",
    ) -> HTMLResponse:
        filters = RecordFilters(
            scenario=scenario,
            status=status,
            split=split,
            quality=quality,
        )
        ctx = _build_context(
            request=request,
            catalog=catalog,
            dataset_root=str(root),
            mount_prefix=mount_prefix,
            selected_token=take,
            filters=filters,
            message=message,
            error=error,
        )
        return templates.TemplateResponse(request, "index.html", ctx)

    @app.post("/label/{token}")
    async def save_label(
        request: Request,
        token: str,
        status: str = Form("unlabeled"),
        quality_flag: str = Form("usable"),
        annotator: str = Form(""),
        rationale_text: str = Form(""),
        notes: str = Form(""),
        expected_decision_json: str = Form(""),
        scenario_filter: str = Form(""),
        status_filter: str = Form(""),
        split_filter: str = Form(""),
        quality_filter: str = Form(""),
    ) -> HTMLResponse:
        record = record_by_token(str(root), token)
        if record is None:
            raise HTTPException(status_code=404, detail="take not found")

        filters = RecordFilters(
            scenario=scenario_filter,
            status=status_filter,
            split=split_filter,
            quality=quality_filter,
        )

        try:
            update_label_from_form(
                record=record,
                status=status,
                quality_flag=quality_flag,
                annotator=annotator,
                rationale_text=rationale_text,
                notes=notes,
                expected_decision_json=expected_decision_json,
            )
        except Exception as exc:
            ctx = _build_context(
                request=request,
                catalog=catalog,
                dataset_root=str(root),
                mount_prefix=mount_prefix,
                selected_token=token,
                filters=filters,
                message="",
                error=str(exc),
            )
            ctx["expected_decision_json"] = expected_decision_json
            return templates.TemplateResponse(request, "index.html", ctx)

        redirect = request.url_for("index")
        params = {
            "take": token,
            "scenario": filters.scenario,
            "status": filters.status,
            "split": filters.split,
            "quality": filters.quality,
            "message": "label saved",
        }
        query = urlencode({key: value for key, value in params.items() if value})
        target = f"{redirect}?{query}" if query else str(redirect)
        return RedirectResponse(url=target, status_code=303)

    return app
