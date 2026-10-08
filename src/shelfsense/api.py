"""Read-only local analytics API. Ingest complete files through the CLI."""

import os

from .storage import Store


def create_app(database_url=None):
    from fastapi import FastAPI, HTTPException, Query

    store = Store(database_url or os.environ.get("SHELFSENSE_DATABASE_URL", "sqlite:///outputs/shelfsense.db"))
    app = FastAPI(title="ShelfSense Analytics", version="0.5.0")

    def call(function, *args):
        try:
            return function(*args)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            # Never expose a connection string, SQL or server credentials in responses.
            raise HTTPException(status_code=503, detail="Analytics storage unavailable; initialize the database") from exc

    @app.get("/health")
    def health():
        call(store.runs)
        return {"status": "ok"}

    @app.get("/runs")
    def runs():
        return call(store.runs)

    @app.get("/sessions")
    def sessions(run_id: str):
        records = call(store.sessions, run_id)
        return [{key: value for key, value in s.items() if key not in {"zones", "heatmap"}} for s in records]

    @app.get("/analytics")
    def analytics(run_id: str, session_id: str | None = None):
        return call(store.report, run_id, session_id)

    @app.get("/analytics/products")
    def products(run_id: str, session_id: str | None = None):
        return call(store.report, run_id, session_id)["products"]

    @app.get("/analytics/heatmap")
    def heatmap(run_id: str, session_id: str | None = None):
        return call(store.report, run_id, session_id)["heatmap"]

    @app.get("/analytics/funnel")
    def funnel(run_id: str, session_id: str | None = None):
        return call(store.report, run_id, session_id)["funnel"]

    @app.get("/events")
    def events(run_id: str, session_id: str | None = None,
               limit: int = Query(default=100, ge=1, le=500), offset: int = Query(default=0, ge=0)):
        return call(store.events, run_id, session_id, limit, offset)

    return app
