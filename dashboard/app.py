"""Run with: streamlit run dashboard/app.py (talks to the local FastAPI service)."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import streamlit as st


def fetch(path, **params):
    base = os.environ.get("SHELFSENSE_API_URL", "http://127.0.0.1:8000").rstrip("/")
    url = base + path + ("?" + urlencode(params) if params else "")
    with urlopen(url, timeout=10) as response:
        return json.load(response)


def main():
    st.set_page_config(page_title="ShelfSense", page_icon="👁", layout="wide")
    st.title("ShelfSense")
    st.caption("Estimated shelf attention · fixed-camera research")
    st.button("Refresh")
    try:
        runs = fetch("/runs")
        if not runs:
            st.info("No completed runs yet. Import an attention export to see analytics.")
            return
        run_id = st.sidebar.selectbox("Processing run", [r["run_id"] for r in runs])
        run = next(r for r in runs if r["run_id"] == run_id)
        st.sidebar.caption(f"Shelf: {run['shelf_id']}")
        sessions = fetch("/sessions", run_id=run_id)
        # Use a distinct null value for All sessions; a real session can have any string ID.
        selected = st.sidebar.selectbox("Session", [None] + [s["session_id"] for s in sessions],
                                        format_func=lambda value: "All sessions" if value is None else value)
        params = {"run_id": run_id}
        if selected is not None:
            params["session_id"] = selected
        report = fetch("/analytics", **params)
        columns = st.columns(4)
        columns[0].metric("Attention", f"{report['total_attention_s']:.2f} s")
        columns[1].metric("Completed episodes", sum(p["episode_count"] for p in report["products"]))
        columns[2].metric("Track observations mapped", f"{report['coverage']:.1%}")
        columns[3].metric("Tracked IDs", report["funnel"]["tracked"])
        st.caption("IDs are session-scoped tracks, not verified unique people. Only the selected run is counted.")
        left, right = st.columns(2)
        with left:
            st.subheader("Zone attention")
            st.vega_lite_chart(spec={"data": {"values": report["products"]},
                "mark": "bar", "encoding": {
                    "x": {"field": "total_attention_s", "type": "quantitative", "title": "Attention (seconds)"},
                    "y": {"field": "zone_id", "type": "nominal", "sort": "-x", "title": "Shelf zone"},
                    "tooltip": [{"field": "zone_id"}, {"field": "total_attention_s", "format": ".2f"},
                                {"field": "episode_count"}]}}, use_container_width=True)
        with right:
            st.subheader("Observed-gaze heatmap")
            heatmap = report["heatmap"]
            cells = [{"x": x, "y": y, "seconds": value}
                     for y, row in enumerate(heatmap["values"]) for x, value in enumerate(row)]
            st.vega_lite_chart(spec={"data": {"values": cells},
                "mark": "rect", "encoding": {
                    "x": {"field": "x", "type": "ordinal", "title": "Shelf column (left → right)"},
                    "y": {"field": "y", "type": "ordinal", "sort": "ascending", "title": "Shelf row (top → bottom)"},
                    "color": {"field": "seconds", "type": "quantitative", "title": "Seconds", "scale": {"scheme": "tealblues"}},
                    "tooltip": [{"field": "seconds", "format": ".2f"}]}}, use_container_width=True)
            st.caption("Includes unconfirmed looks. Missing gaze and long gaps add no time.")
        st.subheader("Attention funnel")
        funnel = report["funnel"]
        st.dataframe([
            {"Stage": "Tracked", "Session track IDs": funnel["tracked"]},
            {"Stage": "Valid shelf observation", "Session track IDs": funnel["viewed"]},
            {"Stage": "Confirmed attention", "Session track IDs": funnel["confirmed"]},
        ], hide_index=True, use_container_width=True)
        st.subheader("Zone details")
        st.dataframe(report["products"], hide_index=True, use_container_width=True)
        with st.expander("Events (first 100 in emission order)"):
            st.dataframe(fetch("/events", **params, limit=100), hide_index=True, use_container_width=True)
        st.download_button("Download selected report", json.dumps(report, indent=2),
                           file_name="shelfsense-report.json", mime="application/json")
        st.caption("Attention episodes are estimates; they do not establish purchase intent or physiological fixations.")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError):
        st.error("Analytics could not be loaded. Check that the API is running and the database is initialized.")


if __name__ == "__main__":
    main()
