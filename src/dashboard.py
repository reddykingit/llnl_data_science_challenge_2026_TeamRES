from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

try:
    import openai
except ImportError:
    openai = None

from mcp_server import (
    segment_ct_dataset,
    visualize_slice,
    skeletonize,
    structural_weakness_analysis_tool,
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
UPLOAD_ROOT = PROJECT_ROOT / "uploads"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "dashboard"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)

SUPPORTED_VOLUME_EXT = {".npy", ".tif", ".tiff"}
SUPPORTED_GRAPH_EXT = {".json"}
SUPPORTED_REPORT_EXT = {".json"}

SYSTEM_PROMPT = (
    "You are a dashboard assistant for a local lattice CT analysis project. "
    "Available tools: segment_ct_dataset, visualize_slice, skeletonize, "
    "structural_weakness_analysis_tool. "
    "When the user asks for an action, suggest the appropriate tool and parameters. "
    "If you cannot perform the request, ask for a valid file selection or parameters."
)


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _find_files() -> dict[str, list[str]]:
    ct_files = []
    graph_files = []
    report_files = []
    for root in (DATA_ROOT, UPLOAD_ROOT):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                if path.suffix.lower() in SUPPORTED_VOLUME_EXT:
                    ct_files.append(_relative_path(path))
                elif path.suffix.lower() in SUPPORTED_GRAPH_EXT:
                    graph_files.append(_relative_path(path))
                elif path.suffix.lower() in SUPPORTED_REPORT_EXT:
                    report_files.append(_relative_path(path))
    return {
        "ct_files": ct_files,
        "graph_files": graph_files,
        "report_files": report_files,
    }


def _apply_openai_chat(query: str) -> tuple[str, dict[str, Any]]:
    if openai is None:
        raise RuntimeError("OpenAI Python package not installed")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    openai.api_key = api_key
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=400,
        temperature=0.5,
    )
    message = response.choices[0].message.content.strip()
    return message, {"model": model, "usage": response.usage.to_dict() if hasattr(response, "usage") else {}}


def _fallback_chat(query: str) -> str:
    normalized = query.strip().lower()
    if "segment" in normalized:
        return (
            "I can run segmentation on a selected CT volume. "
            "Use the segmentation tool with a threshold, input file, and output path."
        )
    if "visualize" in normalized or "slice" in normalized:
        return (
            "I can generate a slice image from a 3D volume. "
            "Select the volume, slice index, and axis to visualize."
        )
    if "heatmap" in normalized or "weak" in normalized:
        return (
            "I can produce a structural weakness heatmap using a defect report and registered graph. "
            "Choose the support plane and load plane and run structural analysis."
        )
    if "help" in normalized or "how" in normalized:
        return (
            "This dashboard supports segmentation, slice visualization, skeletonization, "
            "and structural weakness analysis. Use the forms above to run tools or ask me to guide you."
        )
    return (
        "I can help you run the available lattice CT tools. "
        "Ask me to segment a volume, visualize a slice, skeletonize a mask, or create a weak-zone heatmap."
    )


def _guess_threshold(text: str) -> float | None:
    match = re.search(r"threshold\s*[:=]?\s*([0-9]*\.?[0-9]+)", text)
    if match:
        return float(match.group(1))
    return None


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/files")
def files():
    return jsonify(_find_files())


@app.route("/api/segment", methods=["POST"])
def api_segment():
    data = request.get_json(force=True)
    input_filepath = PROJECT_ROOT / data.get("input_filepath", "")
    output_filename = data.get("output_filename", "segmented_mask.npy")
    threshold = float(data.get("threshold", 0.5))
    output_path = DEFAULT_OUTPUT_DIR / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = segment_ct_dataset(str(input_filepath), str(output_path), threshold)
    return jsonify({"result": result, "output_path": _relative_path(output_path)})


@app.route("/api/visualize", methods=["POST"])
def api_visualize():
    data = request.get_json(force=True)
    input_filepath = PROJECT_ROOT / data.get("input_filepath", "")
    output_filename = data.get("output_filename", "slice_visualization.png")
    slice_index = int(data.get("slice_index", 0))
    axis = int(data.get("axis", 0))
    output_path = DEFAULT_OUTPUT_DIR / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = visualize_slice(str(input_filepath), str(output_path), slice_index, axis)
    return jsonify({"result": result, "output_path": _relative_path(output_path)})


@app.route("/api/structural-analysis", methods=["POST"])
def api_structural_analysis():
    data = request.get_json(force=True)
    defect_report_filepath = PROJECT_ROOT / data.get("defect_report_filepath", "")
    registered_graph_filepath = PROJECT_ROOT / data.get("registered_graph_filepath", "")
    support_plane = data.get("support_plane", "xy")
    load_plane = data.get("load_plane", "xy")
    output_subdir = data.get("output_subdir", "dashboard")
    output_path = PROJECT_ROOT / "analysis" / output_subdir
    output_path.mkdir(parents=True, exist_ok=True)
    result = structural_weakness_analysis_tool(
        str(defect_report_filepath),
        str(registered_graph_filepath),
        support_plane,
        load_plane,
        str(output_path),
    )
    return jsonify(result)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    upload = request.files["file"]
    if upload.filename == "":
        return jsonify({"error": "No filename provided."}), 400
    filename = secure_filename(upload.filename)
    if not filename:
        return jsonify({"error": "Invalid filename."}), 400
    destination = UPLOAD_ROOT / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    upload.save(destination)
    return jsonify({"path": _relative_path(destination)})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is required."}), 400

    if openai is not None and os.environ.get("OPENAI_API_KEY"):
        try:
            message, meta = _apply_openai_chat(query)
            return jsonify({"assistant": message, "meta": meta})
        except Exception as error:
            return jsonify({"assistant": _fallback_chat(query), "warning": str(error)})

    return jsonify({"assistant": _fallback_chat(query)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
