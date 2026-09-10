"""
log_routes.py

REST API for reading activity logs. Kept as a Blueprint so app.py stays clean
and you can register more route modules later (e.g. /api/stats).
"""

from flask import Blueprint, jsonify, request
from backend.services import db_service

log_routes = Blueprint("log_routes", __name__)


@log_routes.route("/api/logs", methods=["GET"])
def get_logs():
    """
    GET /api/logs
    GET /api/logs?status=violation   (optional filter)

    Returns all rows from activity_log as JSON.
    """
    status_filter = request.args.get("status")

    try:
        if status_filter:
            logs = db_service.fetch_logs_by_status(status_filter)
        else:
            logs = db_service.fetch_all_logs()

        return jsonify({"count": len(logs), "logs": logs}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500