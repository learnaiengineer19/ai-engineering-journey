
"""
LESSON 16 - PART 8.2: LOCAL REST API
===================================

OBJECTIVE:
Expose insurance claim records through an HTTP API.

CONCEPTS:
1. HTTP GET requests
2. API endpoints
3. URL query parameters
4. HTTP response codes
5. JSON serialization
6. SQLite integration

RUN:
python -m tool_calling.claims_api

NOTE:
Local development only. Not for public deployment.
"""

import json
import sqlite3

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from tool_calling.setup_claims_db import DB_PATH


# --------------------------------------------------
# STEP 1: DEFINE THE HTTP REQUEST HANDLER
# --------------------------------------------------

class ClaimsAPIHandler(BaseHTTPRequestHandler):

    def send_json(self, status_code: int, data):
        """Send a JSON response with an HTTP status."""

        payload = json.dumps(data).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()

        self.wfile.write(payload)

    # --------------------------------------------------
    # STEP 2: HANDLE GET REQUESTS
    # --------------------------------------------------

    def do_GET(self):

        parsed_url = urlparse(self.path)

        if parsed_url.path != "/claims":
            self.send_json(
                404,
                {"error": "Endpoint not found"}
            )
            return

        parameters = parse_qs(parsed_url.query)

        status = parameters.get("status", [None])[0]
        amount_text = parameters.get("min_amount", [None])[0]

        allowed_statuses = {
            "Approved",
            "Rejected",
            "Under Review"
        }

        if status not in allowed_statuses:
            self.send_json(
                400,
                {"error": "Invalid or missing status"}
            )
            return

        try:
            min_amount = float(amount_text)

            if not (0 <= min_amount < float("inf")):
                raise ValueError()

        except (TypeError, ValueError):
            self.send_json(
                400,
                {"error": "Invalid or missing min_amount"}
            )
            return

        # --------------------------------------------------
        # STEP 3: QUERY SQLITE
        # --------------------------------------------------

        try:
            connection = sqlite3.connect(
                DB_PATH.as_uri() + "?mode=ro",
                uri=True
            )

            try:
                connection.row_factory = sqlite3.Row

                rows = connection.execute(
                    """
                    SELECT claim_id, status, amount, claim_type
                    FROM claims
                    WHERE status = ?
                      AND amount >= ?
                    ORDER BY amount DESC
                    """,
                    (status, min_amount)
                ).fetchall()

                claims = [dict(row) for row in rows]

            finally:
                connection.close()

        except sqlite3.Error:
            self.send_json(
                500,
                {"error": "Database query failed"}
            )
            return

        # --------------------------------------------------
        # STEP 4: RETURN SUCCESS RESPONSE
        # --------------------------------------------------

        self.send_json(
            200,
            {
                "count": len(claims),
                "claims": claims
            }
        )


# --------------------------------------------------
# STEP 5: START LOCAL SERVER
# --------------------------------------------------

if __name__ == "__main__":

    server = HTTPServer(
        ("127.0.0.1", 8765),
        ClaimsAPIHandler
    )

    print("Claims API running at http://127.0.0.1:8765")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping API server.")
    finally:
        server.server_close()
