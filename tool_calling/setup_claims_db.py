
"""
LESSON 16 - PART 8.1: DATABASE SETUP
====================================

OBJECTIVE:
Create a local SQLite database containing
fictional insurance claim records.

CONCEPTS:
1. sqlite3.connect()
2. CREATE TABLE
3. INSERT INTO
4. Parameterized SQL
5. Transaction commits

RUN FROM PROJECT ROOT:
python -m tool_calling.setup_claims_db
"""

import sqlite3
from pathlib import Path


# --------------------------------------------------
# STEP 1: DEFINE DATABASE LOCATION
# --------------------------------------------------

# Store database in our tool_calling folder.
# Path(__file__) refers to this Python script.

DB_PATH = Path(__file__).resolve().parent / "claims.db"


# --------------------------------------------------
# STEP 2: DEFINE SAMPLE RECORDS
# --------------------------------------------------

CLAIMS = [
    ("CLM1001", "Under Review", 4500, "Motor Accident"),
    ("CLM1002", "Approved", 850, "Windshield Damage"),
    ("CLM1003", "Rejected", 2200, "Mechanical Breakdown"),
    ("CLM1004", "Approved", 3200, "Motor Accident"),
    ("CLM1005", "Approved", 1800, "Vehicle Damage"),
]


# --------------------------------------------------
# STEP 3: CREATE AND POPULATE DATABASE
# --------------------------------------------------

def setup_database():

    # The context manager commits or rolls back
    # transactions, and we close the connection
    # explicitly in finally.

    connection = sqlite3.connect(DB_PATH)

    try:

        with connection:

            connection.execute("""
                CREATE TABLE IF NOT EXISTS claims (
                    claim_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    amount REAL NOT NULL,
                    claim_type TEXT NOT NULL
                )
            """)

            # Insert/update sample records so the
            # setup script can be rerun safely.
            connection.executemany("""
                INSERT INTO claims (
                    claim_id, status, amount, claim_type
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(claim_id) DO UPDATE SET
                    status = excluded.status,
                    amount = excluded.amount,
                    claim_type = excluded.claim_type
            """, CLAIMS)

    finally:
        connection.close()

    print(f"Database created: {DB_PATH}")
    print(f"Sample records loaded: {len(CLAIMS)}")


if __name__ == "__main__":
    setup_database()
