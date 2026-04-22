"""
Task definitions for the CodeReviewAgent environment.

Three tasks of increasing difficulty. Each defines:
  - code: Python source to review
  - issues: list of ground-truth issues with grading metadata
  - correct_decision: expected final review decision
"""

from typing import Any, Dict, List

TASKS: List[Dict[str, Any]] = [
    # ── Task 0: Easy ─────────────────────────────────────────────────────────
    {
        "id": 0,
        "name": "Basic Bug Detection",
        "difficulty": "easy",
        "file_name": "utils.py",
        "description": (
            "Review this Python utility module. "
            "Identify any bugs, logical errors, or code quality issues. "
            "Add a comment for each issue you find (include line number, severity, "
            "and category), then submit your review."
        ),
        "max_steps": 15,
        "code": """\
def calculate_average(numbers):
    \"\"\"Calculate the average of a list of numbers.\"\"\"
    total = 0
    for i in range(len(numbers) + 1):  # line 4
        total += numbers[i]
    average = total / len(numbers)
    unused_result = sorted(numbers)  # line 7
    return average


def find_max(items):
    \"\"\"Return the maximum value in a list.\"\"\"
    if len(items) == 0:
        return None
    max_val = items[0]
    for item in items:
        if item > max_val:
            max_val == item  # line 17: should be =, not ==
    return max_val


def is_palindrome(s):
    \"\"\"Check if a string is a palindrome.\"\"\"
    return s == s[::-1]
""",
        "issues": [
            {
                "id": "off_by_one",
                "description": "Off-by-one: range(len+1) causes IndexError on the last iteration",
                "line_range": (4, 5),
                "keywords": [
                    "off-by-one", "off by one", "range", "index", "indexerror",
                    "out of bounds", "len + 1", "+ 1", "index out",
                ],
                "category": "bug",
                "severity": "error",
                "weight": 1.0,
            },
            {
                "id": "unused_variable",
                "description": "unused_result is assigned but never used",
                "line_range": (7, 7),
                "keywords": [
                    "unused", "unused_result", "never used", "dead code",
                    "not used", "unnecessary",
                ],
                "category": "style",
                "severity": "info",
                "weight": 0.5,
            },
            {
                "id": "assignment_not_update",
                "description": "max_val == item uses == (comparison) instead of = (assignment); max is never updated",
                "line_range": (17, 17),
                "keywords": [
                    "==", "assignment", "comparison", "max_val", "never update",
                    "not updating", "wrong operator", "should be =", "max never",
                ],
                "category": "bug",
                "severity": "error",
                "weight": 1.0,
            },
        ],
        "correct_decision": "request_changes",
    },

    # ── Task 1: Medium ───────────────────────────────────────────────────────
    {
        "id": 1,
        "name": "Security Vulnerability Review",
        "difficulty": "medium",
        "file_name": "auth.py",
        "description": (
            "Review this authentication module for security vulnerabilities. "
            "Pay careful attention to credential handling, input sanitization, "
            "and cryptographic choices. Annotate every issue with its severity "
            "and category, then submit your review."
        ),
        "max_steps": 20,
        "code": """\
import sqlite3
import hashlib
import os

DB_PASSWORD = "super_secret_123"   # line 5
ADMIN_TOKEN = "tok_admin_abc123"   # line 6

def authenticate_user(username, password):
    \"\"\"Authenticate a user against the database.\"\"\"
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    # line 12: f-string interpolation → SQL injection
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    user = cursor.fetchone()
    conn.close()
    return user is not None


def hash_password(password):
    \"\"\"Hash a password for storage.\"\"\"
    return hashlib.md5(password.encode()).hexdigest()  # line 21


def execute_admin_command(command):
    \"\"\"Execute an admin maintenance command.\"\"\"
    result = eval(command)   # line 25
    return result


def get_user_data(user_id):
    \"\"\"Fetch user profile from internal service.\"\"\"
    import requests
    url = f"https://internal-api/users/{user_id}"
    response = requests.get(url, verify=False)  # line 32
    return response.json()
""",
        "issues": [
            {
                "id": "hardcoded_credentials",
                "description": "Credentials hard-coded in source (lines 5-6)",
                "line_range": (5, 6),
                "keywords": [
                    "hardcoded", "hard-coded", "hard coded", "hardcode",
                    "db_password", "admin_token", "plaintext credential",
                    "environment variable", "env var", "os.environ",
                ],
                "category": "security",
                "severity": "critical",
                "weight": 1.0,
            },
            {
                "id": "sql_injection",
                "description": "SQL injection via unsanitised f-string interpolation",
                "line_range": (12, 14),
                "keywords": [
                    "sql injection", "sql", "injection", "f-string", "parameterized",
                    "sanitize", "escape", "prepared statement", "placeholder",
                ],
                "category": "security",
                "severity": "critical",
                "weight": 1.0,
            },
            {
                "id": "weak_hashing",
                "description": "MD5 is cryptographically broken for password storage",
                "line_range": (21, 21),
                "keywords": [
                    "md5", "weak", "bcrypt", "argon2", "pbkdf2", "scrypt",
                    "cryptographic", "password hashing", "hash", "broken",
                ],
                "category": "security",
                "severity": "error",
                "weight": 0.75,
            },
            {
                "id": "arbitrary_code_execution",
                "description": "eval() on untrusted input allows arbitrary code execution",
                "line_range": (25, 25),
                "keywords": [
                    "eval", "arbitrary code", "code execution", "rce",
                    "remote code", "dangerous", "unsafe",
                ],
                "category": "security",
                "severity": "critical",
                "weight": 1.0,
            },
            {
                "id": "ssl_verification_disabled",
                "description": "verify=False disables TLS cert validation, enabling MITM attacks",
                "line_range": (32, 32),
                "keywords": [
                    "ssl", "verify", "certificate", "mitm",
                    "man-in-the-middle", "tls", "verify=false", "cert",
                ],
                "category": "security",
                "severity": "error",
                "weight": 0.75,
            },
        ],
        "correct_decision": "request_changes",
    },

    # ── Task 2: Hard ─────────────────────────────────────────────────────────
    {
        "id": 2,
        "name": "Full Architecture and Performance Review",
        "difficulty": "hard",
        "file_name": "data_pipeline.py",
        "description": (
            "Perform a comprehensive review of this data pipeline. "
            "Identify bugs, security vulnerabilities, performance bottlenecks, "
            "and architectural design issues. Each comment should clearly explain "
            "the problem and suggest a fix. Submit your review when done."
        ),
        "max_steps": 30,
        "code": """\
import requests
import json
import time
from threading import Thread

API_KEY = "sk-prod-abc123def456"   # line 6


class DataPipeline:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.results = []
        self.cache = {}   # line 13: unbounded

    def fetch_batch(self, item_ids):
        \"\"\"Fetch items from the API.\"\"\"
        items = []
        for item_id in item_ids:   # line 17: N+1 pattern
            response = requests.get(
                f"{self.endpoint}/items/{item_id}",
                headers={"Authorization": f"Bearer {API_KEY}"},
                verify=False,   # line 22
            )
            items.append(response.json())
        return items

    def process_items(self, items):
        \"\"\"Transform items for storage.\"\"\"
        results = []
        for i in range(len(items)):   # line 28: use enumerate
            item = items[i]
            transformed = {
                "id": item["id"],          # line 31: KeyError not handled
                "value": item["value"] * 2,
                "label": item.get("label", "unknown"),
            }
            results.append(transformed)
            self.cache[item["id"]] = transformed   # line 36
        return results

    def run_async(self, func, *args):
        \"\"\"Run function in a background thread.\"\"\"
        t = Thread(target=func, args=args)
        t.start()
        # line 43: thread not tracked or joined — resource leak

    def save_results(self, results, output_path):
        \"\"\"Persist results to disk.\"\"\"
        with open(output_path, "w") as f:
            json.dump(results, f)

    def retry_failed(self, failed_ids, max_retries=10):   # line 50
        \"\"\"Re-fetch items that previously failed.\"\"\"
        for item_id in failed_ids:
            for attempt in range(max_retries):
                try:
                    result = requests.get(
                        f"{self.endpoint}/items/{item_id}"
                    )
                    if result.status_code == 200:
                        self.results.append(result.json())
                        break
                except Exception:
                    time.sleep(1)   # line 60: no exponential backoff
""",
        "issues": [
            {
                "id": "hardcoded_api_key",
                "description": "API key hard-coded in source instead of an environment variable",
                "line_range": (6, 6),
                "keywords": [
                    "hardcoded", "hard-coded", "hardcode", "api key", "api_key",
                    "environment variable", "env var", "os.environ", "sk-prod",
                ],
                "category": "security",
                "severity": "critical",
                "weight": 1.0,
            },
            {
                "id": "n_plus_one_requests",
                "description": "One HTTP request per item (N+1 pattern); should use a bulk/batch endpoint",
                "line_range": (17, 24),
                "keywords": [
                    "n+1", "n plus 1", "batch", "bulk", "loop",
                    "individual request", "serial", "one request per",
                ],
                "category": "performance",
                "severity": "error",
                "weight": 1.0,
            },
            {
                "id": "ssl_disabled",
                "description": "SSL certificate verification disabled (verify=False)",
                "line_range": (22, 22),
                "keywords": [
                    "ssl", "verify", "certificate", "tls",
                    "mitm", "verify=false", "cert",
                ],
                "category": "security",
                "severity": "error",
                "weight": 0.75,
            },
            {
                "id": "missing_key_error_handling",
                "description": "Direct dict access item['id'] / item['value'] raises KeyError on unexpected payloads",
                "line_range": (31, 32),
                "keywords": [
                    "keyerror", "key error", "error handling", "missing key",
                    "exception", "try", ".get(", "dict access",
                ],
                "category": "bug",
                "severity": "warning",
                "weight": 0.75,
            },
            {
                "id": "unbounded_cache",
                "description": "self.cache grows without bound; will cause OOM on large inputs",
                "line_range": (13, 13),
                "keywords": [
                    "unbounded", "memory leak", "cache size", "limit",
                    "lru", "eviction", "grow", "oom", "memory",
                ],
                "category": "design",
                "severity": "warning",
                "weight": 0.75,
            },
            {
                "id": "thread_not_joined",
                "description": "Thread is started but never stored or joined — silent resource/exception leak",
                "line_range": (40, 43),
                "keywords": [
                    "thread", "join", "track", "resource leak",
                    "daemon", "not joined", "not tracked",
                ],
                "category": "bug",
                "severity": "error",
                "weight": 1.0,
            },
            {
                "id": "no_exponential_backoff",
                "description": "Retry loop sleeps 1 s flat; needs exponential backoff to avoid hammering the API",
                "line_range": (50, 60),
                "keywords": [
                    "backoff", "exponential", "retry", "sleep", "rate limit",
                    "jitter", "aggressive",
                ],
                "category": "design",
                "severity": "warning",
                "weight": 0.5,
            },
        ],
        "correct_decision": "request_changes",
    },
]
