#!/usr/bin/env python3
"""
Render Deployment Automation Utility via Official API.
Allows creating the Web Service, triggering deployments, and checking logs/status.
"""

import json
import os
import sys
import urllib.request
import urllib.error

RENDER_API_KEY = os.environ.get("RENDER_API_KEY", "rnd_hMeYEoidRmWVHQGkYEY2Q0iwVTvF")
BASE_URL = "https://api.render.com/v1"
OWNER_ID = "tea-csppfnt6l47c73djr2t0" # benevalter's Workspace
REPO_URL = "https://github.com/Benevalterjr/nativa"


def make_request(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {RENDER_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content) if content else {}
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        print(f"[HTTP ERROR {e.code}] {err_msg}", file=sys.stderr)
        return {"error": err_msg, "code": e.code}


def list_services():
    services = make_request("services")
    print(json.dumps(services, indent=2))
    return services


def create_service(name: str = "nativa-qwen3-api"):
    payload = {
        "type": "web_service",
        "name": name,
        "ownerId": OWNER_ID,
        "repo": REPO_URL,
        "branch": "main",
        "autoDeployTrigger": "commit",
        "serviceDetails": {
            "runtime": "docker",
            "plan": "free",
            "region": "oregon",
            "dockerfilePath": "./Dockerfile",
            "dockerContext": ".",
            "envVars": [
                {"key": "PORT", "value": "8000"},
                {"key": "OMP_NUM_THREADS", "value": "2"},
                {"key": "OMP_WAIT_POLICY", "value": "PASSIVE"},
            ],
        },
    }
    print(f"[*] Creating Web Service '{name}' on Render...")
    res = make_request("services", method="POST", data=payload)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "list"
    if action == "list":
        list_services()
    elif action == "create":
        name = sys.argv[2] if len(sys.argv) > 2 else "nativa-qwen3-api"
        create_service(name)
    else:
        print("Usage: python deploy_render.py [list|create <name>]")
