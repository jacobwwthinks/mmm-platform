"""
GitHub-backed file persistence for Streamlit Cloud.

On Streamlit Cloud, the filesystem resets on every deploy or container
restart. This module uses the GitHub API to commit file changes back
to the repository, ensuring data survives redeploys.

Requires a GitHub personal access token with 'repo' scope stored in
Streamlit secrets as GITHUB_TOKEN.

Usage:
    from data.github_persist import save_file_to_github
    success = save_file_to_github("events/juniper_events.csv", csv_content, "Update events")
"""

import logging
import base64
from typing import Optional

logger = logging.getLogger(__name__)

# Repository details (auto-detected from Streamlit Cloud environment)
DEFAULT_REPO = "jacobwwthinks/mmm-platform"
DEFAULT_BRANCH = "main"


def _get_github_token() -> Optional[str]:
    """Get GitHub token from Streamlit secrets or environment."""
    try:
        import streamlit as st
        token = st.secrets.get("GITHUB_TOKEN")
        if token:
            return token
    except Exception:
        pass

    import os
    return os.environ.get("GITHUB_TOKEN")


def save_file_to_github(
    file_path: str,
    content: str,
    commit_message: str = "Update file via Streamlit",
    repo: str = DEFAULT_REPO,
    branch: str = DEFAULT_BRANCH,
) -> dict:
    """
    Save a file to the GitHub repository via the API.

    Args:
        file_path: Path within the repo (e.g., "events/juniper_events.csv")
        content: File content as string
        commit_message: Git commit message
        repo: GitHub repo in "owner/name" format
        branch: Branch to commit to

    Returns:
        dict with "success": bool, "message": str
    """
    token = _get_github_token()
    if not token:
        return {
            "success": False,
            "message": "No GITHUB_TOKEN configured. Changes saved locally but will not persist across deploys. "
                       "Add GITHUB_TOKEN to Streamlit secrets for persistent storage.",
        }

    try:
        import urllib.request
        import json

        api_url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "mmm-platform",
        }

        # Get current file SHA (needed for updates)
        sha = None
        try:
            req = urllib.request.Request(api_url + f"?ref={branch}", headers=headers)
            with urllib.request.urlopen(req) as resp:
                existing = json.loads(resp.read().decode())
                sha = existing.get("sha")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            # File doesn't exist yet — that's fine, we'll create it

        # Prepare the update payload
        payload = {
            "message": commit_message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        data = json.dumps(payload).encode()
        req = urllib.request.Request(api_url, data=data, headers=headers, method="PUT")

        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
            commit_sha = result.get("commit", {}).get("sha", "")[:7]
            return {
                "success": True,
                "message": f"Saved to repository ({commit_sha})",
            }

    except Exception as e:
        logger.warning(f"GitHub save failed: {e}")
        return {
            "success": False,
            "message": f"GitHub save failed: {e}. Changes saved locally but may not persist.",
        }
