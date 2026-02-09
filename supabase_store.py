import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import requests


logger = logging.getLogger(__name__)
_warned_missing_config = False


def _get_supabase_config() -> Optional[Dict[str, str]]:
    global _warned_missing_config
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not key:
        key = os.getenv("SUPABASE_ANON_KEY", "").strip()
    table = os.getenv("SUPABASE_RAG_TABLE", "rag_responses").strip()

    if not url or not key:
        if not _warned_missing_config:
            logger.warning("Supabase config missing; set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
            _warned_missing_config = True
        return None

    return {"url": url.rstrip("/"), "key": key, "table": table}


def store_rag_result(question: str, property_data: Dict[str, Any], answer: str, query_mode: str = "hybrid") -> None:
    config = _get_supabase_config()
    if not config:
        return

    payload = {
        "question": question,
        "answer": str(answer),
        "property_data": property_data,
        "property_address": property_data.get("Address") if isinstance(property_data, dict) else None,
        "query_mode": query_mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    headers = {
        "apikey": config["key"],
        "Authorization": f"Bearer {config['key']}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    endpoint = f"{config['url']}/rest/v1/{config['table']}"
    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=10)
        if resp.status_code >= 400:
            logger.warning("Supabase insert failed (%s): %s", resp.status_code, resp.text)
    except Exception:
        logger.exception("Supabase insert failed")


def maybe_clear_local_cache(working_dir: str) -> None:
    disable_cache = os.getenv("RAG_DISABLE_LOCAL_CACHE", "").strip().lower() in {"1", "true", "yes"}
    if not disable_cache:
        if _get_supabase_config() is None:
            return
    cache_path = Path(working_dir) / "kv_store_llm_response_cache.json"
    try:
        if cache_path.exists():
            cache_path.unlink()
    except Exception:
        logger.exception("Failed to clear local RAG cache file")
