"""Session persistence — (de)serialize the full adjustable state to/from JSON.

Auto-restore keeps the app's last session: on close the window serializes the
current photo path + the :class:`Params` and stows the JSON in ``QSettings``;
on launch it deserializes and replays the normal load/apply paths. This module
is the Qt-free half — pure functions over ``Params`` so the fiddly part
(``asdict`` plus restoring the tuple-of-dataclass fields JSON flattened to
lists of dicts) is unit-testable without a display.
"""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from typing import Optional

from .models import paint_from_dict, target_from_dict
from .params import Params

# Bump when the on-disk shape changes incompatibly; a mismatch makes
# ``decode_session`` return ``None`` so the app silently starts fresh.
SCHEMA = 1

# Only feed back fields the current dataclass still defines, so a blob written
# by an older/newer build can't crash ``Params(**data)`` on a renamed field.
_FIELDS = {f.name for f in fields(Params)}


def _params_to_jsonable(p: Params) -> dict:
    return asdict(p)


def _params_from_jsonable(d: dict) -> Params:
    """Rebuild a ``Params``, restoring the tuple types JSON turned into lists.

    ``palette`` and ``manual_targets`` hold frozen dataclasses; without
    rebuilding them a restored instance would not ``==`` the original and its
    ``solve_key()`` would differ across a restart (defeating the stage cache).
    Unknown keys are dropped so old blobs load forward-compatibly.
    """
    data = {k: v for k, v in d.items() if k in _FIELDS}
    if isinstance(data.get("palette"), list):
        data["palette"] = tuple(paint_from_dict(x) for x in data["palette"])
    if isinstance(data.get("manual_targets"), list):
        data["manual_targets"] = tuple(target_from_dict(x) for x in data["manual_targets"])
    return Params(**data)


def encode_session(*, path: Optional[str], page: int, dpi: int, params: Params) -> str:
    """Serialize the full session state to a JSON string for ``QSettings``."""
    return json.dumps(
        {
            "schema": SCHEMA,
            "path": path,
            "page": page,
            "dpi": dpi,
            "params": _params_to_jsonable(params),
        }
    )


def decode_session(text: str) -> Optional[dict]:
    """Parse a saved blob into a state dict with a real ``Params``.

    Returns ``None`` (→ start fresh, never crash) when the text is empty, not
    JSON, the wrong schema, or malformed.
    """
    if not text:
        return None
    try:
        d = json.loads(text)
        if not isinstance(d, dict) or d.get("schema") != SCHEMA:
            return None
        d["params"] = _params_from_jsonable(d.get("params") or {})
        return d
    except (ValueError, TypeError, KeyError):
        return None
