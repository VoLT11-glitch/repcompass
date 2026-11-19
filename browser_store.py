"""Keep small things in the viewer's own browser (localStorage), so a returning user does not
have to upload their export again. Nothing is stored on the server.

How it works: a tiny invisible custom component (Streamlit CCv2) reads the saved items once per
session and sends them to Python, and writes whatever Python queued. Writes stay queued for the
whole session and are deduplicated by id in the browser, so a st.rerun() can never lose one.
"""
from __future__ import annotations

import uuid

import streamlit as st

PREFIX = "repcompass:"
NAMES = ("history", "setup", "lyfta_key", "optout")
_KEY = "_browser_store"
_QUEUE = "_browser_store_queue"

_JS = """
export default function (component) {
  const { data, setStateValue } = component
  if (!data) return
  const P = data.prefix
  window.__repcompassDone = window.__repcompassDone || new Set()
  const done = window.__repcompassDone
  let err = null
  for (const op of (data.ops || [])) {
    if (done.has(op.id)) continue
    done.add(op.id)
    try {
      if (op.value === null) localStorage.removeItem(P + op.name)
      else localStorage.setItem(P + op.name, op.value)
    } catch (e) { err = String(e) }
  }
  if (data.need) {
    const items = {}
    for (const n of data.names) {
      try { items[n] = localStorage.getItem(P + n) } catch (e) { items[n] = null }
    }
    setTimeout(() => setStateValue("items", items), 0)
  }
  if (err) setTimeout(() => setStateValue("error", err), 0)
}
"""

_COMPONENT = st.components.v2.component(
    "repcompass_browser_store",
    html="<span></span>",
    js=_JS,
)


def items() -> dict | None:
    """What this browser had saved when the session started (plus this session's writes).
    None until the browser has answered, which takes one quick rerun after the page opens."""
    got = (st.session_state.get(_KEY) or {}).get("items")
    if got is None:
        return None
    merged = dict(got)
    for name, op in st.session_state.get(_QUEUE, {}).items():
        merged[name] = op["value"]
    return merged


def ready() -> bool:
    return items() is not None


def get(name: str):
    it = items()
    return it.get(name) if it else None


def put(name: str, value: str | None):
    """Queue a write (None deletes). Skipped when the value is unchanged."""
    if ready() and get(name) == value:
        return
    st.session_state.setdefault(_QUEUE, {})[name] = {"id": uuid.uuid4().hex, "name": name, "value": value}


def clear_all():
    for n in NAMES:
        put(n, None)


def mount():
    """Render the invisible component. Call once per run, after all put() calls if possible."""
    data = {"prefix": PREFIX, "names": list(NAMES), "need": not ready(),
            "ops": list(st.session_state.get(_QUEUE, {}).values())}
    _COMPONENT(key=_KEY, data=data, height=0, on_items_change=lambda: None, on_error_change=lambda: None)
    return (st.session_state.get(_KEY) or {}).get("error")
