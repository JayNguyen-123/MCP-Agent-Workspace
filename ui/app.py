"""Operator dashboard: sign in, submit tasks, watch runs, approve/deny package installs."""

from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://agent-service:8000").rstrip("/")
TIMEOUT = 5
STATUS_ICON = {
    "queued": "⏳", "running": "⚙️", "awaiting_approval": "✋",
    "completed": "✅", "terminated": "🛑", "failed": "❌",
}  # fmt: skip

st.set_page_config(page_title="Agent Operations", page_icon="🤖", layout="wide")


# ---------------------------------------------------------------------- client
def api(method: str, path: str, **kwargs) -> requests.Response | None:
    headers = kwargs.pop("headers", {})
    if token := st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.request(method, f"{BACKEND_URL}{path}", headers=headers, timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Backend unreachable: {type(exc).__name__}")
        return None
    if resp.status_code == 401 and st.session_state.get("token"):
        st.session_state.clear()  # expired or revoked: force a fresh sign-in
        st.warning("Your session expired. Please sign in again.")
        st.rerun()
    return resp


def error_detail(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text


# ----------------------------------------------------------------------- login
def login_view() -> None:
    st.title("🤖 Agent Operations")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            resp = api("POST", "/auth/token", json={"username": username, "password": password})
            if resp is not None and resp.ok:
                body = resp.json()
                st.session_state.update(token=body["access_token"], user=username, role=body["role"])
                st.rerun()
            elif resp is not None:
                st.error(error_detail(resp))


def logout() -> None:
    api("POST", "/auth/logout")
    st.session_state.clear()


# ------------------------------------------------------------------------ runs
def submit_task_view() -> None:
    with st.form("new_task", clear_on_submit=True):
        task = st.text_area("New task for the agent team", max_chars=8000, height=120)
        if st.form_submit_button("Start run", type="primary") and task.strip():
            resp = api("POST", "/agent/runs", json={"task": task})
            if resp is not None and resp.status_code == 202:
                st.toast(f"Run {resp.json()['id'][:8]} started")
            elif resp is not None:
                st.error(error_detail(resp))


@st.fragment(run_every=5)
def runs_view() -> None:
    resp = api("GET", "/agent/runs", params={"limit": 25})
    if resp is None or not resp.ok:
        return
    runs = resp.json()["items"]
    if not runs:
        st.caption("No runs yet.")
    for run in runs:
        icon = STATUS_ICON.get(run["status"], "•")
        with st.expander(f"{icon} `{run['id'][:8]}` · {run['status']} · {run['task'][:80]}"):
            st.caption(f"Owner: {run['owner']} · created {run['created_at']}" + (" · cached" if run["cached"] else ""))
            st.markdown("**Task**")
            st.text(run["task"])  # st.text never interprets markdown/HTML from model or user content
            if run["result"]:
                st.markdown("**Result**")
                st.text(run["result"])
            if run["error"]:
                st.error(run["error"])


# ------------------------------------------------------------------- approvals
@st.fragment(run_every=3)
def approvals_view() -> None:
    resp = api("GET", "/agent/approvals", params={"status": "PENDING"})
    if resp is None:
        return
    if not resp.ok:
        st.error(error_detail(resp))
        return
    queue = resp.json()["items"]
    if not queue:
        st.success("No installs awaiting approval.")
        return
    st.subheader(f"Awaiting approval ({len(queue)})")
    for item in queue:
        with st.container(border=True):
            st.markdown(f"**Run** `{item['run_id'][:8]}` requested by **{item['requested_by']}**")
            st.code("pip install --only-binary=:all: " + " ".join(item["packages"]), language="bash")
            own = item["requested_by"] == st.session_state["user"]
            if own:
                st.info("Four-eyes rule: another approver must decide on your own run.")
            # A form keeps the reason and the button click in the same submission.
            with st.form(f"decide_{item['id']}"):
                reason = st.text_input("Reason (recorded in the audit trail)", key=f"reason_{item['id']}")
                c1, c2 = st.columns(2)
                approve = c1.form_submit_button("✅ Approve", disabled=own, use_container_width=True)
                deny = c2.form_submit_button("❌ Deny", disabled=own, use_container_width=True)
            if approve or deny:
                decision = "APPROVED" if approve else "DENIED"
                r = api(
                    "POST",
                    f"/agent/approvals/{item['id']}/decision",
                    json={"decision": decision, "reason": reason or None},
                )
                if r is not None and r.ok:
                    st.toast(f"{decision.title()} by {r.json()['decided_by']}")
                elif r is not None and r.status_code == 409:
                    st.warning("Someone else already decided this request.")
                elif r is not None:
                    st.error(error_detail(r))
                st.rerun(scope="fragment")


def audit_view() -> None:
    resp = api("GET", "/agent/audit-trail", params={"limit": 200})
    if resp is not None and resp.ok:
        rows = resp.json()["items"]
        st.dataframe(
            [{k: r[k] for k in ("id", "run_id", "packages", "status", "requested_by", "decided_by", "reason",
                                "created_at", "decided_at")} for r in rows],
            use_container_width=True,
            hide_index=True,
        )  # fmt: skip


# ------------------------------------------------------------------------ page
if "token" not in st.session_state:
    login_view()
    st.stop()

with st.sidebar:
    st.markdown(f"Signed in as **{st.session_state['user']}** ({st.session_state['role']})")
    st.button("Sign out", on_click=logout)

st.title("🤖 Agent Operations")
is_approver = st.session_state["role"] == "approver"
tabs = st.tabs(["Runs", "Approvals", "Audit trail"] if is_approver else ["Runs"])
with tabs[0]:
    submit_task_view()
    runs_view()
if is_approver:
    with tabs[1]:
        approvals_view()
    with tabs[2]:
        audit_view()
