
# study_planner_app.py
# A feature-complete, single-user Study Planner built with Streamlit + SQLite
#
# To run:
#   1) pip install -r requirements.txt
#   2) streamlit run study_planner_app.py
#
# Additions in this version:
# - Dashboard summary (counts, overdue)
# - Overdue highlighting in task table
# - Bulk CSV import (with downloadable template)
# - .ICS export for today's plan and for all task due dates
# - Weekly progress chart and totals
#
# Roadmap ideas: notifications, repeating tasks, Google Calendar sync

import streamlit as st
import sqlite3
import pandas as pd
import datetime as dt
from dataclasses import dataclass
import io

DB_PATH = "study_planner.db"

# ---------- Utilities ----------

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS subjects(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        color TEXT DEFAULT '#A3A3A3'
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tasks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        subject_id INTEGER,
        due_date TEXT,
        estimate_min INTEGER DEFAULT 30,
        priority INTEGER DEFAULT 3, -- 1=High, 2=Med, 3=Low
        status TEXT DEFAULT 'todo', -- todo, doing, done
        notes TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE SET NULL
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER,
        subject_id INTEGER,
        date TEXT DEFAULT (date('now')),
        start_time TEXT,
        end_time TEXT,
        duration_min INTEGER,
        notes TEXT DEFAULT '',
        FOREIGN KEY(task_id) REFERENCES tasks(id),
        FOREIGN KEY(subject_id) REFERENCES subjects(id)
    )""")
    conn.commit()

def run_query(query, params=(), fetch="all"):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(query, params)
    conn.commit()
    if fetch == "one":
        r = cur.fetchone()
    elif fetch == "all":
        r = cur.fetchall()
    else:
        r = None
    return r

def df_from_rows(rows):
    return pd.DataFrame([dict(r) for r in rows])

def ensure_defaults():
    subs = run_query("SELECT * FROM subjects LIMIT 1")
    if not subs:
        for name, color in [("Math", "#4F46E5"), ("Science", "#059669"), ("History", "#EF4444")]:
            run_query("INSERT INTO subjects(name, color) VALUES (?,?)", (name, color))

# ---------- Scheduling Logic ----------

@dataclass
class TaskPlan:
    task_id: int
    title: str
    subject: str
    start: dt.datetime
    end: dt.datetime
    minutes: int
    due_date: str
    priority: int

def plan_day(tasks_df, start_time, total_minutes, block_len=25, short_break=5):
    """Greedy: due date asc, priority asc, created asc; fill Pomodoro-sized blocks until time is used."""
    if tasks_df.empty or total_minutes <= 0:
        return []
    tasks_df = tasks_df.copy()
    tasks_df["due_date"] = pd.to_datetime(tasks_df["due_date"], errors="coerce")
    tasks_df["created_at"] = pd.to_datetime(tasks_df["created_at"], errors="coerce")
    tasks_df = tasks_df.sort_values(by=["due_date", "priority", "created_at"], ascending=[True, True, True])
    out = []
    t_ptr = start_time
    remaining = total_minutes
    for _, row in tasks_df.iterrows():
        est = int(row.get("estimate_min") or 0)
        blocks = max(1, (est + block_len - 1) // block_len)  # ceil to block_len
        for _ in range(blocks):
            if remaining < block_len:
                return out
            start = t_ptr
            end = start + dt.timedelta(minutes=block_len)
            out.append(TaskPlan(
                task_id=row["id"],
                title=row["title"],
                subject=row.get("subject_name") or "—",
                start=start,
                end=end,
                minutes=block_len,
                due_date=row["due_date"].date().isoformat() if pd.notnull(row["due_date"]) else "",
                priority=int(row["priority"]),
            ))
            t_ptr = end + dt.timedelta(minutes=short_break)
            remaining -= (block_len + short_break)
    return out

# ---------- ICS helpers ----------

def make_ics_from_plan(plan_list, tz="UTC"):
    """
    Build a VCALENDAR string for a given list of TaskPlan items.
    Times are naive and treated as local by many calendar apps; keep simple.
    """
    def dtstamp(dtobj):
        return dtobj.strftime("%Y%m%dT%H%M%S")
    now = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//StudyPlanner//EN",
    ]
    for i, p in enumerate(plan_list, start=1):
        uid = f"plan-{p.task_id}-{i}@studyplanner"
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{now}",
            f"DTSTART:{dtstamp(p.start)}",
            f"DTEND:{dtstamp(p.end)}",
            f"SUMMARY:{p.subject}: {p.title}",
            f"DESCRIPTION:Due {p.due_date} | Priority {p.priority}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)

def make_ics_from_deadlines(tasks_df):
    """Create all-day events on each task's due date."""
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//StudyPlanner//EN",
    ]
    now = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    for _, r in tasks_df.iterrows():
        if pd.isna(r.get("due_date")) or not r.get("due_date"):
            continue
        try:
            d = pd.to_datetime(r["due_date"]).date()
        except Exception:
            continue
        uid = f"deadline-{r['id']}@studyplanner"
        # All-day events use DTSTART;VALUE=DATE and DTEND next day
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{now}",
            f"DTSTART;VALUE=DATE:{d.strftime('%Y%m%d')}",
            f"DTEND;VALUE=DATE:{(d + dt.timedelta(days=1)).strftime('%Y%m%d')}",
            f"SUMMARY:📌 {r.get('subject_name') or 'Task'} due: {r['title']}",
            f"DESCRIPTION:Priority {r.get('priority', '')} | Estimate {r.get('estimate_min','')} min",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)

# ---------- UI ----------

st.set_page_config(page_title="Study Planner", page_icon="📚", layout="wide")
st.title("📚 Study Planner")

init_db()
ensure_defaults()

# -- Dashboard Summary --
colA, colB, colC, colD = st.columns(4)
tasks_df_full = df_from_rows(run_query("SELECT * FROM tasks"))
today = dt.date.today()
overdue_n = 0
if not tasks_df_full.empty:
    dd = pd.to_datetime(tasks_df_full["due_date"], errors="coerce").dt.date
    st.session_state["_due_dates"] = dd  # for internal
    overdue_n = int(((dd < today) & (tasks_df_full["status"].isin(["todo","doing"]))).sum())

with colA:
    st.metric("Total tasks", int(tasks_df_full.shape[0]))
with colB:
    st.metric("Open tasks", int((tasks_df_full["status"].isin(["todo","doing"])).sum() if not tasks_df_full.empty else 0))
with colC:
    st.metric("Overdue", overdue_n)
with colD:
    # Minutes logged this week
    sess_rows = run_query("SELECT duration_min, date FROM sessions")
    sess_df = df_from_rows(sess_rows)
    if sess_df.empty:
        st.metric("Minutes this week", 0)
    else:
        sess_df["date"] = pd.to_datetime(sess_df["date"])
        start_week = (pd.Timestamp(today) - pd.Timedelta(days=pd.Timestamp(today).dayofweek)).normalize()
        total_week = int(sess_df.loc[sess_df["date"] >= start_week, "duration_min"].sum())
        st.metric("Minutes this week", total_week)

tab_tasks, tab_plan, tab_stats = st.tabs(["Tasks", "Plan My Day", "Progress & Export"])

with st.sidebar:
    st.header("Add Subject")
    with st.form("add_subject"):
        s_name = st.text_input("Subject name")
        s_color = st.color_picker("Color", "#4F46E5")
        submitted = st.form_submit_button("Add Subject")
        if submitted and s_name.strip():
            try:
                run_query("INSERT INTO subjects(name, color) VALUES (?,?)", (s_name.strip(), s_color))
                st.success(f"Added subject: {s_name}")
            except Exception as e:
                st.warning(f"Could not add subject: {e}")

    st.header("Quick Add Task")
    subs = run_query("SELECT * FROM subjects")
    subs_df = df_from_rows(subs)
    sub_map = {row["name"]: row["id"] for _, row in subs_df.iterrows()} if not subs_df.empty else {}
    with st.form("quick_add"):
        t_title = st.text_input("Title", placeholder="e.g., Read Ch. 5")
        t_subject = st.selectbox("Subject", list(sub_map.keys()) or ["(none)"])
        t_due = st.date_input("Due date", value=today)
        col1, col2 = st.columns(2)
        with col1:
            t_est = st.number_input("Estimate (min)", 5, 600, 30, 5)
        with col2:
            t_prio = st.selectbox("Priority (1=High,3=Low)", [1,2,3], index=1)
        t_notes = st.text_area("Notes", height=60)
        add_btn = st.form_submit_button("Add Task")
        if add_btn and t_title.strip():
            sub_id = sub_map.get(t_subject)
            run_query("""INSERT INTO tasks(title, subject_id, due_date, estimate_min, priority, notes)
                         VALUES (?,?,?,?,?,?)""",
                      (t_title.strip(), sub_id, t_due.isoformat(), int(t_est), int(t_prio), t_notes))
            st.success("Task added!")

    st.header("Bulk Import")
    # Download template
    template_df = pd.DataFrame([{
        "title": "Read Ch. 5",
        "subject": "Math",
        "due_date": today.isoformat(),
        "estimate_min": 30,
        "priority": 2,
        "status": "todo",
        "notes": "Section 5.1-5.3"
    }])
    buf = io.BytesIO()
    template_df.to_csv(buf, index=False)
    st.download_button("Download CSV template", buf.getvalue(), "tasks_template.csv", "text/csv")
    # Upload
    up = st.file_uploader("Upload tasks CSV", type=["csv"])
    if up is not None:
        try:
            imp = pd.read_csv(up)
            required = {"title","subject","due_date","estimate_min","priority","status","notes"}
            if not required.issubset(set(imp.columns)):
                st.error(f"CSV must include columns: {sorted(required)}")
            else:
                # ensure subjects exist
                for subj in sorted(set(imp["subject"].dropna().astype(str))):
                    if not subj.strip():
                        continue
                    run_query("INSERT OR IGNORE INTO subjects(name) VALUES (?)", (subj.strip(),))
                # map subjects
                subs2 = df_from_rows(run_query("SELECT * FROM subjects"))
                smap2 = {row["name"]: row["id"] for _, row in subs2.iterrows()}
                # insert tasks
                for _, r in imp.iterrows():
                    sub_id = smap2.get(str(r["subject"]), None)
                    run_query("""INSERT INTO tasks(title, subject_id, due_date, estimate_min, priority, status, notes)
                                 VALUES (?,?,?,?,?,?,?)""",
                              (str(r["title"]), sub_id, str(r["due_date"]), int(r["estimate_min"]), int(r["priority"]), str(r["status"]), str(r.get("notes",""))))
                st.success(f"Imported {imp.shape[0]} tasks.")
        except Exception as e:
            st.error(f"Import failed: {e}")

with tab_tasks:
    st.subheader("Your Tasks")
    q = """
    SELECT t.*, s.name AS subject_name, s.color as subject_color
    FROM tasks t LEFT JOIN subjects s ON t.subject_id = s.id
    """
    tasks = df_from_rows(run_query(q))
    if tasks.empty:
        st.info("No tasks yet. Use the sidebar to add some or bulk import.")
    else:
        colf1, colf2, colf3, colf4 = st.columns(4)
        with colf1:
            status_filter = st.multiselect("Status", options=["todo","doing","done"], default=["todo","doing"])
        with colf2:
            subject_filter = st.multiselect("Subject", options=sorted(tasks["subject_name"].fillna("—").unique().tolist()))
        with colf3:
            prio_filter = st.multiselect("Priority", options=[1,2,3])
        with colf4:
            due_until = st.date_input("Due until", value=None)

        df = tasks.copy()
        if status_filter:
            df = df[df["status"].isin(status_filter)]
        if subject_filter:
            df = df[df["subject_name"].fillna("—").isin(subject_filter)]
        if prio_filter:
            df = df[df["priority"].isin(prio_filter)]
        if due_until:
            df = df[pd.to_datetime(df["due_date"]) <= pd.to_datetime(due_until)]

        # Overdue highlighting
        df_display = df.copy()
        dued = pd.to_datetime(df_display["due_date"], errors="coerce").dt.date
        overdue_mask = (dued < today) & (df_display["status"].isin(["todo","doing"]))
        df_display["overdue"] = overdue_mask.map({True:"⚠️", False:""})
        df_display["due_date"] = pd.to_datetime(df_display["due_date"]).dt.date.astype("string")
        df_display = df_display[["id","overdue","title","subject_name","due_date","estimate_min","priority","status","notes"]]\
            .rename(columns={
                "subject_name":"subject",
                "estimate_min":"mins",
            })
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("### Edit Selected Task")
        selected_id = st.selectbox("Pick a task", df_display["id"].tolist())
        row = df[df["id"] == selected_id].iloc[0]
        subs = run_query("SELECT * FROM subjects")
        subs_df = df_from_rows(subs)
        sub_keys = list(subs_df["name"]) if not subs_df.empty else []
        with st.form("edit_task"):
            e_title = st.text_input("Title", value=row["title"])
            e_subject = st.selectbox("Subject", sub_keys or ["(none)"], index=(sub_keys.index(row["subject_name"]) if row["subject_name"] in sub_keys else 0))
            e_due = st.date_input("Due date", value=pd.to_datetime(row["due_date"]).date() if pd.notnull(row["due_date"]) else today)
            c1, c2, c3 = st.columns(3)
            with c1:
                e_est = st.number_input("Estimate (min)", 5, 600, int(row["estimate_min"] or 30), 5)
            with c2:
                e_prio = st.selectbox("Priority", [1,2,3], index=[1,2,3].index(int(row["priority"])) if row["priority"] in [1,2,3] else 1)
            with c3:
                e_status = st.selectbox("Status", ["todo","doing","done"], index=["todo","doing","done"].index(row["status"]) if row["status"] in ["todo","doing","done"] else 0)
            e_notes = st.text_area("Notes", value=row["notes"] or "", height=80)
            colu1, colu2, colu3 = st.columns(3)
            save_btn = colu1.form_submit_button("Save")
            del_btn = colu2.form_submit_button("Delete")
            if save_btn:
                sub_id = int(subs_df.loc[subs_df["name"] == e_subject, "id"].iloc[0]) if not subs_df.empty and e_subject in list(subs_df["name"]) else None
                run_query("""UPDATE tasks SET title=?, subject_id=?, due_date=?, estimate_min=?, priority=?, status=?, notes=? WHERE id=?""",
                          (e_title.strip(), sub_id, e_due.isoformat(), int(e_est), int(e_prio), e_status, e_notes, int(selected_id)))
                st.success("Updated.")
            if del_btn:
                run_query("DELETE FROM tasks WHERE id=?", (int(selected_id),))
                st.success("Deleted. Reload the page to refresh list.")

with tab_plan:
    st.subheader("Plan My Day")
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        start_time = st.time_input("Start time", value=dt.datetime.now().time().replace(second=0, microsecond=0))
    with colp2:
        available_min = st.number_input("Available study time today (minutes)", 25, 600, 180, 5)
    with colp3:
        block_len = st.slider("Pomodoro length (minutes)", 20, 60, 25, 5)

    todo_rows = run_query("""
        SELECT t.*, s.name AS subject_name
        FROM tasks t LEFT JOIN subjects s ON t.subject_id = s.id
        WHERE t.status != 'done'
    """)
    todo_df = df_from_rows(todo_rows)
    if todo_df.empty:
        st.info("No remaining tasks. Add some tasks first.")
    else:
        st.caption("Uses due date ➜ priority to choose tasks.")

        start_dt = dt.datetime.combine(today, start_time)
        plan = plan_day(todo_df, start_dt, total_minutes=int(available_min), block_len=int(block_len))
        if not plan:
            st.warning("Not enough time to schedule any blocks. Increase available minutes.")
        else:
            plan_df = pd.DataFrame([{
                "start": p.start.strftime("%H:%M"),
                "end": p.end.strftime("%H:%M"),
                "subject": p.subject,
                "task": p.title,
                "mins": p.minutes,
                "due": p.due_date,
                "priority": p.priority,
                "task_id": p.task_id
            } for p in plan])
            st.dataframe(plan_df.drop(columns=["task_id"]), use_container_width=True, hide_index=True)

            # ICS download for today's plan
            ics_text = make_ics_from_plan(plan)
            st.download_button("📆 Download today's plan (.ics)", ics_text.encode("utf-8"), file_name=f"study_plan_{today.isoformat()}.ics", mime="text/calendar")

            st.markdown("#### Log Completed Block")
            chosen_idx = st.number_input("Row # you completed (starting from 1)", 1, len(plan_df), 1, 1)
            notes = st.text_input("Notes (optional)", "")
            log_btn = st.button("Log this block")
            if log_btn:
                rec = plan_df.iloc[int(chosen_idx)-1]
                task_id = int(rec["task_id"])
                subj_id = run_query("SELECT subject_id FROM tasks WHERE id=?", (task_id,), fetch="one")["subject_id"]
                run_query("""INSERT INTO sessions(task_id, subject_id, date, start_time, end_time, duration_min, notes)
                             VALUES (?,?,?,?,?,?,?)""",
                          (task_id, subj_id, today.isoformat(), rec["start"], rec["end"], int(rec["mins"]), notes))
                st.success("Logged! You can repeat as you complete more blocks.")
                if st.checkbox("Mark task as done as well?"):
                    run_query("UPDATE tasks SET status='done' WHERE id=?", (task_id,))
                    st.success("Task marked done.")

with tab_stats:
    st.subheader("Progress & Export")
    sess_rows = run_query("""
        SELECT s.*, t.title as task_title, sub.name as subject_name
        FROM sessions s
        LEFT JOIN tasks t ON s.task_id = t.id
        LEFT JOIN subjects sub ON s.subject_id = sub.id
        ORDER BY date DESC, start_time DESC
    """)
    sessions_df = df_from_rows(sess_rows)
    if sessions_df.empty:
        st.info("No sessions logged yet.")
    else:
        st.dataframe(sessions_df[["date","start_time","end_time","duration_min","subject_name","task_title","notes"]]
                     .rename(columns={"subject_name":"subject","task_title":"task"}),
                     use_container_width=True, hide_index=True)

        st.markdown("#### Weekly Summary")
        sessions_df["date"] = pd.to_datetime(sessions_df["date"])
        weekly = (sessions_df
                  .assign(week=sessions_df["date"].dt.to_period("W").astype(str))
                  .groupby(["week","subject_id"])
                  .agg(total_min=("duration_min","sum"))
                  .reset_index())
        subs = df_from_rows(run_query("SELECT * FROM subjects"))
        if not weekly.empty and not subs.empty:
            weekly = weekly.merge(subs[["id","name"]], left_on="subject_id", right_on="id", how="left")
            weekly = weekly[["week","name","total_min"]].rename(columns={"name":"subject"})
            st.bar_chart(weekly.pivot_table(index="week", columns="subject", values="total_min", aggfunc="sum").fillna(0))

    st.markdown("#### Export Data")
    colx1, colx2, colx3 = st.columns(3)
    with colx1:
        if st.button("Export Tasks CSV"):
            tasks_all = df_from_rows(run_query("SELECT * FROM tasks"))
            csv = tasks_all.to_csv(index=False).encode("utf-8")
            st.download_button("Download tasks.csv", csv, "tasks.csv", "text/csv", use_container_width=True)
    with colx2:
        if st.button("Export Sessions CSV"):
            sess_all = df_from_rows(run_query("SELECT * FROM sessions"))
            csv = sess_all.to_csv(index=False).encode("utf-8")
            st.download_button("Download sessions.csv", csv, "sessions.csv", "text/csv", use_container_width=True)
    with colx3:
        # ICS of deadlines
        tasks_with_subjects = df_from_rows(run_query("""
            SELECT t.*, s.name AS subject_name
            FROM tasks t LEFT JOIN subjects s ON t.subject_id = s.id
        """))
        ics_deadlines = make_ics_from_deadlines(tasks_with_subjects)
        st.download_button("📆 Download all deadlines (.ics)", ics_deadlines.encode("utf-8"), file_name="study_deadlines.ics", mime="text/calendar")
        
st.caption("Tip: Back up study_planner.db if you switch computers.")
