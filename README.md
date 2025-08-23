
# Study Planner (Streamlit + SQLite)

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run study_planner_app.py
```

## Features
- Create subjects and tasks (due date, estimate, priority, notes)
- Filter & edit tasks with **overdue highlighting**
- **Plan My Day** (Pomodoro schedule) + **download .ics**
- Log completed blocks; **weekly progress chart**
- **Bulk import CSV** and export tasks/sessions CSV
- **Download deadlines .ics** to add to your calendar

## CSV template columns
title, subject, due_date (YYYY-MM-DD), estimate_min, priority (1-3), status (todo/doing/done), notes

## Ideas to extend
- Calendar sync (Google/Outlook via API credentials)
- Repeating tasks & spaced repetition
- Notifications for upcoming deadlines
