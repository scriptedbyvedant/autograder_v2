# File: pages/3_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from database.postgres_handler import PostgresHandler
import unicodedata

# --- Auth check ---
if "logged_in_prof" not in st.session_state:
    st.warning("Please login first to access this page.", icon="🔒")
    st.stop()

prof         = st.session_state["logged_in_prof"]
professor_id = prof.get("professor", "")
my_email     = prof.get("university_email", "")

def clean(val):
    if pd.isna(val) or str(val).strip().lower() in ["", "none", "unknown"]:
        return "Unknown"
    return val

def normalize_text(val):
    if not isinstance(val, str):
        return ""
    val = val.strip().lower()
    val = unicodedata.normalize('NFKD', val)
    return ''.join(c for c in val if not unicodedata.combining(c))

def normalize_language(lang):
    if not isinstance(lang, str):
        return "Unknown"
    lang = lang.strip().lower()
    mapping = {
        "english": "English", "en": "English",
        "german":  "German",  "de": "German", "deutsch": "German",
        "spanish": "Spanish","es": "Spanish","español": "Spanish"
    }
    return mapping.get(lang, lang.capitalize() if lang else "Unknown")

def semester_sorter(sem):
    try:
        season_map = {"spring":0, "summer":1, "fall":2, "winter":3}
        parts = sem.strip().split()
        year = int(parts[-1])
        season = season_map.get(parts[0].lower(), 4)
        return year*10 + season
    except:
        return 999999

def main():
    st.set_page_config(page_title="📊 Analytics Dashboard", layout="wide")
    st.title("📈 Grading Analytics Dashboard")

    handler   = PostgresHandler()

    # 1️⃣ Fetch only this professor's results + those shared with them
    my_df     = pd.DataFrame(handler.fetch_my_results(professor_id))
    shared_df = pd.DataFrame(handler.fetch_shared_with_me(my_email))

    # Tag ownership
    if not my_df.empty:
        my_df["__owner__"] = "You"
    if not shared_df.empty:
        shared_df["__owner__"] = "Shared"

    # Merge
    df = pd.concat([my_df, shared_df], ignore_index=True, sort=False)
    if df.empty:
        st.warning("No grading data available (yours or shared).")
        return

    # 2️⃣ Clean & prepare fields
    for col in ["course", "semester", "assignment_no", "student_id", "question"]:
        df[col] = df[col].apply(clean)
    df["language"] = df["language"].apply(normalize_language)
    df["score"]    = pd.to_numeric(df["new_score"], errors="coerce").fillna(0)
    df["semester"] = df["semester"].astype(str)

    # two tabs: My vs Shared
    tab_you, tab_shared = st.tabs(["My Results", "Shared With Me"])
    for owner_label, tab in [("You", tab_you), ("Shared", tab_shared)]:
        with tab:
            sub = df[df["__owner__"] == owner_label]
            if sub.empty:
                st.info(f"No records under **{owner_label}**.")
                continue

            # 3️⃣ Sidebar filters
            st.sidebar.header("🔍 Filters")
            opts = lambda c: ["All"] + sorted(df[c].unique())
            selected_course     = st.sidebar.selectbox("Course", opts("course"), key=f"c_{owner_label}")
            selected_semester   = st.sidebar.selectbox("Semester", opts("semester"), key=f"s_{owner_label}")
            selected_assignment = st.sidebar.selectbox("Assignment", opts("assignment_no"), key=f"a_{owner_label}")
            selected_student    = st.sidebar.selectbox("Student", opts("student_id"), key=f"st_{owner_label}")
            selected_language   = st.sidebar.selectbox("Language", opts("language"), key=f"l_{owner_label}")

            mask = pd.Series(True, index=sub.index)
            for field, sel in [
                ("course", selected_course),
                ("semester", selected_semester),
                ("assignment_no", selected_assignment),
                ("student_id", selected_student),
                ("language", selected_language),
            ]:
                if sel != "All":
                    mask &= (sub[field] == sel)

            filtered = sub[mask]
            if filtered.empty:
                st.warning("No data matches the selected filters.")
                continue

            # 4️⃣ Summary cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📄 Submissions", len(filtered))
            c2.metric("💯 Avg Score", f"{filtered['score'].mean():.2f}")
            c3.metric("👥 Students", filtered["student_id"].nunique())
            c4.metric("🌐 Source", owner_label)
            st.divider()

            # ── Share Entire Dashboard ────────────────────────────────────
            with st.expander("🔗 Share Entire Dashboard View", expanded=False):
                target = st.text_input(f"Colleague’s email ({owner_label})", key=f"share_dashboard_email_{owner_label}")
                if st.button(f"Share Dashboard ({owner_label})", key=f"share_dashboard_btn_{owner_label}"):
                    if not target:
                        st.error("Enter a valid email to share with.")
                    else:
                        ids_to_share = filtered["id"].astype(int).unique().tolist()
                        for rid in ids_to_share:
                            handler.share_result(my_email, target, rid)
                        st.success(f"Shared {len(ids_to_share)} records with {target}.")

            # 5️⃣ Bar Chart: Avg Score by Course & Language
            st.subheader("1️⃣ Avg Score by Course & Language")
            bar_df = (
                filtered.groupby(["course","language"])
                        .score.mean()
                        .reset_index(name="Avg Score")
            )
            fig_bar = px.bar(
                bar_df,
                x="course", y="Avg Score", color="language",
                text="Avg Score", barmode="group",
                labels={"course":"Course","Avg Score":"Avg Score","language":"Language"}
            )
            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{owner_label}")

            # 6️⃣ Histogram: Score Distribution
            st.subheader("2️⃣ Score Distribution")
            fig_hist = px.histogram(
                filtered,
                x="score",
                nbins=10,
                labels={"score":"Score"},
                title="Distribution of All Scores"
            )
            st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{owner_label}")

            # 7️⃣ Trend: Avg Score by Semester
            st.subheader("3️⃣ Trend: Avg Score by Semester")
            trend = (
                filtered.groupby("semester", as_index=False)
                        .score.mean()
                        .rename(columns={"score":"Avg Score"})
            )
            trend["sort_key"] = trend["semester"].apply(semester_sorter)
            trend = trend.sort_values("sort_key")
            fig_line = px.line(
                trend, x="semester", y="Avg Score", markers=True,
                labels={"semester":"Semester","Avg Score":"Avg Score"}
            )
            st.plotly_chart(fig_line, use_container_width=True, key=f"line_{owner_label}")

            # 8️⃣ Top & Bottom Performers
            st.subheader("4️⃣ Top & Bottom Performers")
            stud = (
                filtered.groupby("student_id")
                        .agg({"score":["mean","count"]})
            )
            stud.columns = ["Avg Score","Submissions"]
            stud = stud.reset_index()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top 5 Students**")
                st.table(stud.nlargest(5, "Avg Score").reset_index(drop=True))
            with col2:
                st.markdown("**Bottom 5 Students**")
                st.table(stud.nsmallest(5, "Avg Score").reset_index(drop=True))

            # 9️⃣ Questions Performance
            qdf = (
                filtered.groupby("question")
                        .agg({"score":["mean","count"]})
            )
            qdf.columns = ["Avg Score","Attempts"]
            qdf = qdf.reset_index()
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Easiest 5 Questions**")
                st.table(qdf.nlargest(5, "Avg Score").reset_index(drop=True))
            with col4:
                st.markdown("**Toughest 5 Questions**")
                st.table(qdf.nsmallest(5, "Avg Score").reset_index(drop=True))

            # 🔟 Raw data viewer & CSV download
            with st.expander("📄 View Filtered Data"):
                cols = [
                    "__owner__", "student_id", "course", "semester", "assignment_no",
                    "question", "score", "old_score", "new_score",
                    "old_feedback", "new_feedback", "language", "created_at"
                ]
                st.dataframe(filtered[cols].sort_values("created_at", ascending=False))

            st.download_button(
                "📅 Download CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name=f"grading_data_{owner_label.lower()}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
