import streamlit as st
import pandas as pd
from database.postgres_handler import PostgresHandler

# --- Auth check ---
if "logged_in_prof" not in st.session_state:
    st.warning("Please login first to access this page.", icon="🔒")
    st.stop()

prof = st.session_state["logged_in_prof"]
my_email = prof.get("university_email", "")

st.set_page_config(page_title="🔗 Shared with Me", layout="wide")
st.title("🔗 Results Shared With Me")

# Fetch shared results
handler = PostgresHandler()
records = handler.fetch_shared_with_me(my_email)

df = pd.DataFrame(records)
if df.empty:
    st.info("No grading results have been shared with you.")
    st.stop()

# Clean up fields
# Assuming df has same structure as grading_results
# Convert datetime
if 'created_at' in df.columns:
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# Sidebar filters
st.sidebar.header("🔍 Filters")
opts = lambda c: ["All"] + sorted(df[c].dropna().unique())
selected_student  = st.sidebar.selectbox("Student", opts('student_id'))
selected_course   = st.sidebar.selectbox("Course", opts('course'))
selected_semester = st.sidebar.selectbox("Semester", opts('semester'))
selected_language = st.sidebar.selectbox("Language", opts('language'))

mask = pd.Series(True, index=df.index)
for field, sel in [("student_id",selected_student), ("course",selected_course), ("semester",selected_semester), ("language",selected_language)]:
    if sel != "All":
        mask &= (df[field] == sel)
filtered = df[mask]
if filtered.empty:
    st.warning("No shared results match the selected filters.")
    st.stop()

# Summary table
st.subheader("Shared Results Summary")
summary = (
    filtered.groupby(['student_id', 'course', 'semester'])
            .agg(
                avg_score=('new_score', lambda x: sum(map(float, x))/len(x)),
                count=('new_score','count')
            )
            .reset_index()
)
st.table(summary)

# Detailed view by row
st.subheader("Detailed Shared Results")
st.dataframe(
    filtered[['student_id','course','semester','assignment_no','question','old_score','new_score','old_feedback','new_feedback','language','created_at']]
    .sort_values('created_at', ascending=False)
)

# Download
csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button("📅 Download CSV", data=csv, file_name="shared_results.csv", mime="text/csv")
