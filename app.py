import streamlit as st
from recommender import JobRecommender
from utils import extract_text_from_pdf
import pandas as pd

st.set_page_config(page_title="🚀 Smart Job Recommender", layout="wide")
st.title("🚀 Smart Job Recommendation System")

@st.cache_resource
def load_recommender():
    return JobRecommender("postings.csv")

recommender = load_recommender()

# --- Sidebar ---
st.sidebar.header("🔧 Filters & Resume")
location_filter = st.sidebar.text_input("📍 Location Filter")
type_filter = st.sidebar.text_input("📝 Employment Type (Full-Time, Internship...)")
top_n = st.sidebar.slider("📊 Number of Recommendations", 3, 20, 5)

resume_file = st.sidebar.file_uploader("📄 Upload Resume (PDF or TXT)", type=['pdf', 'txt'])
resume_text = ""

if resume_file is not None:
    if resume_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(resume_file)
    else:
        resume_text = resume_file.read().decode("utf-8")

    st.sidebar.success("✅ Resume loaded successfully!")

# --- Main Content ---
tab1, tab2 = st.tabs(["🔍 Search by Keywords", "📄 Match from Resume"])

# Tab 1 - Search by keywords
with tab1:
    query = st.text_input("Enter job title, skills or keywords")
    if query:
        results = recommender.recommend(query, top_n=top_n, location_filter=location_filter, type_filter=type_filter)
        st.markdown("### 🌟 Jobs Matching Your Search")
        for _, row in results.iterrows():
            st.subheader(f"{row['title']} at {row['company_name']}")
            st.text(f"📍 {row['location']} | 📝 {row['formatted_work_type']} | 🔗 Score: {row['similarity_score']:.2f}")
            st.write(row['description'][:500] + "...")
            st.markdown(f"[🔗 Apply Here]({row['job_posting_url']})")
            st.markdown("---")

# Tab 2 - Match from uploaded resume
with tab2:
    if resume_text:
        st.markdown("### 📄 Jobs Matching Your Resume")
        results = recommender.recommend_from_resume(resume_text, top_n=top_n)
        for _, row in results.iterrows():
            st.subheader(f"{row['title']} at {row['company_name']}")
            st.text(f"📍 {row['location']} | 📝 {row['formatted_work_type']} | 🔗 Score: {row['similarity_score']:.2f}")
            st.write(row['description'][:500] + "...")
            st.markdown(f"[🔗 Apply Here]({row['job_posting_url']})")
            st.markdown("---")
    else:
        st.info("Upload your resume from the sidebar to see matched jobs.")
