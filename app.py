import streamlit as st
import PyPDF2
import re
import nltk

from sentence_transformers import SentenceTransformer, util

# Load model (this is the key upgrade)
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------
# Extract PDF text
# ----------------------------
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ----------------------------
# Clean text
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text

# ----------------------------
# Semantic similarity
# ----------------------------
def semantic_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)
    return float(score)

# ----------------------------
# Keyword extraction (important only)
# ----------------------------
def extract_keywords(text):
    words = text.split()
    keywords = [w for w in words if len(w) > 4]
    return set(keywords)

# ----------------------------
# Missing keywords
# ----------------------------
def missing_keywords(resume, jd):
    r = extract_keywords(resume)
    j = extract_keywords(jd)
    return list(j - r)

# ----------------------------
# ATS Score (improved)
# ----------------------------
def ats_score(text):
    score = 0

    if "skills" in text: score += 15
    if "project" in text: score += 15
    if "education" in text: score += 10
    if "python" in text: score += 15
    if "data structures" in text: score += 15
    if "sql" in text: score += 10
    if "html" in text: score += 5
    if "css" in text: score += 5
    if "javascript" in text: score += 5

    return score
def extract_sections(text):
    sections = {
        "skills": "",
        "experience": "",
        "education": "",
        "projects": ""
    }

    text = text.lower()

    # Simple keyword-based splitting
    if "skills" in text:
        sections["skills"] = text.split("skills")[-1][:400]

    if "experience" in text:
        sections["experience"] = text.split("experience")[-1][:400]

    if "education" in text:
        sections["education"] = text.split("education")[-1][:400]

    if "project" in text:
        sections["projects"] = text.split("project")[-1][:400]

    return sections
# ----------------------------
# UI
# ----------------------------
st.set_page_config(layout="wide")
st.title("🚀 Intelligent Resume Screening System")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_desc = st.text_area("Paste Job Description")

if st.button("Analyze"):

    if uploaded_file and job_desc:

        resume_raw = extract_text(uploaded_file)

        resume = clean_text(resume_raw)
        jd = clean_text(job_desc)

        # 🔥 NEW: semantic score
        score = semantic_similarity(resume, jd)

        st.subheader("📊 Overall Match Score")
        st.progress(score)
        st.write(f"{round(score * 100, 2)} %")
        # Section-wise analysis
        st.subheader("📌 Section-wise Analysis")

        sections = extract_sections(resume_raw)

        for sec, content in sections.items():
            if content.strip():
              sec_clean = clean_text(content)
              sec_score = semantic_similarity(sec_clean, jd)
              st.write(f"{sec.capitalize()} Match: {round(sec_score * 100, 2)}%")
        # Missing keywords
        st.subheader("❌ Missing Keywords")
        missing = missing_keywords(resume, jd)
        st.write(missing[:15])

        # ATS Score
        st.subheader("🤖 ATS Score")
        ats = ats_score(resume)
        st.progress(ats / 100)
        st.write(f"{ats}%")

        # Explanation
        st.subheader("🧠 Score Explanation")
        if score > 0.75:
            st.success("Strong match")
        elif score > 0.5:
            st.warning("Moderate match")
        else:
            st.error("Low match")

        # Suggestions
        st.subheader("💡 Suggestions")
        st.write("• Add missing technical keywords")
        st.write("• Include more project details")
        st.write("• Use measurable achievements")

    else:
        st.warning("Upload resume and enter job description")
