import streamlit as st
import PyPDF2
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords

# ----------------------------
# Extract text from PDF
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
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ----------------------------
# Section extraction (simple)
# ----------------------------
def extract_sections(text):
    sections = {
        "skills": "",
        "experience": "",
        "education": "",
        "projects": ""
    }

    text_lower = text.lower()

    if "skills" in text_lower:
        sections["skills"] = text_lower.split("skills")[-1][:300]
    if "experience" in text_lower:
        sections["experience"] = text_lower.split("experience")[-1][:300]
    if "education" in text_lower:
        sections["education"] = text_lower.split("education")[-1][:300]
    if "projects" in text_lower:
        sections["projects"] = text_lower.split("projects")[-1][:300]

    return sections

# ----------------------------
# Similarity calculation
# ----------------------------
def get_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

# ----------------------------
# Missing keywords
# ----------------------------
def get_missing(resume, jd):
    return list(set(jd.split()) - set(resume.split()))

# ----------------------------
# ATS Score
# ----------------------------
def ats_score(resume):
    score = 0

    if len(resume.split()) > 300:
        score += 30
    if "experience" in resume:
        score += 20
    if "skills" in resume:
        score += 20
    if "project" in resume:
        score += 15
    if "education" in resume:
        score += 15

    return score

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("🚀 Intelligent Resume Screening System")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_desc = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if uploaded_file and job_desc:

        resume_raw = extract_text(uploaded_file)

        clean_resume = clean_text(resume_raw)
        clean_jd = clean_text(job_desc)

        # Overall score
        overall_score = get_similarity(clean_resume, clean_jd)

        # Sections
        sections = extract_sections(resume_raw)

        st.subheader("📊 Overall Match Score")
        st.progress(overall_score)
        st.write(f"{round(overall_score*100,2)} %")

        # Section-wise
        st.subheader("📌 Section-wise Analysis")
        for sec, content in sections.items():
            if content:
                score = get_similarity(clean_text(content), clean_jd)
                st.write(f"{sec.capitalize()} Match: {round(score*100,2)}%")

        # Missing skills
        st.subheader("❌ Missing Keywords")
        missing = get_missing(clean_resume, clean_jd)
        st.write(missing[:20])

        # ATS Score
        st.subheader("🤖 ATS Compatibility Score")
        ats = ats_score(resume_raw.lower())
        st.progress(ats/100)
        st.write(f"{ats}%")

        # Explanation
        st.subheader("🧠 Score Explanation")
        if overall_score > 0.7:
            st.success("Strong match with job description.")
        elif overall_score > 0.4:
            st.warning("Moderate match. Improve keywords.")
        else:
            st.error("Low match. Add relevant skills.")

        # Suggestions
        st.subheader("💡 Suggestions")
        st.write("• Add missing keywords")
        st.write("• Include measurable achievements")
        st.write("• Use action verbs like Developed, Built, Optimized")

    else:
        st.warning("Please upload resume and job description.")