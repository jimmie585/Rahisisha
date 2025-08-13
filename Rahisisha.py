import streamlit as st
import json
from pathlib import Path
from hashlib import sha256
from PyPDF2 import PdfReader
from docx import Document as DocxReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import requests

# -------------------- Streamlit Config --------------------

st.set_page_config(page_title="Rahisisha", layout="wide")
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- User Auth Functions --------------------

def load_users():
    if Path("users.json").exists():
        with open("users.json", "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

def hash_password(password):
    return sha256(password.encode()).hexdigest()

def authenticate(username, password):
    users = load_users()
    return username in users and users[username]["password"] == hash_password(password)

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = {"password": hash_password(password)}
    save_users(users)
    return True

# -------------------- Document Handling --------------------

def load_document(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file_path.endswith(".docx"):
        doc = DocxReader(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = ""
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.create_documents([text])

def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embeddings)
    return vectordb

def ask_mistral(context, question):
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

def process_uploads(upload_dir):
    all_docs = []
    for file in Path(upload_dir).glob("*"):
        text = load_document(str(file))
        docs = split_text(text)
        all_docs.extend(docs)
    return all_docs

# -------------------- UI: Login & Signup --------------------

def login_ui():
    st.title("🔐 Rahisisha - Founder Access")
    tab1, tab2 = st.tabs(["Log In", "Create Account"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log In"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Logged in successfully")
            else:
                st.error("Invalid credentials")

    with tab2:
        new_user = st.text_input("New Username", key="new_user")
        new_pass = st.text_input("New Password", type="password", key="new_pass")
        if st.button("Create Account"):
            if register_user(new_user, new_pass):
                st.success("Account created. You can now log in.")
            else:
                st.warning("Username already exists.")

# -------------------- UI: Main Dashboard --------------------

def dashboard():
    st.sidebar.title("Rahisisha Menu")
    page = st.sidebar.radio("Navigate", ["🏠 Upload & Chat", "📄 Uploaded Docs", "🔓 Log Out"])

    username = st.session_state.username
    user_folder = Path("uploads") / username
    user_folder.mkdir(parents=True, exist_ok=True)

    if page == "🏠 Upload & Chat":
        st.header("📤 Upload Company Docs & Ask Questions")

        company_name = st.text_input("Company Name")
        uploaded_files = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=True)

        if st.button("Upload"):
            if not company_name or not uploaded_files:
                st.warning("Please fill all fields.")
            else:
                company_dir = user_folder / company_name.replace(" ", "_")
                company_dir.mkdir(parents=True, exist_ok=True)
                for file in uploaded_files:
                    with open(company_dir / file.name, "wb") as f:
                        f.write(file.read())
                st.success("Documents uploaded!")

        st.divider()
        st.subheader("🤖 Ask a Question")
        question = st.text_input("What do you want to know about your company documents?")
        if question and company_name:
            company_dir = user_folder / company_name.replace(" ", "_")
            if company_dir.exists():
                with st.spinner("Processing..."):
                    docs = process_uploads(company_dir)
                    vectordb = create_vectorstore(docs)
                    retriever = vectordb.as_retriever()
                    results = retriever.get_relevant_documents(question)
                    context = "\n".join([doc.page_content for doc in results])
                    answer = ask_mistral(context, question)
                    st.success("Answer:")
                    st.write(answer)
            else:
                st.error("No documents found.")

    elif page == "📄 Uploaded Docs":
        st.header("📁 Your Uploaded Companies")
        if user_folder.exists():
            companies = [d.name for d in user_folder.iterdir() if d.is_dir()]
            if companies:
                for company in companies:
                    st.markdown(f"- **{company}**")
            else:
                st.info("You haven't uploaded any documents yet.")
        else:
            st.info("No uploads yet.")

    elif page == "🔓 Log Out":
        st.session_state.logged_in = False
        st.experimental_rerun()

# -------------------- Run App --------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_ui()
else:
    dashboard()
