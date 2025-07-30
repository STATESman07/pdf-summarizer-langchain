import tempfile

import streamlit as st
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# ---------- Helper Functions ----------


def save_uploadedfile(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(uploaded_file.read())
    temp_file.flush()
    return temp_file.name


def load_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)


def get_groq_llm(api_key, model_name):
    return ChatGroq(api_key=api_key, model_name=model_name)


# ---------- Streamlit App Setup ----------

st.set_page_config(page_title="PDF Summarizer using LangChain 🦜", layout="wide")
st.title("📄 LLM-Powered PDF Summarizer")
st.markdown("Upload a PDF file and get a summarized version using Groq-hosted LLMs.")

# ---------- Sidebar: API Key & PDF Upload ----------

with st.sidebar:
    st.header("🔧 Settings")
    st.subheader("🔐 API Key")

    # Session state to track key validation
    if "is_valid_key" not in st.session_state:
        st.session_state.is_valid_key = False

    groq_api_key = st.text_input(
        "Enter your Groq API Key", type="password", placeholder="gsk_..."
    )

    if st.button("✅ Validate API Key"):
        if groq_api_key.startswith("gsk_"):
            st.success("API key looks valid.")
            st.session_state.is_valid_key = True
        else:
            st.error("Invalid API key format. Must start with `gsk_`.")
            st.session_state.is_valid_key = False

    # If API key is valid, show upload & model selection
    if st.session_state.is_valid_key:
        st.subheader("📄 Upload PDF & Choose Model")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        models_lst = ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]

        if uploaded_file is not None:
            selected_model = st.selectbox("Choose LLM Model", models_lst)
        else:
            st.warning("📤 Please upload a PDF to enable summarization.")

# ---------- Main Area: PDF Summarization ----------
if st.button("✨ Summarize"):

    # Prompt templates
    chunk_prompt = """You are an expert summarizer.\n\nSummarize the following text in 3-4 sentences:\n\n{text}"""
    final_prompt = """You are a document analyst.\n\nBelow are summaries of several sections of a larger document. Combine them into a single, concise summary:\n\n{text}"""

    chunk_prompt_template = PromptTemplate(
        input_variables=["text"], template=chunk_prompt
    )
    final_prompt_template = PromptTemplate(
        input_variables=["text"], template=final_prompt
    )

    # Process and summarize
    with st.spinner("🔄 Processing and summarizing PDF..."):
        temp_path = save_uploadedfile(uploaded_file)
        chunks = load_split_pdf(temp_path)
        st.success(f"✅ PDF split into {len(chunks)} chunks.")

        llm = get_groq_llm(groq_api_key, selected_model)
        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=chunk_prompt_template,
            combine_prompt=final_prompt_template,
            verbose=True,
        )

        summary = summary_chain.run(chunks)

    # Display result in main area
    st.subheader("📝 Final Summary")
    st.write(summary)
