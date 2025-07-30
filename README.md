# pdf-summarizer-langchain
A Streamlit web app to summarize PDF documents using Groq-hosted LLMs via LangChain. Supports LLaMA 3, Mixtral, and Gemma models.


🦜📄 LLM-Powered PDF Summarizer
This Streamlit web app allows users to upload a PDF file and generate a concise summary using LLMs hosted on Groq (like LLaMA 3, Mixtral, and Gemma) via the LangChain framework.

🔍 Features
Upload and process PDF files of any length

Automatically split the PDF into manageable chunks

Generate accurate summaries using map_reduce strategy

Choose from powerful LLMs: llama3-8b-8192, mixtral-8x7b-32768, and gemma-7b-it

Secure integration using your own Groq API key

Simple, elegant user interface built with Streamlit

🧠 How It Works
Upload PDF ➝ The app uses PyPDFLoader to read and split the PDF.

Chunking ➝ Text is split using RecursiveCharacterTextSplitter (chunk size = 1000 chars).

Summarization ➝ Each chunk is summarized using a Groq-hosted LLM.

Combining ➝ Individual summaries are merged into a final summary using map_reduce.

📦 Tech Stack
Frontend/UI: Streamlit

LLM Interface: LangChain

Models via Groq: llama3, mixtral, gemma

PDF Parsing: PyPDFLoader from LangChain

🛠️ Installation
1. Clone the Repo

git clone https://github.com/yourusername/pdf-summarizer
cd pdf-summarizer

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

3. Install Dependencies
pip install -r requirements.txt


Sample requirements.txt:

streamlit
langchain
langchain_groq
pypdf

pip install faiss-cpu tiktoken
