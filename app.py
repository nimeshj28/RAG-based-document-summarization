import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
import tempfile

# Page Configuration
st.set_page_config(page_title="RAG PDF Summarizer Chatbot")

st.title("📄 RAG-based PDF Summarizer & Chatbot")
st.markdown("""
Upload your PDF, get a summary (5–10 key points), and then chat with your document.  
**Questions outside the PDF scope will be denied.**
""")

# Sidebar - API Key Input
openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")

# File upload
uploaded_file = st.file_uploader("📤 Upload a PDF", type="pdf")

if uploaded_file and openai_api_key:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # 1. Load PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # 2. Split document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_chunks = splitter.split_documents(docs)

    # 3. Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(doc_chunks, embeddings)

    # 4. Summarize PDF
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = summary_chain.run(doc_chunks)

    st.subheader("📌 Summary (5–10 Key Points)")
    for i, point in enumerate(summary.split("\n"), 1):
        if point.strip():
            st.write(f"{i}. {point.strip()}")

    # 5. Create Retrieval-based QA system
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type="stuff")

    # 6. Chat Interface
    st.subheader("💬 Chat with your Document")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Start prompt
    if not st.session_state["history"]:
        st.session_state["history"].append({
            "user": None,
            "bot": "What do you want to know?"
        })

    user_query = st.chat_input("Type your question here...")

    def answer_query(query):
        result = qa_chain.run(query)
        # Block if out of scope or weak response
        if (
            "I don't know" in result 
            or len(result.strip()) < 5 
            or "not found" in result.lower()
        ):
            return "Request to access same should be denied as response."
        return result

    if user_query:
        answer = answer_query(user_query)
        st.session_state["history"].append({"user": user_query, "bot": answer})

    # 7. Display chat history
    for chat in st.session_state["history"]:
        if chat["user"]:
            st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

elif not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
elif not uploaded_file:
    st.info("Please upload a PDF file to continue.")
