import streamlit as st
import tempfile
import os

from pydantic import SecretStr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from tavily import TavilyClient


# -------------------- Streamlit Setup --------------------

st.set_page_config(
    page_title="Corrective RAG",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Corrective RAG Assistant")
st.caption(
    "Ask questions about your PDFs with query rewriting "
    "and web-search fallback."
)


# -------------------- API Keys --------------------

with st.sidebar:
    st.header("API Configuration")

    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password"
    )

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

    tavily_api_key = st.text_input(
        "Tavily API Key",
        type="password"
    )


# -------------------- Helper Functions --------------------

def grade_document(llm, question, document):
    """Check whether a document is relevant to the question."""

    prompt = f"""
    Decide whether this document contains information useful
    for answering the question.

    Question:
    {question}

    Document:
    {document}

    Return exactly one word: RELEVANT or IRRELEVANT.
    """

    response = llm.invoke(prompt)
    grade = response.content.strip().upper()

    # Check IRRELEVANT first because it contains RELEVANT
    if "IRRELEVANT" in grade:
        return False

    return "RELEVANT" in grade


def rewrite_query(llm, question):
    """Rewrite the question for web search."""

    prompt = f"""
    Rewrite this question as a clear web search query.
    Keep the original meaning.
    Return only the rewritten query.

    Question: {question}
    """

    response = llm.invoke(prompt)
    return response.content.strip()


def web_search(question, api_key):
    """Search the web using Tavily."""

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=question,
        search_depth="advanced",
        max_results=5
    )

    return response.get("results", [])


def generate_answer(llm, question, context):
    """Generate an answer from the supplied evidence."""

    prompt = f"""
    Answer the question using only the evidence below.

    If the evidence does not contain enough information,
    clearly say so. Do not invent facts.

    Evidence:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)
    return response.content


# -------------------- Upload PDFs --------------------

st.subheader("1. Upload Your Documents")

uploaded_files = st.file_uploader(
    "Choose PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) uploaded.")

    for file in uploaded_files:
        st.write("📄", file.name)


# -------------------- Process PDFs --------------------

st.subheader("2. Process Documents")

if st.button("Process PDFs", type="primary"):

    if not gemini_api_key:
        st.error("Please enter your Gemini API key.")

    elif not uploaded_files:
        st.error("Please upload at least one PDF.")

    else:
        try:
            with st.spinner("Extracting and embedding PDFs..."):

                documents = []

                # Load every uploaded PDF
                for uploaded_file in uploaded_files:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(uploaded_file.getvalue())
                        temp_pdf = temp_file.name

                    try:
                        loader = PyPDFLoader(temp_pdf)
                        pages = loader.load()

                        for page in pages:
                            page.metadata["source"] = uploaded_file.name

                        documents.extend(pages)

                    finally:
                        if os.path.exists(temp_pdf):
                            os.remove(temp_pdf)

                if not documents:
                    st.error("No readable text found in the PDFs.")

                else:
                    # Split pages into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=2000,
                        chunk_overlap=200
                    )

                    splits = splitter.split_documents(documents)

                    # Create embeddings
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="gemini-embedding-2",
                        api_key=SecretStr(gemini_api_key)
                    )

                    # Create a fresh vector database for this processing run
                    vector = Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        collection_name="corrective_rag",
                        persist_directory="./chroma_data"
                    )

                    st.session_state.vector = vector
                    st.session_state.processed_files = [
                        file.name for file in uploaded_files
                    ]

                    st.success(
                        f"Processed {len(uploaded_files)} PDF(s) "
                        f"into {len(splits)} chunks."
                    )

        except Exception as e:
            st.error(f"Document processing failed: {e}")


# -------------------- Ask a Question --------------------

st.subheader("3. Ask Your Question")

question = st.text_input(
    "Your Question",
    placeholder="Example: What is quantum computing?"
)

if st.button("Ask"):

    if not groq_api_key:
        st.warning("Please enter your Groq API key.")

    elif "vector" not in st.session_state:
        st.warning("Please process your PDFs first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            # Initialize Groq
            llm = ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=0,
                api_key=SecretStr(groq_api_key)
            )

            # 1. Retrieve PDF chunks
            with st.spinner("Searching your documents..."):

                retriever = st.session_state.vector.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 3,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )

                results = retriever.invoke(question)

            # 2. Grade PDF chunks
            relevant_docs = []

            with st.spinner("Checking document relevance..."):
                for doc in results:
                    if grade_document(
                        llm,
                        question,
                        doc.page_content
                    ):
                        relevant_docs.append(doc)

            # 3. Answer from PDFs if relevant chunks exist
            if relevant_docs:

                context = "\n\n".join(
                    doc.page_content for doc in relevant_docs
                )

                with st.spinner("Generating answer from PDFs..."):
                    answer = generate_answer(
                        llm,
                        question,
                        context
                    )

                st.subheader("Answer")
                st.write(answer)
                st.info("Source: Uploaded PDFs")

                with st.expander("View retrieved sources"):
                    for i, doc in enumerate(relevant_docs, start=1):
                        st.markdown(f"**Source {i}**")
                        st.write(doc.page_content)

                        st.caption(
                            f"File: {doc.metadata.get('source', 'Unknown')} "
                            f"| Page: {doc.metadata.get('page', 'Unknown')}"
                        )

            # 4. Otherwise, use Tavily if available
            else:
                if not tavily_api_key:
                    st.warning(
                        "The PDFs do not contain relevant information. "
                        "Enter your Tavily API key to search the web."
                    )

                else:
                    st.info(
                        "No relevant PDF chunks found. "
                        "Searching the web instead..."
                    )

                    # Rewrite the question
                    with st.spinner("Rewriting your search query..."):
                        search_query = rewrite_query(llm, question)

                    st.caption(f"Web search query: {search_query}")

                    # Search the web
                    with st.spinner("Searching with Tavily..."):
                        web_results = web_search(
                            search_query,
                            tavily_api_key
                        )

                    if not web_results:
                        st.warning("Tavily returned no search results.")

                    else:
                        web_context = "\n\n".join(
                            f"Title: {item.get('title', 'Untitled')}\n"
                            f"URL: {item.get('url', '')}\n"
                            f"Content: {item.get('content', '')}"
                            for item in web_results
                        )

                        # Generate answer from web evidence
                        with st.spinner("Generating answer from web results..."):
                            answer = generate_answer(
                                llm,
                                question,
                                web_context
                            )

                        st.subheader("Answer")
                        st.write(answer)
                        st.info("Source: Tavily web search")

                        with st.expander("View web sources"):
                            for item in web_results:
                                st.markdown(
                                    f"**{item.get('title', 'Untitled')}**"
                                )
                                st.write(item.get("content", ""))

                                url = item.get("url", "")
                                if url:
                                    st.markdown(f"[Open source]({url})")

        except Exception as e:
            st.error(f"Question answering failed: {e}")

