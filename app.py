import streamlit as st
import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

st.set_page_config(
    page_title="ThaiGuide AI",
    page_icon="🇹🇭",
    layout="centered"
)

GEMINI_API_KEY = st.secrets["AIzaSyBOE1XjyiRX3fI1QYJYPv39mh69oyfBHTc"]

@st.cache_resource
def load_rag_chain():
    df = pd.read_csv("thai_tourist_attractions.csv")

    def create_document(row):
        content = (
            f"Name: {row['name']}\n"
            f"Region: {row['region']}\n"
            f"Province: {row['province']}\n"
            f"Type: {row['type']}\n"
            f"Description: {row['description']}\n"
            f"Best Season: {row['best_season']}\n"
            f"Highlights: {row['highlights']}\n"
            f"Entrance Fee: {row['entrance_fee']}\n"
            f"Nearby: {row['nearby']}"
        )
        metadata = {
            "id": int(row["id"]),
            "name": row["name"],
            "region": row["region"],
            "province": row["province"],
            "type": row["type"],
        }
        return Document(page_content=content, metadata=metadata)

    documents = [create_document(row) for _, row in df.iterrows()]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=1024,
    )

    SYSTEM_PROMPT = """\
You are ThaiGuide AI, an expert Thai travel guide assistant.
Use the following retrieved context to answer the user accurately and helpfully.

Context:
{context}

Guidelines:
- Answer in the same language the user used (Thai or English).
- Be friendly and enthusiastic like a real travel guide.
- Include practical info: fees, best seasons, tips.
- Use emojis sparingly.
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"--- Attraction {i}: {doc.metadata['name']} ---\n{doc.page_content}")
        return "\n\n".join(parts)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ── UI ────────────────────────────────────────────────
st.title("🇹🇭 ThaiGuide AI")
st.caption("AI-Powered Thai Tourism Chatbot — RAG + FAISS + Gemini")
st.caption("ถามเป็นภาษาไทยหรืออังกฤษได้เลย 🗣️")

with st.expander("💡 ตัวอย่างคำถาม / Example Questions"):
    cols = st.columns(2)
    examples = [
        "What are the top attractions in Chiang Rai?",
        "I want to go diving. Where should I go?",
        "อยากเที่ยวน้ำตก ไปที่ไหนดี?",
        "What is the best time to visit Phuket?",
        "Which UNESCO sites are in Thailand?",
        "How much to visit the Grand Palace?",
        "ภาคเหนือมีที่เที่ยวอะไรบ้าง?",
        "แนะนำที่เที่ยวสำหรับครอบครัวหน่อย",
    ]
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, key=f"ex_{i}"):
                st.session_state["example_input"] = ex

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.spinner("Loading AI model..."):
    rag_chain = load_rag_chain()

user_input = None
if "example_input" in st.session_state:
    user_input = st.session_state.pop("example_input")

prompt_input = st.chat_input("Ask about Thai tourist attractions...")
if prompt_input:
    user_input = prompt_input

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_input)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

st.divider()
st.caption("Powered by Gemini 2.0 Flash + LangChain RAG + FAISS | 25 Thai attractions")