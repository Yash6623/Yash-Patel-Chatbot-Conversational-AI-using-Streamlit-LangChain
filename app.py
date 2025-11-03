# !pip install langchain transformers sentence-transformers chromadb streamlit streamlit-chat

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline
import streamlit as st
from streamlit_chat import message

# --- Load Yash's Data ---
with open("yash.txt", "r", encoding="utf-8") as f:
    text = f.read()

splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = [d for d in splitter.split_text(text) if d.strip()]
documents = [Document(page_content=d) for d in docs]

# --- Embeddings & Database ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(documents, embeddings, persist_directory="db")
retriever = db.as_retriever(search_kwargs={"k": 3})

# --- Model Pipeline ---
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # More powerful than small model
    device=-1,
    max_new_tokens=400,
    temperature=0.7,
)
llm = HuggingFacePipeline(pipeline=pipe)

# --- Conversational Retrieval Chain ---
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever
)

# --- Streamlit App ---
st.title("💬 Ask questions about Yash Patel's projects and experiences!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Type your question here:")

if user_input:
    chat_history_pairs = [(chat["user"], chat["bot"]) for chat in st.session_state.chat_history]
    result = qa_chain({"question": user_input, "chat_history": chat_history_pairs})
    answer = result["answer"].strip()

    # Clean up and enhance response
    if len(answer) < 20:
        answer = "Hmm, I couldn’t find that info clearly — maybe try rephrasing your question? 😊"
    elif not answer.endswith("."):
        answer += "."

    answer = f"Sure! Here's what I found — {answer} 😊"
    st.session_state.chat_history.append({"user": user_input, "bot": answer})

# --- Display Chat ---
for i, chat in enumerate(st.session_state.chat_history):
    message(chat["user"], is_user=True, key=f"user_{i}")
    message(chat["bot"], key=f"bot_{i}")
