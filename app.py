import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# ========================
# 環境変数
# ========================
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("APIキーが設定されていません。")
    st.stop()


# ========================
# Gemini
# ========================
@st.cache_resource
def get_gemini_model():
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

# ========================
# CSV読み込み
# ========================
@st.cache_data
def load_data(csv_file_path):
    return pd.read_csv(csv_file_path)

# ========================
# TF-IDF
# ========================
@st.cache_resource
def build_tfidf_model(texts):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer

# ========================
# Embedding
# ========================
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def build_embedding_model(texts):
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings

# ========================
# 検索（ハイブリッド）
# ========================
def hybrid_search(query, tfidf_matrix, tfidf_vectorizer, embeddings, texts, top_n=3):
    # TF-IDF
    query_vec = tfidf_vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    # Embedding
    model = get_embedding_model()
    query_emb = model.encode([query])
    emb_scores = cosine_similarity(query_emb, embeddings)[0]

    # 合算
    scores = tfidf_scores + emb_scores

    top_indices = scores.argsort()[-top_n:][::-1]
    return top_indices

# ========================
# チャット履歴
# ========================
def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# ========================
# Gemini応答
# ========================
def respond_with_gemini(query, indices, texts):
    client = get_gemini_model()

    context = "\n\n".join([texts[i][:300] for i in indices])

    prompt = f"""
    以下のニュース記事を参考に答えてください：

    {context}

    質問：{query}
    """

    model = get_gemini_model()

response = model.generate_content(prompt)

return response.text

# ========================
# Streamlit
# ========================
st.title("RAG System")

csv_file_path = "yahoo_news_articles_preprocessed.csv"
df = load_data(csv_file_path)

# ★ここ重要（カラム名確認）
texts = df["text"].fillna("").tolist()

tfidf_matrix, tfidf_vectorizer = build_tfidf_model(texts)
embeddings = build_embedding_model(texts)

init_chat_history()
display_chat_history()

user_input = st.chat_input("質問を入力してください")

if user_input:
    # ユーザー表示
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # 検索
    indices = hybrid_search(user_input, tfidf_matrix, tfidf_vectorizer, embeddings, texts, top_n=1)

    # 応答
    answer = respond_with_gemini(user_input, indices, texts)

    # AI表示
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)
