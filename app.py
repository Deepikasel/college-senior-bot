import streamlit as st
from src.bot import SeniorBot
from pathlib import Path

st.set_page_config(page_title="College Senior Bot", page_icon="🎓", layout="centered")
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "college_qa.jsonl"




@st.cache_resource
def get_bot():
    try:
        if not (MODEL_DIR / "tfidf_vectorizer.joblib").exists():
            return SeniorBot.build(str(DATA_PATH), str(MODEL_DIR))
        return SeniorBot.load(str(MODEL_DIR), str(DATA_PATH))
    except Exception as e:
        st.error("Bot failed to load")
        st.exception(e)
        st.stop()


st.title("🎓 College Senior Chatbot")
st.caption("Sarcastic, friendly, and slightly too honest. (Lightweight retrieval-based AI)")

bot = get_bot()

with st.sidebar:
    st.header("Settings")
    spice = st.slider("Sarcasm level", 0, 2, 2, help="0=mild, 1=normal, 2=spicy")
    st.divider()
    show_debug = st.toggle("Show debug matches", value=False)
    st.caption("No cheating/unsafe requests. Be nice ✌️")

if "messages" not in st.session_state:
    st.session_state.messages = []

# chat history
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("Ask me anything about college life...")
if prompt:
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    bot.spice = spice
    answer, debug = bot.reply(prompt)

    with st.chat_message("assistant"):
        st.markdown(answer)
        if show_debug and debug:
            st.caption("Top matches (question, similarity):")
            for q, s in debug:
                st.code(f"{s:.3f} — {q}")

    st.session_state.messages.append(("assistant", answer))
