import streamlit as st
import json
import os
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import speech_recognition as sr
import pyttsx3

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Memory Assistant",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.chat-title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:white;
}

.card {
    padding:15px;
    border-radius:15px;
    background:#1f2937;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ENV ---------------- #

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        st.error("GEMINI_API_KEY not found")
        st.stop()

client = genai.Client(api_key=api_key)

# ---------------- VOICE ---------------- #

def listen_voice():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        st.info("🎤 Listening... Speak now")

        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)

        return text

    except:
        return None


def speak(text):

    engine = pyttsx3.init()

    engine.say(text)

    engine.runAndWait()
# ---------------- FILES ---------------- #

MEMORY_FILE = "memory.json"
CHAT_FILE = "chat_history.json"

# ---------------- LOAD MEMORY ---------------- #

if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)
    except:
        memory = {}
else:
    memory = {}
# Ensure learned_facts exists
if "learned_facts" not in memory:
    memory["learned_facts"] = []

# ---------------- LOAD CHAT ---------------- #

if os.path.exists(CHAT_FILE):
    try:
        with open(CHAT_FILE, "r") as f:
            chat_history = json.load(f)
    except:
        chat_history = []
else:
    chat_history = []

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("🧠 Memory Dashboard")

    st.subheader("👤 User Profile")

    name = st.text_input(
        "Name",
        value=memory.get("name", "")
    )

    skill = st.text_input(
        "Skill",
        value=memory.get("skill", "")
    )

    interest = st.text_input(
        "Interest",
        value=memory.get("interest", "")
    )

    if st.button("💾 Save Memory"):

        memory["name"] = name
        memory["skill"] = skill
        memory["interest"] = interest

        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=4)

        st.success("Memory Saved")

    st.divider()
    st.subheader("🧠 Learned Facts")

    for fact in memory["learned_facts"]:
        st.write("•", fact)

    st.subheader("📊 Stats")

    st.metric("Facts Learned", len(memory["learned_facts"]))
    st.metric("Chats", len(chat_history))

    st.divider()

    if st.button("🧠 Show Memory"):
        st.json(memory)

    voice_input = st.button("🎤 Speak")
    st.divider()

    uploaded_pdf = st.file_uploader(
        "📄 Upload PDF",
        type=["pdf"]
    )
    pdf_text = ""
    if uploaded_pdf:

        pdf_reader = PdfReader(uploaded_pdf)

        for page in pdf_reader.pages:

            text = page.extract_text()

            if text:
                pdf_text += text

        st.sidebar.success("✅ PDF Loaded")

# ---------------- MAIN ---------------- #

st.markdown(
    "<div class='chat-title'>🧠 AI Memory Assistant</div>",
    unsafe_allow_html=True
)

st.write("Personalized AI powered by Gemini")

# ---------------- CHAT HISTORY ---------------- #

for item in chat_history:

    with st.chat_message("user"):
        st.write(item["question"])

    with st.chat_message("assistant"):
        st.write(item["answer"])

# ---------------- USER INPUT ---------------- #

question = st.chat_input("Ask anything...")
if voice_input:

    voice_text = listen_voice()

    if voice_text:

        st.success(f"You said: {voice_text}")

        question = voice_text

if question:

    with st.chat_message("user"):
        st.write(question)
    if question.lower().startswith("remember:"):

        fact = question.replace("remember:", "").strip()

        if fact not in memory["learned_facts"]:
            memory["learned_facts"].append(fact)

            with open(MEMORY_FILE, "w") as f:
                json.dump(memory, f, indent=4)
        with st.chat_message("assistant"):
            st.write(f"✅ I will remember: {fact}")

        st.stop()

    if question.lower() in [
            "what do you know about me",
            "who am i",
            "my profile"
        ]:

            profile = f"""
        👤 Name: {memory.get('name', 'Unknown')}
        💻 Skill: {memory.get('skill', 'Unknown')}
        🎯 Interest: {memory.get('interest', 'Unknown')}

        🧠 Learned Facts:
        """

            for fact in memory["learned_facts"]:
                profile += f"• {fact}\n"

            with st.chat_message("assistant"):
                st.write(profile)

            st.stop()

    prompt = f"""
    You are a smart AI assistant.

    User Information:

    Name: {memory.get('name', 'Unknown')}
    Skill: {memory.get('skill', 'Unknown')}
    Interest: {memory.get('interest', 'Unknown')}

    Learned Facts:
    {memory.get('learned_facts', [])}

    PDF Content:
    {pdf_text[:10000]}

    IMPORTANT:
    - Use all stored memory and learned facts.
    - Use PDF content if a PDF is uploaded.
    - If the user asks "What do you know about me?",
      give a complete profile using all available information.
    - Mention learned facts whenever relevant.

    Question:
    {question}
    """

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        answer = response.text

        # Learn user facts automatically

        fact_prompt = f"""
        Extract ONLY personal facts from the message below.

        Message:
        {question}

        IMPORTANT:
        Return ONLY facts.
        No explanation.
        No greeting.
        No extra text.

        Examples:

        Input:
        My favorite game is BGMI

        Output:
        Favorite game is BGMI

        Input:
        I want to become an AI Engineer

        Output:
        Career goal is AI Engineer

        Input:
        Tell me about India

        Output:
        NONE
        """

        try:
            fact_response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=fact_prompt
            )

            learned_text = fact_response.text.strip()
            learned_text = learned_text.split("\n")[0]

            if learned_text.upper() != "NONE":

                facts = learned_text.split("\n")

                for fact in facts:

                    fact = fact.strip()

                    if fact and fact not in memory["learned_facts"]:
                        memory["learned_facts"].append(fact)

                with open(MEMORY_FILE, "w") as f:
                    json.dump(memory, f, indent=4)

        except:
            pass

        with st.chat_message("assistant"):
            st.write(answer)

        chat_history.append({
            "question": question,
            "answer": answer
        })

        with open(CHAT_FILE, "w") as f:
            json.dump(chat_history, f, indent=4)

    except Exception as e:
        st.error(str(e))