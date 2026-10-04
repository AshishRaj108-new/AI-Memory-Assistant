# 🧠 MEMS ZERO — Long-Term AI Memory Assistant

A personal AI assistant that **remembers you**. MEMS ZERO learns facts from your
conversations, keeps them in a persistent memory, and uses them to give
personalised answers across chats. Built with **Streamlit** and **Google Gemini**.

## ✨ Features

| Area | What it does |
|---|---|
| **Long-term memory** | Learns personal facts automatically, or on command with `remember: ...` / `forget: ...` |
| **Smart chats** | Every chat gets an AI-generated topic title (no more "Chat 1, 2, 3") |
| **Pin & search** | Pin important chats to the top, search across titles and messages |
| **Chat tools** | Rename, delete, export a chat as Markdown, regenerate the last answer |
| **Streaming answers** | Responses appear word by word, with response time shown |
| **Response styles** | Balanced, Concise, Detailed or Teacher mode |
| **PDF Q&A** | Upload a PDF and ask questions about it |
| **Voice (optional)** | Speak your question, and have answers read aloud |
| **Memory dashboard** | View, delete and export everything the assistant knows about you |

## 🏗️ How it works

```
User message ──► command? (remember / forget / profile) ──► handled locally
      │
      └─► Gemini (system prompt = profile + learned facts + PDF + style
                   + last 8 conversation turns)
              │
              ├─► streamed answer ──► saved to chat_history.json
              ├─► fact extraction ──► saved to memory.json
              └─► topic title (first message only) ──► chat name
```

## 🚀 Setup

```bash
# 1. clone and enter the project
git clone <your-repo-url>
cd MemoryAgentProject

# 2. create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 3. add your Gemini API key
echo GEMINI_API_KEY=your_key_here > .env

# 4. run
streamlit run app.py
```

Optional: set `GEMINI_MODEL` in `.env` to use a different Gemini model.

## 💡 Usage tips

- `remember: my favorite language is Python` — save a fact manually
- `forget: Python` — remove facts that contain this text
- `what do you know about me` — instant profile summary
- ★ next to a chat pins it to the top of the list

## 📁 Project structure

```
MemoryAgentProject/
├── app.py                 # the whole application
├── requirements.txt
├── .env                   # API key (never commit this)
├── .streamlit/config.toml # dark theme
├── assets/
│   ├── style.css          # premium dark UI
│   ├── logo.png  user.png  chatbot.png  background.png
├── memory.json            # created automatically: profile + learned facts
└── chat_history.json      # created automatically: all chats
```

## 🔒 Privacy

All memory and chat data is stored **locally** in JSON files. Messages are sent
to the Gemini API only to generate answers, titles and fact extraction.

## 🗺️ Future scope

- Semantic memory search with embeddings
- Multi-user login
- Cloud database instead of JSON files
- Memory categories and expiry