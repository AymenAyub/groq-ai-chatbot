# Chat with Groq AI

An AI-powered chatbot built using **Python, Streamlit, and Groq API**. It provides a simple, clean interface for interacting with an AI assistant.

## Live Demo

[**Try Chat with Groq AI →**](https://chat-with-groq-ai.streamlit.app/)

## Overview

Chat with Groq AI is a web-based chatbot application that integrates Groq's API to provide AI-powered conversational experiences through an interactive Streamlit interface.

The application focuses on a clean user experience, simple navigation, and an intuitive chat interface.

## Tech Stack

- **Python** – Core programming language
- **Streamlit** – Web application framework
- **Groq API** – AI inference and chatbot responses

## Project Structure

```text
AI_Chatbot/
│
├── .streamlit/
│   └── config.toml
│
├── chatbot.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd AI_Chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.streamlit/secrets.toml` file and add your Groq API key:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Keep your API key private. Never commit `secrets.toml` to GitHub.

### 5. Run the Application

```bash
streamlit run chatbot.py
```

## Deployment

The application is deployed using **Streamlit Community Cloud**.

[Visit Live Application](https://chat-with-groq-ai.streamlit.app/)

## Author

**Aymen Ayub**

[GitHub](https://github.com/AymenAyub)

---

*Built with Python, Streamlit, and Groq AI.*
