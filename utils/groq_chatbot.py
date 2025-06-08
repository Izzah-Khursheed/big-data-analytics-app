import streamlit as st
import requests  # to make HTTP API requests

# Use Streamlit secrets or environment variables for security
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  # Your API key

def run_groq_chatbot_tab():
    st.header("AI Chatbot 💬")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Chat with the Big Data Bot", placeholder="Ask anything related to Big Data, Analytics, or Technologies...")

    if user_input:
        st.session_state["chat_history"].append({"user": user_input})
        bot_response = generate_bot_response(user_input)
        st.session_state["chat_history"].append({"bot": bot_response})

    for chat in st.session_state["chat_history"]:
        if "user" in chat:
            st.markdown(f"**You:** {chat['user']}")
        if "bot" in chat:
            st.markdown(f"**GroqChat:** {chat['bot']}")

def generate_bot_response(user_input: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct", 
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        # The structure depends on Groq's API response format
        # Here, we assume the AI response is in data['choices'][0]['message']['content']
        ai_text = data['choices'][0]['message']['content']
        return ai_text
    except Exception as e:
        return f"Error fetching response from Groq API: {e}"
