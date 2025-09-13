# Persona ChatBot (Streamlit)

A minimal Streamlit chat application that demonstrates:

- Continuous conversation without rerunning the Python process
- Multiple persona options (RoastBot, ShakespeareBot, Emoji Translator Bot, etc.)
- Simple memory that keeps a short buffer of conversation
- Flexible LLM backends: OpenAI (via LangChain), HuggingFace Hub (via LangChain), or local `transformers` fallback

## Files

- `chatbot.py` — Streamlit app (save from this document)

## Requirements

- Python 3.8+
- Install dependencies (recommended in a virtualenv):

```bash
pip install streamlit
# Optional but recommended for nicer LLMs
pip install langchain
pip install openai            # if you want to use OpenAI
pip install huggingface_hub  # if you want to use HF via LangChain
pip install transformers     # for local transformer fallback