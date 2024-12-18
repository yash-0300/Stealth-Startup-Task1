import os
from groq import Groq
from PROMPTS import *

# API KEYS
os.environ['GROQ_API_KEY'] = 'PASTE YOUR KEY'
os.environ['OPENAI_API_KEY'] = 'PASTE YOUR KEY'
os.environ['TAVILY_API_KEY'] = 'PASTE YOUR KEY'

# Speech to Text conversion using Whisper Large V3 turbo model
def speech_to_text(audio_data):
    groq_client = Groq(api_key = GROQ_API_KEY)
    with open(audio_data, "rb") as audio_file:
        transcript = groq_client.audio.transcriptions.create(
            file=("temp.mp3", audio_file),
            model="whisper-large-v3-turbo",
            prompt="Clear and precise transcription",
            response_format="json",
            language="en",
            temperature=0.0,
        )
    return transcript.text


def getQuestionConversation(conversation):
    groq_client = Groq(api_key = GROQ_API_KEY)
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": system_prompt_conversation_groq},
            {"role": "user", "content": conversation,}
        ],
        model = "llama3-8b-8192",
        temperature = 0.5,
        max_tokens = 1024,
        top_p = 1,
        stop = None,
        stream = False,
    )
    return chat_completion.choices[0].message.content

