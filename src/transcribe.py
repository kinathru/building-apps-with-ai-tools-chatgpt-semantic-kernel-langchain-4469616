from openai import OpenAI
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

user_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=user_api_key)

audio_file = open("./lost_debit_card.wav", "rb")
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"
)
print(transcription)
