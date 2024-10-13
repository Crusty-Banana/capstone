import os
import helpers
from dotenv import load_dotenv

load_dotenv()

a = helpers.transcribe_audio("input_audio/fixed_audio.wav", "Deepgram")
print(a)