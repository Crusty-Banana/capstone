import os
import soundfile as sf
from pydub import AudioSegment
from google.cloud import speech, texttospeech
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from helpers import read_audio_file, transcribe_audio, process_with_llm, convert_text_to_speech
import torch

# Initializeed GOOGLE API Client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "serviceAcc.json"
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

# Initialized LLM
device = "cuda"
llm_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(llm_name, device=device, clean_up_tokenization_spaces=True)
llm = GPT2LMHeadModel.from_pretrained(llm_name).to(device)

def inference(input_audio_file, output_audio_file):
    # Read and transcribe audio
    transcript = transcribe_audio(input_audio_file, speech_client)
    print(f"Transcribed text: {transcript}")

    # Process transcribed text with local LLM
    transcript_tokens = tokenizer.encode(transcript, return_tensors='pt').to(device)
    processed_text_tokens = process_with_llm(transcript_tokens, llm, device)
    processed_text = tokenizer.decode(processed_text_tokens[0], skip_special_tokens=True)
    print(f"Processed text: {processed_text}")

    return convert_text_to_speech(processed_text, output_audio_file, tts_client, device)