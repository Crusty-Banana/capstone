import os
import soundfile as sf
from pydub import AudioSegment
from google.cloud import speech, texttospeech
from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LlamaForCausalLM, LlamaTokenizer
import torch

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# stt models
deepgram_stt = DeepgramClient(DEEPGRAM_API_KEY)
google_stt = speech.SpeechClient()

# llm models
gpt2_llm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", device=device, clean_up_tokenization_spaces=True)
gpt2_llm = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

llama3_llm_tokenizer = None # LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
llama3_llm = None # LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b").to(device)

# tts models
google_tts = texttospeech.TextToSpeechClient()

def read_audio_file(file_path):
    # Convert audio to WAV if it's not in WAV format (Google API works best with WAV)
    audio = AudioSegment.from_file(file_path)
    wav_path = "temp_audio.wav"
    audio.export(wav_path, format="wav")

    return wav_path

def transcribe_audio(file_path, STT_model):
    if (STT_model == "Google"):
        # Read audio data
        with sf.SoundFile(file_path) as audio_file:
            audio_content = audio_file.read(dtype="int16").tobytes()
            sample_rate = audio_file.samplerate

        # Prepare the audio for Google Speech API
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="en-US"
        )

        # Perform transcription
        response = google_stt.recognize(config=config, audio=audio)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        
        return transcript
    elif (STT_model == "Deepgram"):
        with open(file_path, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )
        response = deepgram_stt.listen.prerecorded.v("1").transcribe_file(payload, options)
        transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
        return transcript
    return 404

def process_with_llm(transcript, LLM_model):
    if (LLM_model == "gpt2"):
        transcript_tokens = gpt2_llm_tokenizer.encode(transcript, return_tensors='pt').to(device)
        response = gpt2_llm.generate(
            transcript_tokens,
            max_length=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
        )    
        processed_text = gpt2_llm_tokenizer.decode(response[0], skip_special_tokens=True)
        return processed_text
    elif (LLM_model == "2.7B-llama3"):
        transcript_tokens = llama3_llm_tokenizer(transcript, return_tensors="pt")
        outputs = llama3_llm.generate(transcript_tokens["input_ids"], max_length=50)
        response = llama3_llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
def convert_text_to_speech(text, output_audio_file, TTS_model):
    if TTS_model == "Google":
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Config
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", name="en-US-Wavenet-D"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # Generate the speech
        response = google_tts.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Write the output audio to a file
        with open(output_audio_file, "wb") as out:
            out.write(response.audio_content)
        
        print(f"Audio content written to file {output_audio_file}")
        return response
