from helpers import transcribe_audio, process_with_llm, convert_text_to_speech

def inference(input_audio_file, output_audio_file, STT_Model, LLM_Model, TTS_Model):
    transcript = transcribe_audio(input_audio_file, STT_Model)
    processed_text = process_with_llm(transcript, LLM_Model)
    return convert_text_to_speech(processed_text, output_audio_file, TTS_Model)