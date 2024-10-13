from flask import Flask, request, send_file, jsonify
from test_framework import inference
from helpers import transcribe_audio, process_with_llm, convert_text_to_speech
import ffmpeg

input_audio_path = "/home/LENOVO/capstone/input_audio/input_audio.wav"
fixed_audio_path = "/home/LENOVO/capstone/input_audio/fixed_audio.wav"
output_audio_path = "/home/LENOVO/capstone/output_audio/output_audio.wav"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "recordings/"

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "API is working!"})

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return "No audio file uploaded", 400

    audio = request.files['audio']
    STT_Model = request.form.get('STT_Model')
    LLM_Model = request.form.get('LLM_Model')
    TTS_Model = request.form.get('TTS_Model')
    
    audio.save(input_audio_path)
    ffmpeg.input(input_audio_path).output(fixed_audio_path, format='wav').run(overwrite_output=True)
    inference(fixed_audio_path, output_audio_path, TTS_Model, LLM_Model, STT_Model)
    return send_file(output_audio_path, mimetype='audio/wav')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return "No audio file uploaded", 400

    audio = request.files['audio']
    STT_Model = request.form.get('STT_Model')
    
    audio.save(input_audio_path)
    ffmpeg.input(input_audio_path).output(fixed_audio_path, format='wav').run(overwrite_output=True)
    transcript = transcribe_audio(fixed_audio_path, STT_Model)
    return jsonify({"transcript": transcript})

@app.route('/llmresponse', methods=['POST'])
def llmresponse():
    data = request.get_json()
    transcript = data['transcript']
    LLM_Model = data['LLM_Model']

    response = process_with_llm(transcript, LLM_Model)
    return jsonify({"response": response})

@app.route('/voiceresponse', methods=['POST'])
def voiceresponse():
    data = request.get_json()
    response = data['response']
    TTS_Model = data['TTS_Model']

    convert_text_to_speech(response, output_audio_path, TTS_Model)
    return send_file(output_audio_path, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
