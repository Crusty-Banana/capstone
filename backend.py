from flask import Flask, request, send_file, jsonify
from test_framework import inference
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
    
    audio.save(input_audio_path)
    ffmpeg.input(input_audio_path).output(fixed_audio_path, format='wav').run(overwrite_output=True)
    inference(fixed_audio_path, output_audio_path)
    return send_file(output_audio_path, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
