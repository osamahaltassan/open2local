#!/usr/bin/env python3
"""
OpenAI API to Whisper-ASR-Webservice Proxy
Converts OpenAI API calls to whisper-asr-webservice format
"""

from flask import Flask, request, jsonify
import requests
import logging
from os import environ

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

WHISPER_URL = environ.get("WHISPER_URL", "http://localhost:9000")

@app.route('/v1/audio/transcriptions', methods=['POST'])
def transcriptions():
    try:
        # Get file from OpenAI format (field name: 'file')
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['file']
        
        # Get OpenAI parameters
        model = request.form.get('model', '')
        language = request.form.get('language', 'auto')
        response_format = request.form.get('response_format', 'json')
        
        # Convert to whisper-asr-webservice format
        files = {'audio_file': (audio_file.filename, audio_file.stream, audio_file.content_type)}
        
        params = {
            'output': 'json',  # Force JSON output
            'language': language if language != 'auto' else None,
            'task': 'transcribe'
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        # Forward to whisper service
        response = requests.post(
            f"{WHISPER_URL}/asr",
            files=files,
            params=params,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Convert whisper-asr response to OpenAI format
            if isinstance(result, dict) and 'text' in result:
                openai_response = {
                    "text": result['text']
                }
                # Add segments if available
                if 'segments' in result:
                    openai_response['segments'] = result['segments']
                
                return jsonify(openai_response)
            else:
                # Handle plain text response
                return jsonify({"text": str(result)})
        else:
            app.logger.error(f"Whisper service error: {response.status_code} - {response.text}")
            return jsonify({"error": "Transcription failed"}), response.status_code
            
    except Exception as e:
        app.logger.error(f"Proxy error: {str(e)}")
        return jsonify({"error": "Internal proxy error"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    print("Starting OpenAI to Whisper-ASR proxy on port 9001...")
    print("Point Open WebUI to: http://localhost:9001")
    app.run(host='0.0.0.0', port=9001, debug=False)