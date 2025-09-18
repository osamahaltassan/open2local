#!/usr/bin/env python3
"""
Proxy: Whisper OpenAI's API to Whisper ASR Webservice's API
Converts OpenAI API calls to Whisper ASR Webservice API format
"""

from uuid import uuid4
from flask import Flask, request, jsonify
import requests
import logging
from os import environ

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Configuration
WHISPER_URL = environ.get("WHISPER_URL", "http://localhost:9000")
TIMEOUT = int(environ.get("WHISPER_TIMEOUT", "None"))
DEBUG_FLAG = environ.get("DEBUG", "False").lower() == "false"

# Constants
OPENAI_TO_WHISPER_FORMAT = {
    'json': 'json',
    'text': 'text',
    'srt': 'srt',
    'vtt': 'vtt',
    'verbose_json': 'json',  # Whisper returns same JSON regardless
}

def safe_json_or_text(response):
    """Helper function to safely parse JSON or return text"""
    try:
        return response.json()
    except ValueError:
        return {"text": response.text}

def check_whisper_health():
    """Check if the Whisper service is healthy"""
    try:
        resp = requests.get(f"{WHISPER_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False

@app.route('/v1/audio/transcriptions', methods=['POST'])
def transcriptions():
    req_id = str(uuid4())[:8]
    app.logger.info(f"[{req_id}] Starting transcription request")
    
    try:
        # Validate file upload
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({"error": "No valid audio file provided"}), 400
        
        audio_file = request.files['file']
        audio_data = audio_file.read()
        
        # Map OpenAI response_format to Whisper output format
        response_format = request.form.get('response_format', 'json')
        output_format = OPENAI_TO_WHISPER_FORMAT.get(response_format, 'json')
        
        # Build parameters
        params = {
            'output': output_format,
            'task': 'transcribe'
        }
        language = request.form.get('language', 'auto')
        if language != 'auto':
            params['language'] = language
        
        # Forward to Whisper service
        response = requests.post(
            f"{WHISPER_URL}/asr",
            files={'audio_file': (audio_file.filename, audio_data, audio_file.content_type)},
            params=params,
            timeout=None
        )
        
        # Handle response
        if response.status_code != 200:
            result = safe_json_or_text(response)
            return jsonify(result), response.status_code
        
        # Handle successful response
        result = safe_json_or_text(response)
        openai_response = {"text": result.get('text', '')}
        if 'segments' in result:
            openai_response['segments'] = result['segments']
        return jsonify(openai_response)
                
    except Exception as e:
        app.logger.error(f"[{req_id}] Proxy error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal proxy error"}), 500

@app.route('/health', methods=['GET'])
def health():
    if check_whisper_health():
        return jsonify({"status": "healthy", "whisper": "ok"})
    else:
        if DEBUG_FLAG:
            app.logger.warning("Health check failed: Whisper service unreachable")
        return jsonify({"status": "unhealthy", "whisper": "unreachable"}), 503