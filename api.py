# -*- coding: utf-8 -*-
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from explain_model import SentenceBasedTextDetector
from ensemble_image_detector import EnsembleImageDetector
import traceback

app = Flask(__name__)
CORS(app)

# Load models
TEXT_MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"
IMAGE_MODEL = "Organika/sdxl-detector"

print("Starting server and loading models...")
text_detector = SentenceBasedTextDetector(TEXT_MODEL)
image_detector = EnsembleImageDetector()
print("Server ready!")

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'DetectAI API is running',
        'endpoints': {
            'health': '/health',
            'text': '/analyze',
            'image': '/analyze-image'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Check if server is running"""
    return jsonify({
        'status': 'ok', 
        'message': 'Server is running',
        'text_model': TEXT_MODEL,
        'image_model': IMAGE_MODEL
    })

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text and return prediction"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text'].strip()

        if len(text) == 0:
            return jsonify({'error': 'Text is empty'}), 400

        if len(text) < 10:
            return jsonify({'error': 'Text is too short (minimum 10 characters)'}), 400

        print(f"Analyzing text ({len(text)} characters)...")
        result = text_detector.explain(text)
        print(f"Result: {result['prediction']} ({result['ai_probability']}%)")

        return jsonify(result)

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze image and return prediction"""
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_base64 = data['image']
        
        print("Analyzing image...")
        result = image_detector.detect_from_base64(image_base64)
        print(f"Result: {result['prediction']} ({result['ai_probability']}%)")

        return jsonify(result)

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Analysis failed'}), 500

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 70)
    print("DetectAI API Server")
    print("=" * 70)
    print(f"Text Model: {TEXT_MODEL}")
    print(f"Image Model: {IMAGE_MODEL}")
    print(f"Server running on port: {PORT}")
    print("=" * 70 + "\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=False)
