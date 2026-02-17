"""
Flask REST API for Complaint Classification
Exposes LSTM model as HTTP endpoints
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from typing import Dict, List
import logging
from datetime import datetime
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from inference import ComplaintAPIWrapper

# ============================================
# FLASK APP SETUP
# ============================================
app = Flask(__name__)
CORS(app)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global inference engine
inference_engine = None

# Model path
MODEL_DIR = r"d:\Majorproject\resolvex-citizen-connect-main\src\ml\models\best_model"


def initialize_engine():
    """Initialize inference engine on app startup"""
    global inference_engine
    try:
        logger.info("🚀 Initializing Complaint Classification Engine...")
        inference_engine = ComplaintAPIWrapper()
        logger.info("✅ Engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize engine: {e}")
        raise


# ============================================
# HEALTH CHECK ENDPOINTS
# ============================================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Complaint Classification API',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': inference_engine is not None
    }), 200


@app.route('/api/v1/status', methods=['GET'])
def get_status():
    """Get API and model status"""
    try:
        if inference_engine is None:
            return jsonify({
                'status': 'not_initialized',
                'error': 'Engine not initialized'
            }), 503
        
        stats = inference_engine.get_complaint_stats()
        return jsonify({
            'status': 'ready',
            **stats
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


# ============================================
# MAIN PREDICTION ENDPOINTS
# ============================================
@app.route('/api/v1/classify', methods=['POST'])
def classify_complaint():
    """
    Classify a single complaint
    
    Request body:
    {
        "text": "Complaint text here",
        "user_id": 1,
        "save_to_db": true
    }
    
    Response:
    {
        "complaint_text": "...",
        "language": "English",
        "sentiment": {...},
        "department": {...},
        "urgency": {...},
        "complaint_id": 123,
        "timestamp": "...",
        "status": "success"
    }
    """
    try:
        if inference_engine is None:
            return jsonify({
                'status': 'failed',
                'error': 'Engine not initialized'
            }), 503
        
        data = request.get_json()
        
        # Validate request
        if not data or 'text' not in data:
            return jsonify({
                'status': 'failed',
                'error': 'Missing "text" field in request'
            }), 400
        
        complaint_text = data.get('text', '').strip()
        user_id = data.get('user_id', 1)
        save_to_db = data.get('save_to_db', True)
        
        if len(complaint_text) < 5:
            return jsonify({
                'status': 'failed',
                'error': 'Complaint text must be at least 5 characters long'
            }), 400
        
        # Classify
        logger.info(f"📝 Classifying complaint from user {user_id}")
        result = inference_engine.predict(complaint_text, user_id, save_to_db)
        
        # Return response
        status_code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"❌ Classification error: {e}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/v1/classify-batch', methods=['POST'])
def classify_batch():
    """
    Classify multiple complaints at once
    
    Request body:
    {
        "complaints": [
            {"text": "...", "user_id": 1},
            {"text": "...", "user_id": 2}
        ],
        "save_to_db": true
    }
    
    Response:
    {
        "total": 2,
        "successful": 2,
        "failed": 0,
        "predictions": [...]
    }
    """
    try:
        if inference_engine is None:
            return jsonify({
                'status': 'failed',
                'error': 'Engine not initialized'
            }), 503
        
        data = request.get_json()
        
        if not data or 'complaints' not in data:
            return jsonify({
                'status': 'failed',
                'error': 'Missing "complaints" field in request'
            }), 400
        
        complaints = data.get('complaints', [])
        save_to_db = data.get('save_to_db', True)
        
        if not complaints:
            return jsonify({
                'status': 'failed',
                'error': 'Complaints list is empty'
            }), 400
        
        logger.info(f"📝 Batch classifying {len(complaints)} complaints")
        
        # Create list of complaints with text and user_id
        formatted_complaints = [
            {
                'text': c.get('text', ''),
                'user_id': c.get('user_id', 1)
            }
            for c in complaints
        ]
        
        result = inference_engine.predict_batch(formatted_complaints, save_to_db)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"❌ Batch classification error: {e}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# ============================================
# UTILITY ENDPOINTS
# ============================================
@app.route('/api/v1/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages"""
    return jsonify({
        'supported_languages': ['English', 'Hindi', 'Tamil'],
        'auto_detection': True
    }), 200


@app.route('/api/v1/departments', methods=['GET'])
def get_supported_departments():
    """Get list of supported departments"""
    if inference_engine is None:
        return jsonify({
            'status': 'failed',
            'error': 'Engine not initialized'
        }), 503
    
    stats = inference_engine.get_complaint_stats()
    return jsonify({
        'departments': stats.get('departments_supported', [])
    }), 200


@app.route('/api/v1/sentiments', methods=['GET'])
def get_sentiment_labels():
    """Get available sentiment labels"""
    return jsonify({
        'sentiments': ['Positive', 'Neutral', 'Negative']
    }), 200


# ============================================
# ERROR HANDLERS
# ============================================
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'error': 'Endpoint not found',
        'path': request.path
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'status': 'error',
        'error': 'Method not allowed',
        'method': request.method,
        'path': request.path
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'status': 'error',
        'error': 'Internal server error'
    }), 500


# ============================================
# API DOCUMENTATION
# ============================================
@app.route('/api/v1/docs', methods=['GET'])
def get_documentation():
    """Get API documentation"""
    docs = {
        'title': 'Complaint Classification API',
        'version': '1.0.0',
        'description': 'LSTM-based multilingual complaint classification system',
        'base_url': '/api/v1',
        'endpoints': {
            'health': {
                'method': 'GET',
                'path': '/health',
                'description': 'Health check endpoint'
            },
            'status': {
                'method': 'GET',
                'path': '/status',
                'description': 'Get API and model status'
            },
            'classify': {
                'method': 'POST',
                'path': '/classify',
                'description': 'Classify a single complaint',
                'request_body': {
                    'text': 'str (required)',
                    'user_id': 'int (optional, default=1)',
                    'save_to_db': 'bool (optional, default=True)'
                }
            },
            'classify_batch': {
                'method': 'POST',
                'path': '/classify-batch',
                'description': 'Classify multiple complaints',
                'request_body': {
                    'complaints': 'list of {text, user_id} (required)',
                    'save_to_db': 'bool (optional, default=True)'
                }
            },
            'languages': {
                'method': 'GET',
                'path': '/languages',
                'description': 'Get supported languages'
            },
            'departments': {
                'method': 'GET',
                'path': '/departments',
                'description': 'Get supported departments'
            },
            'sentiments': {
                'method': 'GET',
                'path': '/sentiments',
                'description': 'Get sentiment labels'
            }
        },
        'example_requests': {
            'single_complaint': {
                'method': 'POST',
                'url': '/api/v1/classify',
                'body': {
                    'text': 'Power outage in my area since 12 hours',
                    'user_id': 1,
                    'save_to_db': True
                }
            },
            'batch_complaints': {
                'method': 'POST',
                'url': '/api/v1/classify-batch',
                'body': {
                    'complaints': [
                        {'text': 'Complaint 1', 'user_id': 1},
                        {'text': 'Complaint 2', 'user_id': 2}
                    ],
                    'save_to_db': True
                }
            }
        }
    }
    return jsonify(docs), 200


# ============================================
# APP STARTUP
# ============================================
@app.before_request
def before_request():
    """Initialize engine before first request"""
    global inference_engine
    if inference_engine is None:
        initialize_engine()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 COMPLAINT CLASSIFICATION API")
    print("=" * 70)
    print("\n📚 API Documentation available at: http://localhost:5000/api/v1/docs")
    print("\n⚠️  Make sure the trained model is available at:")
    print(f"   {MODEL_DIR}\n")
    
    # Initialize engine
    initialize_engine()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Prevent double initialization
    )
