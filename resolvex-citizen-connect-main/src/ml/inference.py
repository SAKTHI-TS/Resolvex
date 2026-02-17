"""
Inference API for Real-time Complaint Classification
- Load trained model
- Perform predictions
- Database integration
- REST API endpoints
"""

import torch
import os
import json
from typing import Dict, List
import joblib
import pandas as pd
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import logging

from lstm_model import ComplaintPredictor, load_model
from data_preprocessing import TextPreprocessor

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# DATABASE CONFIGURATION
# ============================================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_password',  # Change this
    'database': 'resolveX_grievance_system',
    'raise_on_warnings': False
}

MODEL_DIR = r"d:\Majorproject\resolvex-citizen-connect-main\src\ml\models\best_model"

# ============================================
# LANGUAGE DETECTION
# ============================================
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


class LanguageDetector:
    """Detect language of complaint text"""
    
    LANGUAGE_MAP = {
        'en': 'English',
        'hi': 'Hindi',
        'ta': 'Tamil'
    }
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect language and return language code"""
        try:
            lang_code = detect(text)
            return LanguageDetector.LANGUAGE_MAP.get(lang_code, 'English')
        except:
            return 'English'


# ============================================
# DATABASE OPERATIONS
# ============================================
class ComplaintDatabaseManager:
    """Manage database operations for complaints"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.connection = None
        self.connect()
    
    def connect(self):
        """Connect to MySQL database"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                logger.info("✅ Connected to MySQL database")
        except Error as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")
    
    def save_complaint(self, complaint_data: Dict) -> int:
        """
        Save complaint and predictions to database
        
        Returns: complaint_id
        """
        try:
            cursor = self.connection.cursor()
            
            # Generate complaint number
            complaint_number = f"{complaint_data['department'][:3].upper()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Insert complaint
            query = """
            INSERT INTO complaints 
            (complaint_number, user_id, department_id, complaint_type_id, language, 
             complaint_text, status, priority, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """
            
            # Get department_id
            dept_cursor = self.connection.cursor()
            dept_query = "SELECT department_id FROM departments WHERE department_name = %s"
            dept_cursor.execute(dept_query, (complaint_data['department'],))
            dept_result = dept_cursor.fetchone()
            department_id = dept_result[0] if dept_result else 1
            dept_cursor.close()
            
            values = (
                complaint_number,
                complaint_data.get('user_id', 1),
                department_id,
                None,  # complaint_type_id
                complaint_data['language'],
                complaint_data['complaint_text'],
                'Registered',
                complaint_data['priority'],
            )
            
            cursor.execute(query, values)
            complaint_id = cursor.lastrowid
            
            # Save sentiment analysis
            self._save_sentiment_analysis(cursor, complaint_id, complaint_data)
            
            # Save priority analysis
            self._save_priority_analysis(cursor, complaint_id, complaint_data)
            
            self.connection.commit()
            logger.info(f"✅ Complaint saved with ID: {complaint_id}")
            cursor.close()
            
            return complaint_id
            
        except Error as e:
            logger.error(f"❌ Error saving complaint: {e}")
            self.connection.rollback()
            return None
    
    def _save_sentiment_analysis(self, cursor, complaint_id: int, complaint_data: Dict):
        """Save sentiment analysis results"""
        query = """
        INSERT INTO sentiment_analysis 
        (complaint_id, sentiment_label, sentiment_score, analyzed_at)
        VALUES (%s, %s, %s, NOW())
        """
        
        sentiment = complaint_data['sentiment']['label']
        sentiment_score = complaint_data['sentiment']['scores'][sentiment.lower()]
        
        values = (complaint_id, sentiment, sentiment_score)
        cursor.execute(query, values)
    
    def _save_priority_analysis(self, cursor, complaint_id: int, complaint_data: Dict):
        """Save priority/urgency analysis results"""
        query = """
        INSERT INTO priority_analysis 
        (complaint_id, urgency_score, priority_level)
        VALUES (%s, %s, %s)
        """
        
        urgency_score = complaint_data['urgency']['score']
        priority_level = complaint_data['urgency']['priority']
        
        values = (complaint_id, urgency_score, priority_level)
        cursor.execute(query, values)
    
    def get_complaint_by_id(self, complaint_id: int) -> Dict:
        """Retrieve complaint details from database"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
            SELECT c.*, s.sentiment_label, s.sentiment_score, 
                   p.urgency_score, p.priority_level
            FROM complaints c
            LEFT JOIN sentiment_analysis s ON c.complaint_id = s.complaint_id
            LEFT JOIN priority_analysis p ON c.complaint_id = p.complaint_id
            WHERE c.complaint_id = %s
            """
            cursor.execute(query, (complaint_id,))
            result = cursor.fetchone()
            cursor.close()
            return result
        except Error as e:
            logger.error(f"Error retrieving complaint: {e}")
            return None


# ============================================
# COMPLAINT INFERENCE ENGINE
# ============================================
class ComplaintInferenceEngine:
    """Main inference engine for complaint classification"""
    
    def __init__(self, model_dir: str, device='cpu'):
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.department_encoder = None
        self.db_manager = None
        
        self._load_model()
        self._load_encoders()
    
    def _load_model(self):
        """Load trained model and tokenizer"""
        try:
            self.model, self.tokenizer, self.department_encoder = load_model(
                self.model_dir, self.device
            )
            logger.info(f"✅ Model loaded from {self.model_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _load_encoders(self):
        """Load label encoders"""
        try:
            self.sentiment_encoder = joblib.load(
                os.path.join(self.model_dir, '../data/sentiment_encoder.pkl')
            )
            logger.info("✅ Encoders loaded")
        except Exception as e:
            logger.warning(f"⚠️  Could not load encoders: {e}")
    
    def preprocess_text(self, text: str, language: str) -> str:
        """Preprocess complaint text"""
        return TextPreprocessor.clean_complaint_text(text, language)
    
    def classify_complaint(self, complaint_text: str, user_id: int = 1) -> Dict:
        """
        Classify complaint and return predictions
        
        Args:
            complaint_text: Raw complaint text
            user_id: User ID submitting the complaint
        
        Returns:
            Dictionary with all predictions
        """
        try:
            # Detect language
            language = LanguageDetector.detect_language(complaint_text)
            logger.info(f"🌍 Detected language: {language}")
            
            # Preprocess
            processed_text = self.preprocess_text(complaint_text, language)
            
            if not processed_text:
                return {
                    'error': 'Text too short or invalid after preprocessing',
                    'status': 'failed'
                }
            
            # Create predictor
            predictor = ComplaintPredictor(
                self.model,
                self.tokenizer,
                self.device,
                self.department_encoder
            )
            
            # Get predictions
            prediction = predictor.predict(processed_text, language)
            
            # Add metadata
            prediction['original_text'] = complaint_text
            prediction['processed_text'] = processed_text
            prediction['user_id'] = user_id
            prediction['status'] = 'success'
            prediction['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"✅ Classification complete: {prediction['sentiment']['label']}, "
                       f"{prediction['department']['name']}, {prediction['urgency']['priority']}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Classification failed: {e}")
            return {
                'error': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_classify(self, complaints: List[Dict]) -> List[Dict]:
        """
        Classify multiple complaints
        
        Args:
            complaints: List of complaint dictionaries with 'text' and 'user_id'
        
        Returns:
            List of predictions
        """
        results = []
        for complaint in complaints:
            result = self.classify_complaint(
                complaint.get('text', ''),
                complaint.get('user_id', 1)
            )
            results.append(result)
        
        return results
    
    def save_prediction(self, prediction: Dict) -> int:
        """Save prediction to database"""
        if prediction.get('status') != 'success':
            logger.warning("Not saving failed prediction to database")
            return None
        
        # Connect to database
        if not self.db_manager:
            self.db_manager = ComplaintDatabaseManager(DB_CONFIG)
        
        complaint_data = {
            'user_id': prediction['user_id'],
            'department': prediction['department']['name'],
            'language': prediction['language'],
            'complaint_text': prediction['original_text'],
            'priority': prediction['urgency']['priority'],
            'sentiment': prediction['sentiment'],
            'urgency': prediction['urgency']
        }
        
        return self.db_manager.save_complaint(complaint_data)


# ============================================
# REST API WRAPPER (Flask Integration)
# ============================================
class ComplaintAPIWrapper:
    """Flask API wrapper for inference engine"""
    
    def __init__(self):
        self.engine = ComplaintInferenceEngine(MODEL_DIR)
    
    def predict(self, complaint_text: str, user_id: int = 1, save_to_db: bool = True) -> Dict:
        """
        API endpoint for single complaint classification
        
        Returns:
            JSON-serializable dictionary
        """
        prediction = self.engine.classify_complaint(complaint_text, user_id)
        
        # Save to database if successful
        if save_to_db and prediction.get('status') == 'success':
            complaint_id = self.engine.save_prediction(prediction)
            prediction['complaint_id'] = complaint_id
        
        return prediction
    
    def predict_batch(self, complaints: List[Dict], save_to_db: bool = True) -> Dict:
        """
        API endpoint for batch classification
        
        Args:
            complaints: List of {'text': str, 'user_id': int}
        
        Returns:
            List of predictions
        """
        results = []
        for complaint in complaints:
            result = self.predict(
                complaint.get('text', ''),
                complaint.get('user_id', 1),
                save_to_db
            )
            results.append(result)
        
        return {
            'total': len(results),
            'successful': sum(1 for r in results if r.get('status') == 'success'),
            'failed': sum(1 for r in results if r.get('status') == 'failed'),
            'predictions': results
        }
    
    def get_complaint_stats(self) -> Dict:
        """Get statistics about classifications"""
        return {
            'model_loaded': self.engine.model is not None,
            'device': str(self.engine.device),
            'departments_supported': list(self.engine.department_encoder.classes_),
            'timestamp': datetime.now().isoformat()
        }


# ============================================
# EXAMPLE USAGE
# ============================================
def example_usage():
    """Example of using the inference engine"""
    
    print("\n" + "=" * 70)
    print("COMPLAINT CLASSIFICATION INFERENCE")
    print("=" * 70)
    
    # Initialize engine
    api = ComplaintAPIWrapper()
    
    # Test complaints in different languages
    test_complaints = [
        {
            'text': 'Power outage in my area since last 12 hours. Very frustrating situation.',
            'user_id': 1,
            'language': 'English'
        },
        {
            'text': 'बीते 2 दिनों से बिजली नहीं आ रही है। यह बहुत गंभीर समस्या है।',
            'user_id': 2,
            'language': 'Hindi'
        },
        {
            'text': 'வெள்ளம் காரணமாக சாலை முற்றிலும் சேதமடைந்துள்ளது.',
            'user_id': 3,
            'language': 'Tamil'
        }
    ]
    
    print(f"\n🔍 Processing {len(test_complaints)} complaints...\n")
    
    for complaint in test_complaints:
        print(f"📝 Original: {complaint['text'][:60]}...")
        
        result = api.predict(complaint['text'], complaint['user_id'])
        
        if result.get('status') == 'success':
            print(f"✅ Status: Success")
            print(f"   Sentiment: {result['sentiment']['label']} "
                  f"(Confidence: {result['sentiment']['confidence']:.2%})")
            print(f"   Department: {result['department']['name']} "
                  f"(Confidence: {result['department']['confidence']:.2%})")
            print(f"   Priority: {result['urgency']['priority']} "
                  f"(Urgency: {result['urgency']['score']:.2f})")
            if result.get('complaint_id'):
                print(f"   💾 Saved to DB with ID: {result['complaint_id']}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        print()


if __name__ == "__main__":
    example_usage()
