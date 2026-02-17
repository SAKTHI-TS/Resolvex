"""
Data Preprocessing Pipeline for Complaint Classification
- Load CSV datasets from all departments
- Handle multiple languages (English, Hindi, Tamil)
- Preprocess and prepare data for LSTM training
- Create train/validation/test splits
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re
import unicodedata
from tqdm import tqdm
import joblib

# ============================================
# CONFIGURATION
# ============================================
DATASET_BASE_PATH = r"d:\Majorproject\resolvex-citizen-connect-main\documents\dataset"

DEPARTMENT_MAPPING = {
    'Education Services': 'Education Services',
    'Health Services': 'Health Services',
    'Municipal Administration': 'Municipal Administration',
    'Publicworks': 'Public Works',
    'Transport Services': 'Transport Services',
    'Watersupply': 'Water Supply',
    'electricity': 'Electricity',
    'Sanitation & Waste Management': 'Sanitation & Waste Management'
}

LANGUAGE_MAPPING = {
    'english': 'English',
    'English': 'English',
    'hindi': 'Hindi',
    'Hindi': 'Hindi',
    'tamil': 'Tamil',
    'Tamil': 'Tamil'
}

SENTIMENT_THRESHOLDS = {
    # (lower_bound, upper_bound): sentiment_label, urgency_multiplier
    (0.0, 0.33): ('Positive', 0.3),
    (0.33, 0.66): ('Neutral', 0.6),
    (0.66, 1.0): ('Negative', 0.95)
}


# ============================================
# DATA CLEANING & PREPROCESSING
# ============================================
class TextPreprocessor:
    """Text preprocessing for multilingual complaint data"""
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Unicode characters"""
        return unicodedata.normalize('NFKD', text)
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """Remove extra whitespaces"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def remove_special_characters(text: str, keep_tamil=False, keep_hindi=False) -> str:
        """Remove special characters while preserving language scripts"""
        # Keep alphanumeric, spaces, and language-specific characters
        if keep_tamil:
            # Tamil Unicode range: U+0B80 to U+0BFF
            text = re.sub(r'[^\w\s\u0B80-\u0BFF]', '', text)
        elif keep_hindi:
            # Hindi/Devanagari Unicode range: U+0900 to U+097F
            text = re.sub(r'[^\w\s\u0900-\u097F]', '', text)
        else:
            text = re.sub(r'[^\w\s]', '', text)
        return text
    
    @staticmethod
    def clean_complaint_text(text: str, language: str = 'en') -> str:
        """Comprehensive text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Normalize
        text = TextPreprocessor.normalize_unicode(text)
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\d{10}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
        
        # Clean based on language
        if language.lower() == 'ta':
            text = TextPreprocessor.remove_special_characters(text, keep_tamil=True)
        elif language.lower() == 'hi':
            text = TextPreprocessor.remove_special_characters(text, keep_hindi=True)
        else:
            text = TextPreprocessor.remove_special_characters(text, keep_tamil=False, keep_hindi=False)
        
        # Remove extra whitespace
        text = TextPreprocessor.remove_extra_whitespace(text)
        
        return text if len(text) > 2 else ""


# ============================================
# KEYWORD-BASED URGENCY SCORING
# ============================================
class UrgeneyAnalyzer:
    """Analyze urgency based on keywords and sentiment"""
    
    URGENCY_KEYWORDS = {
        'en': {
            'critical': ['emergency', 'urgent', 'critical', 'severe', 'fatal', 'dead', 'death', 'injury', 'wounded', 'fire', 'outbreak'],
            'high': ['danger', 'risk', 'fault', 'fault', 'broken', 'damaged', 'not working', 'not functioning', 'urgent'],
            'medium': ['delay', 'slow', 'pending', 'waiting', 'stuck', 'blocked'],
            'low': ['suggestion', 'feedback', 'improvement', 'enhancement']
        },
        'hi': {
            'critical': ['आपातकाल', 'जरूरी', 'गंभीर', 'मौत', 'आग', 'चोट', 'रोग'],
            'high': ['खतरा', 'जोखिम', 'टूटा', 'क्षतिग्रस्त', 'काम नहीं'],
            'medium': ['विलंब', 'धीमा', 'लंबित', 'प्रतीक्षा'],
            'low': ['सुझाव', 'प्रतिक्रिया']
        },
        'ta': {
            'critical': ['அவசரம்', 'கடுமை', 'மரணம்', 'தீ', 'காயம்', 'நோய்'],
            'high': ['ஆபத்து', 'அபாயம்', 'உடைந்த', 'பழுது', 'வேலை செய்யவில்லை'],
            'medium': ['தாமதம்', 'மெதுவாக', 'நிலுவையில்', 'காத்திருக்க'],
            'low': ['பரামரிசு', 'கருத்து']
        }
    }
    
    SENTIMENT_TO_URGENCY = {
        'Positive': 0.2,
        'Neutral': 0.6,
        'Negative': 0.9
    }
    
    @staticmethod
    def calculate_urgency(text: str, sentiment_score: float, language: str = 'en') -> float:
        """
        Calculate urgency score (0.0 to 1.0)
        
        Factors:
        1. Sentiment strength
        2. Keyword presence
        3. Duration mentions
        """
        text_lower = text.lower()
        urgency_score = 0.0
        
        # Base score from sentiment
        if 0.0 <= sentiment_score <= 0.33:
            urgency_score += 0.1
        elif 0.33 < sentiment_score <= 0.66:
            urgency_score += 0.5
        else:
            urgency_score += 0.8
        
        # Check for urgency keywords
        lang_code = 'hi' if language.lower() == 'hindi' else ('ta' if language.lower() == 'tamil' else 'en')
        keywords = UrgeneyAnalyzer.URGENCY_KEYWORDS.get(lang_code, UrgeneyAnalyzer.URGENCY_KEYWORDS['en'])
        
        if any(keyword in text_lower for keyword in keywords.get('critical', [])):
            urgency_score = min(1.0, urgency_score + 0.3)
        elif any(keyword in text_lower for keyword in keywords.get('high', [])):
            urgency_score = min(1.0, urgency_score + 0.2)
        elif any(keyword in text_lower for keyword in keywords.get('medium', [])):
            urgency_score = min(1.0, urgency_score + 0.1)
        
        # Check for duration mentions
        if re.search(r'\d+\s*(day|week|month|hour|minute|दिन|सप्ताह|महीना|घंटा|நாள்|வாரம்|மாதம்|மணிநேரம்)', text_lower):
            urgency_score = min(1.0, urgency_score + 0.15)
        
        return min(1.0, max(0.0, urgency_score))


# ============================================
# DATASET LOADER
# ============================================
class ComplaintDatasetLoader:
    """Load and combine datasets from all departments"""
    
    @staticmethod
    def get_all_csv_files() -> Dict[str, List[str]]:
        """Find all CSV files by department"""
        file_structure = {}
        
        for folder in os.listdir(DATASET_BASE_PATH):
            folder_path = os.path.join(DATASET_BASE_PATH, folder)
            if os.path.isdir(folder_path):
                csv_files = []
                
                # Check if language folders exist (like electricity/)
                if any(os.path.isdir(os.path.join(folder_path, sub)) for sub in os.listdir(folder_path)):
                    # Language subdirectories
                    for lang_folder in os.listdir(folder_path):
                        lang_path = os.path.join(folder_path, lang_folder)
                        if os.path.isdir(lang_path):
                            for file in os.listdir(lang_path):
                                if file.endswith('.csv'):
                                    csv_files.append(os.path.join(lang_path, file))
                else:
                    # Direct CSV files
                    for file in os.listdir(folder_path):
                        if file.endswith('.csv'):
                            csv_files.append(os.path.join(folder_path, file))
                
                if csv_files:
                    file_structure[folder] = csv_files
        
        return file_structure
    
    @staticmethod
    def extract_language_from_filename(filename: str) -> str:
        """Extract language from filename"""
        filename_lower = filename.lower()
        if 'hindi' in filename_lower or '_hi' in filename_lower:
            return 'Hindi'
        elif 'tamil' in filename_lower or '_ta' in filename_lower:
            return 'Tamil'
        else:
            return 'English'
    
    @staticmethod
    def load_and_combine_datasets(sample_size: int = 5000) -> pd.DataFrame:
        """
        Load all CSV files and combine into single DataFrame
        """
        all_data = []
        file_structure = ComplaintDatasetLoader.get_all_csv_files()
        
        print(f"Found {len(file_structure)} departments")
        
        for department, csv_files in file_structure.items():
            print(f"\n📂 Processing {department}...")
            
            for csv_file in csv_files:
                try:
                    print(f"  📄 Loading {os.path.basename(csv_file)}...")
                    
                    df = pd.read_csv(csv_file)
                    
                    # Extract language
                    language = ComplaintDatasetLoader.extract_language_from_filename(csv_file)
                    df['language'] = language
                    
                    # Map department name
                    mapped_dept = DEPARTMENT_MAPPING.get(department, department)
                    df['department'] = mapped_dept
                    
                    # Clean complaint text
                    print(f"    🧹 Cleaning text ({language})...")
                    df['complaint_text'] = df['complaint_text'].apply(
                        lambda x: TextPreprocessor.clean_complaint_text(x, language)
                    )
                    
                    # Remove empty complaints
                    df = df[df['complaint_text'].str.len() > 5]
                    
                    # Calculate urgency if not present
                    if 'urgency_score' not in df.columns:
                        print(f"    📊 Calculating urgency scores...")
                        df['urgency_score'] = df.apply(
                            lambda row: UrgeneyAnalyzer.calculate_urgency(
                                row['complaint_text'],
                                float(row.get('urgency_score', 0.5)),
                                language
                            ),
                            axis=1
                        )
                    else:
                        df['urgency_score'] = df['urgency_score'].astype(float)
                    
                    # Add sentiment if not present
                    if 'sentiment' not in df.columns:
                        print(f"    😊 Inferring sentiment from urgency scores...")
                        df['sentiment'] = df['urgency_score'].apply(
                            lambda x: UrgeneyAnalyzer.SENTIMENT_TO_URGENCY.keys()[
                                min(2, int(x * 3))
                            ] if 0.0 <= x <= 1.0 else 'Neutral'
                        )
                    
                    # Sample if too large
                    if len(df) > sample_size:
                        df = df.sample(n=sample_size, random_state=42)
                    
                    all_data.append(df)
                    print(f"    ✅ Loaded {len(df)} records")
                    
                except Exception as e:
                    print(f"    ❌ Error loading {csv_file}: {str(e)}")
                    continue
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Clean duplicates
        combined_df = combined_df.drop_duplicates(subset=['complaint_text'], keep='first')
        
        print(f"\n✅ Total records loaded: {len(combined_df)}")
        print(f"📊 Departments: {combined_df['department'].unique()}")
        print(f"🌍 Languages: {combined_df['language'].unique()}")
        print(f"😊 Sentiments: {combined_df.get('sentiment', pd.Series()).unique()}")
        
        return combined_df


# ============================================
# DATASET PREPARATION
# ============================================
class DatasetPreparer:
    """Prepare data for model training"""
    
    @staticmethod
    def prepare_data(df: pd.DataFrame, test_size=0.15, val_size=0.1) -> Tuple:
        """
        Prepare and split dataset
        
        Returns:
            train_df, val_df, test_df, sentiment_encoder, department_encoder
        """
        # Ensure required columns
        required_cols = ['complaint_text', 'department', 'sentiment', 'urgency_score', 'language']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Encode labels
        sentiment_encoder = LabelEncoder()
        department_encoder = LabelEncoder()
        
        df['sentiment_encoded'] = sentiment_encoder.fit_transform(df['sentiment'])
        df['department_encoded'] = department_encoder.fit_transform(df['department'])
        
        # Split: train -> (actual_train, val), test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['department']
        )
        
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=val_size / (1 - test_size),
            random_state=42,
            stratify=train_val_df['department']
        )
        
        print(f"\n📊 Data Split:")
        print(f"  🚅 Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  ✅ Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  🧪 Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df, sentiment_encoder, department_encoder
    
    @staticmethod
    def save_encoders(sentiment_encoder, department_encoder, output_dir):
        """Save encoders for later use"""
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(sentiment_encoder, os.path.join(output_dir, 'sentiment_encoder.pkl'))
        joblib.dump(department_encoder, os.path.join(output_dir, 'department_encoder.pkl'))
        print(f"✅ Encoders saved to {output_dir}")


if __name__ == "__main__":
    print("=" * 60)
    print("COMPLAINT DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Load datasets
    print("\n🔄 Loading datasets...")
    combined_df = ComplaintDatasetLoader.load_and_combine_datasets(sample_size=5000)
    
    # Display statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"  Total Records: {len(combined_df)}")
    print(f"\n  By Department:")
    print(combined_df['department'].value_counts())
    print(f"\n  By Language:")
    print(combined_df['language'].value_counts())
    print(f"\n  Sentiment Distribution:")
    print(combined_df['sentiment'].value_counts())
    print(f"\n  Urgency Score Stats:")
    print(combined_df['urgency_score'].describe())
    
    # Prepare data
    output_dir = r"d:\Majorproject\resolvex-citizen-connect-main\src\ml\data"
    train_df, val_df, test_df, sentiment_enc, dept_enc = DatasetPreparer.prepare_data(combined_df)
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    DatasetPreparer.save_encoders(sentiment_enc, dept_enc, output_dir)
    
    print(f"\n✅ Data prepared and saved to {output_dir}")
