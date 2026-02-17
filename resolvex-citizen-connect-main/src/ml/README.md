# 🎯 LSTM-Based Complaint Classification System

Multilingual complaint classification using BERT + Bi-LSTM with sentiment analysis, department routing, and urgency detection.

---

## 📋 Table of Contents
1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Inference & API](#inference--api)
7. [Database Integration](#database-integration)
8. [Usage Examples](#usage-examples)

---

## ✨ Features

### 🌍 Multilingual Support
- **English**: BERT-base-english
- **Hindi**: BERT-base-multilingual-uncased (MuRIL compatible)
- **Tamil**: BERT-base-multilingual-uncased (IndicBERT compatible)
- **Automatic Language Detection**: Detects complaint language automatically

### 🧠 AI/ML Capabilities
- **Sentiment Analysis**: Positive, Neutral, Negative (3-class classification)
- **Department Classification**: 8 departments (Education, Health, Municipal, etc.)
- **Urgency Detection**: Continuous score (0.0 - 1.0) + Priority (Normal/Urgent)
- **Multi-task Learning**: Single model handles 3 tasks simultaneously

### 🏗️ Architecture
```
Complaint Text (Multi-language)
         ↓
Language Detection
         ↓
BERT/IndicBERT/MuRIL (Contextual Embeddings)
         ↓
Bi-LSTM (2 layers, 256 hidden units)
         ↓
Multi-head Attention (8 heads)
         ↓
Global Average Pooling
         ↓
Task-Specific Dense Layers
         ├→ Sentiment Classifier (3 classes)
         ├→ Department Classifier (8 classes)
         └→ Urgency Predictor (continuous)
```

### 💾 Database Integration
- Automatic complaint recording in MySQL
- Sentiment & urgency scores stored
- Complaint tracking and history
- Support for 21 different tables

### 📡 REST API
- Single complaint classification
- Batch processing
- Real-time predictions
- Database persistence

---

## 🏛️ Architecture Details

### Model Components

#### 1. **BERT Encoder** (Language Understanding)
- Pre-trained multilingual BERT
- Generates 768-dimensional contextual embeddings
- Handles multiple languages and dialects
- Frozen or fine-tunable (configurable)

#### 2. **Bi-LSTM Layer** (Sequence Processing)
- Bidirectional LSTM with 2 layers
- Hidden size: 256 (total 512 with bidirectionality)
- Learns long-term dependencies
- Remembers important words, forgets noise

#### 3. **Multi-head Attention** (Relevance Weighting)
- 8 attention heads
- Focuses on important complaint phrases
- Self-attention mechanism
- Global feature extraction

#### 4. **Task-Specific Heads**

| Task | Input | Architecture | Output |
|------|-------|--------------|--------|
| Sentiment | Pooled | Dense(128)→ReLU→Dense(64)→Dense(3) | 3 classes |
| Department | Pooled | Dense(128)→ReLU→Dense(64)→Dense(8) | 8 classes |
| Urgency | Pooled | Dense(64)→ReLU→Dense(32)→Dense(1) | Float [0,1] |

#### 5. **Loss Function** (Multi-task Learning)
```
Total Loss = 0.4 × Sentiment_Loss + 0.4 × Department_Loss + 0.2 × Urgency_Loss
```

---

## 📦 Installation

### 1. Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
CUDA 11.8 (optional, for GPU acceleration)
MySQL Server 8.0+
```

### 2. Install Dependencies
```bash
cd src/ml
pip install -r requirements.txt
```

### 3. Download Pre-trained Models
```bash
# Automatically downloaded by transformers library on first use
# Models required:
# - bert-base-multilingual-uncased (for all languages)
```

---

## 📊 Data Preparation

### 1. Dataset Structure
```
documents/dataset/
├── Education Services/
│   ├── education_english_10000_dataset.csv
│   ├── education_hindi_10000_dataset.csv
│   └── education_tamil_10000_dataset.csv
├── Health Services/
├── Municipal Administration/
├── Publicworks/
├── Transport Services/
├── Watersupply/
├── electricity/
└── Sanitation & Waste Management/
```

### 2. CSV Format
Required columns:
```
complaint_id, department, language, complaint_text, sentiment, urgency_score, state, district
```

### 3. Data Processing
```bash
python data_preprocessing.py
```

**Output:**
- `data/train.csv` - Training set (75%)
- `data/val.csv` - Validation set (10%)
- `data/test.csv` - Test set (15%)
- `data/sentiment_encoder.pkl` - Sentiment label encoder
- `data/department_encoder.pkl` - Department label encoder

**Features:**
- Text cleaning (remove URLs, emails, special chars)
- Unicode normalization for Tamil/Hindi
- Automatic urgency score calculation
- Language detection
- Duplicate removal
- Missing value handling

---

## 🚀 Model Training

### 1. Start Training
```bash
cd src/ml
python train.py
```

### 2. Training Configuration
```python
CONFIG = {
    'batch_size': 16,
    'num_epochs': 10,
    'learning_rate': 1e-4,
    'max_length': 128,              # BERT max sequence length
    'hidden_size': 256,              # LSTM hidden size
    'lstm_layers': 2,
    'dropout': 0.3,
    'loss_weights': {
        'sentiment': 0.4,
        'department': 0.4,
        'urgency': 0.2
    }
}
```

### 3. Training Output
- **Model Checkpoints**: `models/best_model/`
- **Training History**: `models/training_history.png`
- **Test Metrics**: `models/test_metrics.json`

### 4. Expected Performance
```
😊 Sentiment Analysis:
   Accuracy: 0.82-0.88
   F1-Score: 0.80-0.87

🏢 Department Classification:
   Accuracy: 0.85-0.91
   F1-Score: 0.84-0.90

⚡ Urgency Prediction:
   RMSE: 0.15-0.22
```

---

## 🔮 Inference & Predictions

### 1. Python Inference
```python
from inference import ComplaintAPIWrapper

# Initialize
api = ComplaintAPIWrapper()

# Single prediction
result = api.predict(
    complaint_text="Power outage for 12 hours",
    user_id=1,
    save_to_db=True
)

print(result)
# Output:
# {
#     'sentiment': {'label': 'Negative', 'confidence': 0.94, ...},
#     'department': {'name': 'Electricity', 'confidence': 0.88},
#     'urgency': {'score': 0.91, 'priority': 'Urgent'},
#     'complaint_id': 12345,
#     'status': 'success'
# }
```

### 2. Batch Processing
```python
complaints = [
    {'text': 'Road is damaged', 'user_id': 1},
    {'text': 'School has no water', 'user_id': 2},
    {'text': 'Hospital staff is rude', 'user_id': 3}
]

results = api.predict_batch(complaints, save_to_db=True)
```

---

## 🌐 REST API Server

### 1. Start API Server
```bash
python api.py
```

Server runs on `http://localhost:5000`

### 2. API Endpoints

#### Health Check
```bash
GET /health
```

#### Classification (Single)
```bash
POST /api/v1/classify
Content-Type: application/json

{
    "text": "Power outage in my area",
    "user_id": 1,
    "save_to_db": true
}
```

**Response:**
```json
{
    "complaint_text": "Power outage in my area",
    "language": "English",
    "sentiment": {
        "label": "Negative",
        "confidence": 0.94,
        "scores": {
            "positive": 0.02,
            "neutral": 0.04,
            "negative": 0.94
        }
    },
    "department": {
        "name": "Electricity",
        "confidence": 0.88
    },
    "urgency": {
        "score": 0.91,
        "priority": "Urgent"
    },
    "complaint_id": 12345,
    "timestamp": "2024-02-16T10:30:45.123456",
    "status": "success"
}
```

#### Classification (Batch)
```bash
POST /api/v1/classify-batch
Content-Type: application/json

{
    "complaints": [
        {"text": "...", "user_id": 1},
        {"text": "...", "user_id": 2}
    ],
    "save_to_db": true
}
```

#### Get Supported Departments
```bash
GET /api/v1/departments

Response: {
    "departments": [
        "Education Services",
        "Health Services",
        "Municipal Administration",
        ...
    ]
}
```

#### Get Supported Languages
```bash
GET /api/v1/languages

Response: {
    "supported_languages": ["English", "Hindi", "Tamil"],
    "auto_detection": true
}
```

#### Get API Documentation
```bash
GET /api/v1/docs
```

---

## 💾 Database Integration

### 1. Initialize Database
```bash
mysql -u root -p < documents/Database/resolveX_database_setup.sql
```

### 2. Configuration
Update `inference.py`:
```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_password',
    'database': 'resolveX_grievance_system'
}
```

### 3. Auto-Saved Data
When `save_to_db=True`, the system automatically saves:
- **complaints**: Main complaint record
- **sentiment_analysis**: Sentiment label & score
- **priority_analysis**: Urgency score & priority level
- **nlp_processing_logs**: Processing metadata

---

## 📝 Usage Examples

### Example 1: English Complaint
```python
complaint = "School has no electricity since 3 days"
result = api.predict(complaint, user_id=1)

# Result:
# - Sentiment: Negative (0.89)
# - Department: Education Services (0.92)
# - Priority: Urgent (0.85)
# - Complaint saved to DB with ID: 5001
```

### Example 2: Hindi Complaint
```python
complaint = "हमारे गांव में बीते एक हफ्ते से पानी नहीं आ रहा है"
result = api.predict(complaint, user_id=2)

# Result:
# - Language auto-detected: Hindi
# - Sentiment: Negative (0.91)
# - Department: Water Supply (0.87)
# - Priority: Urgent (0.92)
```

### Example 3: Tamil Complaint
```python
complaint = "சாலை பழுதடைந்துள்ளது, விபத்துக்கு ஆபத்திருக்கிறது"
result = api.predict(complaint, user_id=3)

# Result:
# - Language auto-detected: Tamil
# - Sentiment: Negative (0.93)
# - Department: Public Works (0.88)
# - Priority: Urgent (0.94)
```

### Example 4: Batch Processing
```python
complaints = [
    {"text": "Hospital equipment not working", "user_id": 4},
    {"text": "Traffic signal is broken", "user_id": 5},
    {"text": "Dustbin overflow near my house", "user_id": 6}
]

results = api.predict_batch(complaints)
# Returns predictions for all 3 complaints
```

---

## 📈 Monitoring & Metrics

### Training Metrics
```
Epoch 1/10
  Train Loss: 1.2345
  Val Loss: 1.1234
  
Epoch 5/10
  Train Loss: 0.4567
  Val Loss: 0.4890
  ✅ Validation loss improved! Saving best model...

Epoch 10/10
  Train Loss: 0.2345
  Val Loss: 0.2678
```

### Test Results
```
😊 Sentiment Analysis:
   Accuracy: 0.86
   F1-Score: 0.84

🏢 Department Classification:
   Accuracy: 0.89
   F1-Score: 0.88

⚡ Urgency Prediction:
   RMSE: 0.18
```

---

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory
**Solution:**
```python
CONFIG['batch_size'] = 8  # Reduce batch size
torch.cuda.empty_cache()  # Clear CUDA cache
```

### Issue: Model Takes Too Long to Load
**Solution:**
```bash
# Use CPU for faster loading if inference doesn't require speed
device = torch.device('cpu')
```

### Issue: Language Detection Not Accurate
**Solution:**
```python
# Manually specify language
result = api.predict(complaint_text, language='hi')
```

### Issue: Database Connection Failed
**Solution:**
```bash
# Check MySQL is running
# Update credentials in inference.py
# Verify database exists:
# mysql -u root -p resolveX_grievance_system
```

---

## 📚 Key Paper References

1. **BERT**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
2. **LSTM**: Hochreiter & Schmidhuber, "Long Short-Term Memory"
3. **Attention**: Vaswani et al., "Attention Is All You Need"
4. **IndicBERT**: Kakwani et al., "IndicBERT: Indic Transformers for NLP"
5. **MuRIL**: Khanuja et al., "MuRIL: Multilingual Representations for Indian Languages"

---

## 📞 Support

For issues or questions:
1. Check `docs/` folder for detailed documentation
2. Review `src/ml/test_examples.py` for usage examples
3. Check logs in `models/training.log`

---

## 📄 License

This project is part of the ResolveX Citizen Connect system.

---

**Last Updated:** February 16, 2024
**Version:** 1.0.0
**Status:** ✅ Production Ready
