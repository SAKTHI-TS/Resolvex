"""
LSTM-Based Complaint Classification System
Multilingual Support: English, Hindi, Tamil
Sentiment Analysis + Urgency + Department Classification
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from pathlib import Path
from typing import Tuple, Dict, List
import joblib

# ============================================
# DEVICE CONFIGURATION
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================
# LANGUAGE MODELS CONFIGURATION
# ============================================
LANGUAGE_MODELS = {
    'en': {
        'model_name': 'bert-base-english',
        'tokenizer_name': 'bert-base-english'
    },
    'hi': {
        'model_name': 'bert-base-multilingual-uncased',
        'tokenizer_name': 'bert-base-multilingual-uncased'
    },
    'ta': {
        'model_name': 'bert-base-multilingual-uncased',
        'tokenizer_name': 'bert-base-multilingual-uncased'
    }
}

SENTIMENT_LABELS = {'Positive': 0, 'Neutral': 1, 'Negative': 2}
SENTIMENT_LABEL_REVERSE = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}
PRIORITY_LABELS = {'Normal': 0, 'Urgent': 1}
PRIORITY_LABEL_REVERSE = {0: 'Normal', 1: 'Urgent'}

DEPARTMENTS = [
    'Education Services',
    'Health Services',
    'Municipal Administration',
    'Public Works',
    'Transport Services',
    'Water Supply',
    'Electricity',
    'Sanitation & Waste Management'
]

# ============================================
# CUSTOM DATASET CLASS
# ============================================
class ComplaintDataset(Dataset):
    """Dataset for complaint handling with multilingual support"""
    
    def __init__(self, texts, sentiments, departments, urgency_scores, language, 
                 tokenizer, max_length=128):
        self.texts = texts
        self.sentiments = sentiments
        self.departments = departments
        self.urgency_scores = urgency_scores
        self.language = language
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize with BERT tokenizer
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'sentiment': torch.tensor(self.sentiments[idx], dtype=torch.long),
            'department': torch.tensor(self.departments[idx], dtype=torch.long),
            'urgency_score': torch.tensor(self.urgency_scores[idx], dtype=torch.float32)
        }


# ============================================
# LSTM MODEL ARCHITECTURE
# ============================================
class ComplaintLSTMClassifier(nn.Module):
    """
    Multi-task LSTM model for complaint classification
    Tasks:
    1. Sentiment Analysis (3 classes: Positive, Neutral, Negative)
    2. Department Classification (8 departments)
    3. Urgency/Priority Prediction
    """
    
    def __init__(self, bert_model, hidden_size=256, lstm_layers=2, dropout=0.3, 
                 num_departments=8, num_sentiments=3):
        super(ComplaintLSTMClassifier, self).__init__()
        
        # BERT encoder (frozen or fine-tunable)
        self.bert = bert_model
        self.bert_hidden_size = bert_model.config.hidden_size
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # ============================================
        # LSTM for sequence processing
        # ============================================
        self.lstm = nn.LSTM(
            input_size=self.bert_hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        self.lstm_hidden_size = hidden_size * 2  # bidirectional
        
        # ============================================
        # ATTENTION MECHANISM
        # ============================================
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # ============================================
        # TASK 1: SENTIMENT ANALYSIS
        # ============================================
        self.sentiment_dense1 = nn.Linear(self.lstm_hidden_size, 128)
        self.sentiment_dense2 = nn.Linear(128, 64)
        self.sentiment_classifier = nn.Linear(64, num_sentiments)
        self.sentiment_dropout = nn.Dropout(dropout)
        
        # ============================================
        # TASK 2: DEPARTMENT CLASSIFICATION
        # ============================================
        self.department_dense1 = nn.Linear(self.lstm_hidden_size, 128)
        self.department_dense2 = nn.Linear(128, 64)
        self.department_classifier = nn.Linear(64, num_departments)
        self.department_dropout = nn.Dropout(dropout)
        
        # ============================================
        # TASK 3: URGENCY/PRIORITY PREDICTION
        # ============================================
        self.urgency_dense1 = nn.Linear(self.lstm_hidden_size, 64)
        self.urgency_dense2 = nn.Linear(64, 32)
        self.urgency_predictor = nn.Linear(32, 1)
        self.urgency_dropout = nn.Dropout(dropout)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model
        
        Args:
            input_ids: Tokenized input (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
        
        Returns:
            sentiment_logits: (batch_size, num_sentiments)
            department_logits: (batch_size, num_departments)
            urgency_scores: (batch_size, 1)
        """
        
        # ============================================
        # BERT Encoding
        # ============================================
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state from BERT
        bert_embeddings = bert_output.last_hidden_state  # (batch_size, seq_length, bert_hidden)
        bert_embeddings = self.dropout_layer(bert_embeddings)
        
        # ============================================
        # LSTM Processing
        # ============================================
        lstm_output, (hidden, cell) = self.lstm(bert_embeddings)
        # lstm_output shape: (batch_size, seq_length, lstm_hidden_size * 2)
        
        # ============================================
        # ATTENTION MECHANISM
        # ============================================
        attention_output, attention_weights = self.attention(
            query=lstm_output,
            key=lstm_output,
            value=lstm_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Global average pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(attention_output.shape).float()
        pooled = (attention_output * mask_expanded).sum(1) / mask_expanded.sum(1)
        # pooled shape: (batch_size, lstm_hidden_size * 2)
        
        # ============================================
        # SENTIMENT ANALYSIS HEAD
        # ============================================
        sentiment_features = self.relu(self.sentiment_dense1(pooled))
        sentiment_features = self.sentiment_dropout(sentiment_features)
        sentiment_features = self.relu(self.sentiment_dense2(sentiment_features))
        sentiment_logits = self.sentiment_classifier(sentiment_features)
        
        # ============================================
        # DEPARTMENT CLASSIFICATION HEAD
        # ============================================
        dept_features = self.relu(self.department_dense1(pooled))
        dept_features = self.department_dropout(dept_features)
        dept_features = self.relu(self.department_dense2(dept_features))
        department_logits = self.department_classifier(dept_features)
        
        # ============================================
        # URGENCY/PRIORITY PREDICTION HEAD
        # ============================================
        urgency_features = self.relu(self.urgency_dense1(pooled))
        urgency_features = self.urgency_dropout(urgency_features)
        urgency_features = self.relu(self.urgency_dense2(urgency_features))
        urgency_scores = self.sigmoid(self.urgency_predictor(urgency_features))
        
        return sentiment_logits, department_logits, urgency_scores


# ============================================
# MODEL TRAINER CLASS
# ============================================
class ComplaintModelTrainer:
    """Trainer for the complaint classification model"""
    
    def __init__(self, model, device, learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Loss functions for each task
        self.sentiment_loss_fn = nn.CrossEntropyLoss()
        self.department_loss_fn = nn.CrossEntropyLoss()
        self.urgency_loss_fn = nn.MSELoss()
        
    def train_epoch(self, train_loader, alpha=0.4, beta=0.4, gamma=0.2):
        """
        Train for one epoch with multi-task learning
        
        Loss = alpha * sentiment_loss + beta * department_loss + gamma * urgency_loss
        """
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            sentiments = batch['sentiment'].to(self.device)
            departments = batch['department'].to(self.device)
            urgency_scores = batch['urgency_score'].to(self.device).unsqueeze(1)
            
            # Forward pass
            sentiment_logits, department_logits, urgency_pred = self.model(
                input_ids, attention_mask
            )
            
            # Compute losses
            sentiment_loss = self.sentiment_loss_fn(sentiment_logits, sentiments)
            department_loss = self.department_loss_fn(department_logits, departments)
            urgency_loss = self.urgency_loss_fn(urgency_pred, urgency_scores)
            
            # Combined loss with weights
            loss = alpha * sentiment_loss + beta * department_loss + gamma * urgency_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                sentiments = batch['sentiment'].to(self.device)
                departments = batch['department'].to(self.device)
                urgency_scores = batch['urgency_score'].to(self.device).unsqueeze(1)
                
                sentiment_logits, department_logits, urgency_pred = self.model(
                    input_ids, attention_mask
                )
                
                sentiment_loss = self.sentiment_loss_fn(sentiment_logits, sentiments)
                department_loss = self.department_loss_fn(department_logits, departments)
                urgency_loss = self.urgency_loss_fn(urgency_pred, urgency_scores)
                
                loss = 0.4 * sentiment_loss + 0.4 * department_loss + 0.2 * urgency_loss
                total_loss += loss.item()
        
        return total_loss / len(val_loader)


# ============================================
# INFERENCE CLASS
# ============================================
class ComplaintPredictor:
    """Inference module for making predictions on new complaints"""
    
    def __init__(self, model, tokenizer, device, department_encoder):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.department_encoder = department_encoder
        self.model.eval()
    
    def predict(self, complaint_text: str, language: str = 'en') -> Dict:
        """
        Predict sentiment, department, and urgency for a complaint
        
        Args:
            complaint_text: The complaint text
            language: Language code ('en', 'hi', 'ta')
        
        Returns:
            Dictionary with predictions and scores
        """
        with torch.no_grad():
            # Tokenize
            encoding = self.tokenizer(
                complaint_text,
                add_special_tokens=True,
                max_length=128,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Forward pass
            sentiment_logits, department_logits, urgency_pred = self.model(
                input_ids, attention_mask
            )
            
            # Process outputs
            sentiment_probs = torch.softmax(sentiment_logits, dim=1)
            sentiment_pred = torch.argmax(sentiment_logits, dim=1)
            sentiment_confidence = torch.max(sentiment_probs, dim=1).values
            
            department_probs = torch.softmax(department_logits, dim=1)
            department_pred = torch.argmax(department_logits, dim=1)
            department_confidence = torch.max(department_probs, dim=1).values
            
            urgency_score = urgency_pred.squeeze().item()
            priority = 'Urgent' if urgency_score > 0.6 else 'Normal'
            
            # Decode predictions
            sentiment_label = SENTIMENT_LABEL_REVERSE[sentiment_pred.item()]
            department_name = self.department_encoder.inverse_transform([department_pred.item()])[0]
            
            return {
                'complaint_text': complaint_text,
                'language': language,
                'sentiment': {
                    'label': sentiment_label,
                    'confidence': sentiment_confidence.item(),
                    'scores': {
                        'positive': sentiment_probs[0, 0].item(),
                        'neutral': sentiment_probs[0, 1].item(),
                        'negative': sentiment_probs[0, 2].item()
                    }
                },
                'department': {
                    'name': department_name,
                    'confidence': department_confidence.item()
                },
                'urgency': {
                    'score': urgency_score,
                    'priority': priority
                }
            }
    
    def batch_predict(self, complaint_texts: List[str], language: str = 'en') -> List[Dict]:
        """Predict on batch of complaints"""
        return [self.predict(text, language) for text in complaint_texts]


# ============================================
# UTILITY FUNCTIONS
# ============================================
def save_model(model, tokenizer, department_encoder, output_dir):
    """Save model, tokenizer, and encoders"""
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    joblib.dump(department_encoder, os.path.join(output_dir, 'department_encoder.pkl'))
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")


def load_model(model_dir, device):
    """Load model, tokenizer, and encoders"""
    from transformers import AutoTokenizer, AutoModel
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load BERT
    bert = AutoModel.from_pretrained('bert-base-multilingual-uncased')
    
    # Load department encoder
    department_encoder = joblib.load(os.path.join(model_dir, 'department_encoder.pkl'))
    
    # Create model
    model = ComplaintLSTMClassifier(bert)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location=device))
    
    return model, tokenizer, department_encoder


if __name__ == "__main__":
    print("✅ LSTM Model Architecture Loaded")
    print(f"Device: {device}")
    print(f"Available Models for Languages: {list(LANGUAGE_MODELS.keys())}")
    print(f"Supported Departments: {DEPARTMENTS}")
