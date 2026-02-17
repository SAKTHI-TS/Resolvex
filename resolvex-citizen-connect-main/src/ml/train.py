"""
Training Script for LSTM Complaint Classification Model
Trains on preprocessed complaint data with multilingual support
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
from typing import Dict, List

# Import custom modules
from lstm_model import (
    ComplaintLSTMClassifier,
    ComplaintModelTrainer,
    ComplaintDataset,
    ComplaintPredictor,
    save_model,
    load_model,
    DEPARTMENTS
)
from data_preprocessing import (
    ComplaintDatasetLoader,
    DatasetPreparer,
    TextPreprocessor
)

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    'batch_size': 16,
    'num_epochs': 10,
    'learning_rate': 1e-4,
    'max_length': 128,
    'hidden_size': 256,
    'lstm_layers': 2,
    'dropout': 0.3,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'seed': 42,
    'loss_weights': {
        'sentiment': 0.4,
        'department': 0.4,
        'urgency': 0.2
    }
}

MODELS_DIR = r"d:\Majorproject\resolvex-citizen-connect-main\src\ml\models"
DATA_DIR = r"d:\Majorproject\resolvex-citizen-connect-main\src\ml\data"

# Set seed
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])


# ============================================
# TRAINING PIPELINE
# ============================================
class TrainingPipeline:
    """Complete training pipeline for complaint classification"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = config['device']
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
        
    def load_data(self) -> tuple:
        """Load preprocessed datasets"""
        print("\n" + "=" * 70)
        print("📂 LOADING PREPROCESSED DATA")
        print("=" * 70)
        
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
        test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
        
        # Load encoders
        sentiment_encoder = joblib.load(os.path.join(DATA_DIR, 'sentiment_encoder.pkl'))
        department_encoder = joblib.load(os.path.join(DATA_DIR, 'department_encoder.pkl'))
        
        print(f"✅ Train set: {len(train_df)} samples")
        print(f"✅ Validation set: {len(val_df)} samples")
        print(f"✅ Test set: {len(test_df)} samples")
        
        return train_df, val_df, test_df, sentiment_encoder, department_encoder
    
    def create_dataloaders(self, train_df, val_df, test_df, tokenizer) -> tuple:
        """Create PyTorch DataLoaders"""
        print("\n" + "=" * 70)
        print("🔄 CREATING DATA LOADERS")
        print("=" * 70)
        
        train_dataset = ComplaintDataset(
            texts=train_df['complaint_text'].values,
            sentiments=train_df['sentiment_encoded'].values,
            departments=train_df['department_encoded'].values,
            urgency_scores=train_df['urgency_score'].values,
            language=train_df['language'].values,
            tokenizer=tokenizer,
            max_length=self.config['max_length']
        )
        
        val_dataset = ComplaintDataset(
            texts=val_df['complaint_text'].values,
            sentiments=val_df['sentiment_encoded'].values,
            departments=val_df['department_encoded'].values,
            urgency_scores=val_df['urgency_score'].values,
            language=val_df['language'].values,
            tokenizer=tokenizer,
            max_length=self.config['max_length']
        )
        
        test_dataset = ComplaintDataset(
            texts=test_df['complaint_text'].values,
            sentiments=test_df['sentiment_encoded'].values,
            departments=test_df['department_encoded'].values,
            urgency_scores=test_df['urgency_score'].values,
            language=test_df['language'].values,
            tokenizer=tokenizer,
            max_length=self.config['max_length']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        print(f"✅ Train loader: {len(train_loader)} batches")
        print(f"✅ Val loader: {len(val_loader)} batches")
        print(f"✅ Test loader: {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    def initialize_model(self):
        """Initialize LSTM model with BERT encoder"""
        print("\n" + "=" * 70)
        print("🧠 INITIALIZING MODEL")
        print("=" * 70)
        
        # Load multilingual BERT
        print("📥 Loading BERT-multilingual-uncased...")
        bert = AutoModel.from_pretrained('bert-base-multilingual-uncased')
        
        # Create LSTM classifier
        print("🏗️  Building LSTM architecture...")
        model = ComplaintLSTMClassifier(
            bert_model=bert,
            hidden_size=self.config['hidden_size'],
            lstm_layers=self.config['lstm_layers'],
            dropout=self.config['dropout'],
            num_departments=len(DEPARTMENTS),
            num_sentiments=3
        )
        
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
    
    def train(self, model, train_loader, val_loader, tokenizer, 
              department_encoder, sentiment_encoder):
        """Train the model"""
        print("\n" + "=" * 70)
        print("🚀 STARTING TRAINING")
        print("=" * 70)
        
        trainer = ComplaintModelTrainer(
            model=model,
            device=self.device,
            learning_rate=self.config['learning_rate']
        )
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            print(f"{'=' * 70}")
            
            # Training
            train_loss = trainer.train_epoch(train_loader)
            print(f"📊 Train Loss: {train_loss:.4f}")
            
            # Validation
            val_loss = trainer.validate(val_loader)
            print(f"📊 Val Loss: {val_loss:.4f}")
            
            # Track history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"✅ Validation loss improved! Saving best model...")
                save_model(model, tokenizer, department_encoder,
                          os.path.join(MODELS_DIR, 'best_model'))
            else:
                patience_counter += 1
                print(f"⚠️  No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                    break
        
        print(f"\n✅ Training completed!")
        return model
    
    def evaluate(self, model, test_loader, sentiment_encoder, department_encoder):
        """Evaluate model on test set"""
        print("\n" + "=" * 70)
        print("📈 EVALUATING ON TEST SET")
        print("=" * 70)
        
        model.eval()
        
        all_sentiments_true = []
        all_sentiments_pred = []
        all_departments_true = []
        all_departments_pred = []
        all_urgency_true = []
        all_urgency_pred = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                sentiment_logits, department_logits, urgency_pred = model(
                    input_ids, attention_mask
                )
                
                # Sentiment
                sentiment_pred = torch.argmax(sentiment_logits, dim=1)
                all_sentiments_true.extend(batch['sentiment'].cpu().numpy())
                all_sentiments_pred.extend(sentiment_pred.cpu().numpy())
                
                # Department
                department_pred = torch.argmax(department_logits, dim=1)
                all_departments_true.extend(batch['department'].cpu().numpy())
                all_departments_pred.extend(department_pred.cpu().numpy())
                
                # Urgency
                all_urgency_true.extend(batch['urgency_score'].cpu().numpy())
                all_urgency_pred.extend(urgency_pred.squeeze().cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
        
        sentiment_acc = accuracy_score(all_sentiments_true, all_sentiments_pred)
        sentiment_f1 = f1_score(all_sentiments_true, all_sentiments_pred, average='weighted')
        
        department_acc = accuracy_score(all_departments_true, all_departments_pred)
        department_f1 = f1_score(all_departments_true, all_departments_pred, average='weighted')
        
        urgency_mse = mean_squared_error(all_urgency_true, all_urgency_pred)
        urgency_rmse = np.sqrt(urgency_mse)
        
        print(f"\n😊 SENTIMENT ANALYSIS:")
        print(f"   Accuracy: {sentiment_acc:.4f}")
        print(f"   F1-Score: {sentiment_f1:.4f}")
        
        print(f"\n🏢 DEPARTMENT CLASSIFICATION:")
        print(f"   Accuracy: {department_acc:.4f}")
        print(f"   F1-Score: {department_f1:.4f}")
        
        print(f"\n⚡ URGENCY PREDICTION:")
        print(f"   RMSE: {urgency_rmse:.4f}")
        
        return {
            'sentiment_accuracy': sentiment_acc,
            'sentiment_f1': sentiment_f1,
            'department_accuracy': department_acc,
            'department_f1': department_f1,
            'urgency_rmse': urgency_rmse
        }
    
    def plot_training_history(self):
        """Plot training history"""
        print("\n📊 Plotting training history...")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epoch'], self.history['train_loss'], label='Train Loss', marker='o')
        plt.plot(self.history['epoch'], self.history['val_loss'], label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(MODELS_DIR, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {plot_path}")
        plt.close()


# ============================================
# MAIN EXECUTION
# ============================================
def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 70)
    print("🎯 COMPLAINT CLASSIFICATION WITH LSTM + BERT")
    print("=" * 70)
    print(f"Device: {CONFIG['device']}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Learning Rate: {CONFIG['learning_rate']}")
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialize pipeline
    pipeline = TrainingPipeline(CONFIG)
    
    # Load data
    train_df, val_df, test_df, sentiment_encoder, department_encoder = pipeline.load_data()
    
    # Initialize tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
    
    # Create data loaders
    train_loader, val_loader, test_loader = pipeline.create_dataloaders(
        train_df, val_df, test_df, tokenizer
    )
    
    # Initialize model
    model = pipeline.initialize_model()
    
    # Train model
    model = pipeline.train(model, train_loader, val_loader, tokenizer, 
                          department_encoder, sentiment_encoder)
    
    # Evaluate
    test_metrics = pipeline.evaluate(model, test_loader, sentiment_encoder, department_encoder)
    
    # Plot history
    pipeline.plot_training_history()
    
    # Save final model
    print(f"\n💾 Saving final model...")
    final_model_dir = os.path.join(MODELS_DIR, 'final_model')
    save_model(model, tokenizer, department_encoder, final_model_dir)
    
    # Save metrics
    metrics_path = os.path.join(MODELS_DIR, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"✅ Metrics saved to {metrics_path}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Models saved to: {MODELS_DIR}")
    print(f"📁 Data saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
