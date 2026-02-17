#!/usr/bin/env python3
"""
QUICK START GUIDE - LSTM Complaint Classification System

This script provides a complete workflow to:
1. Prepare data from CSV files
2. Train the model
3. Run inference
4. Save to database
"""

import os
import sys
from pathlib import Path

# Configuration
WORKSPACE_ROOT = r"d:\Majorproject\resolvex-citizen-connect-main"
ML_DIR = os.path.join(WORKSPACE_ROOT, "src", "ml")
DATA_DIR = os.path.join(ML_DIR, "data")
MODELS_DIR = os.path.join(ML_DIR, "models")


def print_banner(title):
    """Print formatted banner"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def check_dependencies():
    """Check if all required packages are installed"""
    print_banner("1️⃣  CHECKING DEPENDENCIES")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Hugging Face Transformers'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('mysql', 'MySQL Connector'),
        ('flask', 'Flask'),
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"\n📦 Install with:")
        print(f"   cd {ML_DIR}")
        print(f"   pip install -r requirements.txt")
        return False
    
    print(f"\n✅ All dependencies installed!\n")
    return True


def prepare_data():
    """Prepare datasets"""
    print_banner("2️⃣  PREPARING DATA")
    
    print("📂 Loading datasets from all departments...")
    print("🧹 Cleaning text and preprocessing...")
    print("📊 Creating train/val/test splits...")
    
    print(f"\nRun this command:")
    print(f"  cd {ML_DIR}")
    print(f"  python data_preprocessing.py")
    
    print(f"\n⏱️  Expected time: 5-10 minutes")
    print(f"💾 Output: {DATA_DIR}/")


def train_model():
    """Train the model"""
    print_banner("3️⃣  TRAINING MODEL")
    
    print("🧠 Initializing LSTM + BERT architecture...")
    print("📥 Loading preprocessed datasets...")
    print("🚀 Starting training loop...")
    print("💾 Saving best model checkpoints...")
    
    print(f"\nRun this command:")
    print(f"  cd {ML_DIR}")
    print(f"  python train.py")
    
    print(f"\n⏱️  Expected time: 30-60 minutes (GPU: 5-10 min)")
    print(f"💾 Output: {MODELS_DIR}/best_model/")


def run_inference():
    """Run inference on test data"""
    print_banner("4️⃣  RUNNING INFERENCE")
    
    print("📝 Testing classification on example complaints...")
    print("💾 Saving predictions to database...")
    
    print(f"\nRun this command:")
    print(f"  cd {ML_DIR}")
    print(f"  python inference.py")
    
    print(f"\n✅ Outputs: Predictions with confidence scores")


def start_api():
    """Start Flask API server"""
    print_banner("5️⃣  STARTING API SERVER")
    
    print("🚀 Starting REST API on http://localhost:5000")
    print("\n📡 Available endpoints:")
    print("  - POST /api/v1/classify")
    print("  - POST /api/v1/classify-batch")
    print("  - GET  /api/v1/departments")
    print("  - GET  /api/v1/languages")
    print("  - GET  /api/v1/docs")
    
    print(f"\nRun this command:")
    print(f"  cd {ML_DIR}")
    print(f"  python api.py")
    
    print(f"\n🌐 Web UI: http://localhost:5000/api/v1/docs")


def test_predictions():
    """Test with sample complaints"""
    print_banner("6️⃣  TESTING PREDICTIONS")
    
    print("📝 Sample test cases:\n")
    
    examples = [
        ("English", "Power outage in my area since 12 hours", "Electricity"),
        ("Hindi", "बीते 2 दिनों से बिजली नहीं है", "Electricity"),
        ("Tamil", "வெள்ளம் காரணமாக சாலை சேதமடைந்தது", "Public Works"),
        ("English", "Hospital staff is very rude", "Health Services"),
        ("English", "School has no clean toilets", "Education Services"),
    ]
    
    for lang, text, expected_dept in examples:
        print(f"  [{lang}] {text}")
        print(f"         → Expected: {expected_dept}")
        print()


def show_curl_examples():
    """Show CURL examples for testing API"""
    print_banner("7️⃣  TESTING API WITH CURL")
    
    print("📝 Single Classification:\n")
    print("""curl -X POST http://localhost:5000/api/v1/classify \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Power outage in my area",
    "user_id": 1,
    "save_to_db": true
  }'
""")
    
    print("\n📝 Batch Classification:\n")
    print("""curl -X POST http://localhost:5000/api/v1/classify-batch \\
  -H "Content-Type: application/json" \\
  -d '{
    "complaints": [
      {"text": "Road is damaged", "user_id": 1},
      {"text": "No water supply", "user_id": 2}
    ],
    "save_to_db": true
  }'
""")


def show_python_examples():
    """Show Python code examples"""
    print_banner("8️⃣  PYTHON USAGE EXAMPLES")
    
    print("""from inference import ComplaintAPIWrapper

# Initialize
api = ComplaintAPIWrapper()

# Example 1: Single complaint
result = api.predict(
    complaint_text="Power outage since 12 hours",
    user_id=1,
    save_to_db=True
)

print(f"Sentiment: {result['sentiment']['label']}")
print(f"Department: {result['department']['name']}")
print(f"Priority: {result['urgency']['priority']}")

# Example 2: Batch processing
complaints = [
    {"text": "Road damaged", "user_id": 1},
    {"text": "No water", "user_id": 2}
]

results = api.predict_batch(complaints)
print(f"Processed {results['total']} complaints")
print(f"Successful: {results['successful']}")
""")


def database_setup():
    """Show database setup instructions"""
    print_banner("9️⃣  DATABASE SETUP")
    
    print("🗄️  Initialize MySQL Database:\n")
    print(f"mysql -u root -p < {WORKSPACE_ROOT}/documents/Database/resolveX_database_setup.sql")
    
    print("\n📝 Update credentials in src/ml/inference.py:")
    print("""
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_password',
    'database': 'resolveX_grievance_system'
}
""")


def print_directory_structure():
    """Print the complete directory structure"""
    print_banner("PROJECT STRUCTURE")
    
    structure = """
src/ml/
├── requirements.txt              ← Dependencies
├── README.md                     ← Full documentation
│
├── lstm_model.py                 ← LSTM architecture
├── data_preprocessing.py         ← Data loading & cleaning
├── train.py                      ← Training pipeline
├── inference.py                  ← Predictions
├── api.py                        ← Flask REST API
│
├── models/
│   ├── best_model/              ← Best trained model
│   │   ├── model.pth
│   │   ├── department_encoder.pkl
│   │   └── config.json
│   └── training_history.png
│
└── data/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── sentiment_encoder.pkl
    └── department_encoder.pkl
"""
    print(structure)


def show_complete_workflow():
    """Show the complete workflow"""
    print("\n" + "=" * 80)
    print("  🚀 COMPLETE WORKFLOW")
    print("=" * 80 + "\n")
    
    print("STEP 1: Install Dependencies")
    print("─" * 80)
    print("cd src/ml")
    print("pip install -r requirements.txt")
    print()
    
    print("STEP 2: Prepare Datasets")
    print("─" * 80)
    print("python data_preprocessing.py")
    print("  ✅ Loads 80,000+ complaints from 8 departments")
    print("  ✅ Handles 3 languages (English, Hindi, Tamil)")
    print("  ✅ Creates train/val/test splits")
    print()
    
    print("STEP 3: Train Model")
    print("─" * 80)
    print("python train.py")
    print("  ✅ Trains BERT + Bi-LSTM model")
    print("  ✅ Performs multi-task learning")
    print("  ✅ Saves best model checkpoint")
    print()
    
    print("STEP 4: Test Predictions")
    print("─" * 80)
    print("python inference.py")
    print("  ✅ Tests on example complaints")
    print("  ✅ Shows sentiment, department, urgency")
    print("  ✅ Saves to database")
    print()
    
    print("STEP 5: Start API Server")
    print("─" * 80)
    print("python api.py")
    print("  ✅ REST API on http://localhost:5000")
    print("  ✅ Real-time predictions")
    print("  ✅ Batch processing support")
    print()
    
    print("STEP 6: Integrate with Frontend")
    print("─" * 80)
    print("POST /api/v1/classify")
    print("  ✅ React/Vue integration ready")
    print("  ✅ WebSocket support (optional)")
    print()


def main():
    """Main menu"""
    print("\n" * 2)
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  🎯 LSTM COMPLAINT CLASSIFICATION - QUICK START".center(78) + "║")
    print("║" + "  Multilingual + Sentiment + Department + Urgency Detection".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    
    print("\n📚 DOCUMENTATION & GUIDES:\n")
    print("  1. Check Dependencies")
    print("  2. Data Preparation Guide")
    print("  3. Model Training Guide")
    print("  4. Inference Guide")
    print("  5. API Server Guide")
    print("  6. Test Predictions")
    print("  7. API CURL Examples")
    print("  8. Python Code Examples")
    print("  9. Database Setup")
    print("  10. Project Structure")
    print("  11. Complete Workflow")
    print("  0. Exit")
    
    while True:
        choice = input("\n👉 Select an option (0-11): ").strip()
        
        if choice == "0":
            print("\n✅ Exiting. Happy coding! 🚀\n")
            break
        elif choice == "1":
            check_dependencies()
        elif choice == "2":
            prepare_data()
        elif choice == "3":
            train_model()
        elif choice == "4":
            run_inference()
        elif choice == "5":
            start_api()
        elif choice == "6":
            test_predictions()
        elif choice == "7":
            show_curl_examples()
        elif choice == "8":
            show_python_examples()
        elif choice == "9":
            database_setup()
        elif choice == "10":
            print_directory_structure()
        elif choice == "11":
            show_complete_workflow()
        else:
            print("❌ Invalid option. Please try again.")
        
        input("\n\n👈 Press Enter to continue...")


if __name__ == "__main__":
    # Auto-run workflow if environment variable set
    if os.getenv("AUTO_WORKFLOW") == "1":
        show_complete_workflow()
    else:
        main()
