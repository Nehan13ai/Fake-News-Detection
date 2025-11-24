# Fake News Detection System - Project Overview

## System Architecture

This is a complete fake news detection system built using deep learning models implemented in TypeScript/JavaScript. The system classifies news articles and headlines as either REAL or FAKE with confidence scores.

## Features

### 1. Multiple Deep Learning Models
- **LSTM (Long Short-Term Memory)**: Sequential model that processes text forward
- **BiLSTM (Bidirectional LSTM)**: Processes text in both directions for better context understanding
- **CNN (Convolutional Neural Network)**: Uses filters to extract features from text for classification

### 2. Text Preprocessing Pipeline
- Text cleaning (lowercase, remove punctuation, numbers, special characters)
- Stop word removal
- Tokenization
- Sequence padding and vectorization
- TF-IDF calculation support

### 3. Model Training & Evaluation
- Training on sample fake/real news dataset (12 articles included)
- Performance metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix visualization
- Train/test split (80/20)

### 4. Interactive Web Interface
- Model selection (LSTM/BiLSTM/CNN)
- Real-time news classification
- Confidence scores with percentage
- Probability distribution visualization
- Example news articles for testing

### 5. Analytics Dashboard
- Prediction history tracking
- Real-time statistics
- Model usage analytics
- Fake vs Real news distribution

### 6. Data Persistence
- Supabase integration for storing predictions
- Historical data analysis
- Row Level Security enabled

## Project Structure

```
src/
├── models/
│   ├── LSTMModel.ts       # LSTM implementation
│   ├── BiLSTMModel.ts     # Bidirectional LSTM
│   └── CNNModel.ts        # CNN for text classification
├── utils/
│   └── textPreprocessing.ts  # Text cleaning and tokenization
├── services/
│   ├── trainingService.ts    # Training pipeline and evaluation
│   └── supabaseClient.ts     # Database operations
├── components/
│   ├── ModelTrainer.tsx      # Model training UI
│   ├── NewsClassifier.tsx    # Classification interface
│   └── PredictionHistory.tsx # Analytics dashboard
└── App.tsx                    # Main application
```

## Technologies Used

- **Frontend**: React, TypeScript, Tailwind CSS
- **Icons**: Lucide React
- **Database**: Supabase (PostgreSQL)
- **Build Tool**: Vite
- **Deep Learning**: Custom implementations inspired by TensorFlow/PyTorch

## How It Works

### 1. Training Phase
- Loads sample dataset of real and fake news
- Preprocesses text (cleaning, tokenization)
- Creates vocabulary mapping
- Trains three models (LSTM, BiLSTM, CNN)
- Evaluates each model on test set
- Displays performance metrics

### 2. Prediction Phase
- User inputs news text or headline
- Text is preprocessed identically to training
- Converted to numerical sequence using vocabulary
- Model predicts probability of being fake
- Results displayed with confidence score
- Prediction saved to database

### 3. Model Architectures

**LSTM**
- Embedding layer (vocab_size × embedding_dim)
- LSTM layer with 64 units
- Dense output layer with sigmoid activation

**BiLSTM**
- Embedding layer
- Forward LSTM (64 units) + Backward LSTM (64 units)
- Concatenated outputs → Dense layer

**CNN**
- Embedding layer
- Multiple 1D convolution filters (sizes 3, 4, 5)
- Max pooling over time
- Dense output layer

## Database Schema

### predictions table
- `id`: UUID primary key
- `text`: News text analyzed
- `prediction`: 'REAL' or 'FAKE'
- `confidence`: Score between 0 and 1
- `model_type`: Which model was used
- `probabilities`: JSON with detailed breakdown
- `created_at`: Timestamp

## Sample Dataset

The system includes 12 sample news articles:
- 6 real news articles (scientific discoveries, policy announcements, etc.)
- 6 fake news articles (conspiracy theories, clickbait, misinformation)

## Evaluation Metrics

Each model is evaluated using:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: TP, TN, FP, FN counts

## Usage Instructions

1. **Train Models**: Click "Train Models" button to initialize and train all three models
2. **Select Model**: Choose between LSTM, BiLSTM, or CNN
3. **Input Text**: Paste news article or headline
4. **Analyze**: Click "Analyze News" to get prediction
5. **View Results**: See classification, confidence score, and probability distribution
6. **Check History**: View past predictions and analytics

## Key Characteristics of Fake vs Real News

The models learn to detect patterns such as:

**Fake News Indicators**:
- Sensational language and clickbait phrases
- Lack of credible sources or attribution
- Emotional manipulation tactics
- Conspiracy theories and unverified claims
- Poor grammar and writing quality

**Real News Indicators**:
- Factual, objective language
- Attribution to credible sources
- Specific details and data
- Professional writing style
- Verifiable information

## Performance Notes

- Models use simplified architectures suitable for browser execution
- Training is performed client-side with sample data
- For production use, models should be trained on larger datasets
- Current implementation is educational and demonstrates the concepts
- Real-world systems would use pre-trained models or server-side inference

## Future Enhancements

- Integration with larger datasets (Kaggle Fake News Dataset)
- BERT-based transformer models
- Model persistence (save/load trained weights)
- Batch prediction capability
- Export functionality for results
- Advanced visualizations (word clouds, attention maps)
- API integration for fact-checking services
