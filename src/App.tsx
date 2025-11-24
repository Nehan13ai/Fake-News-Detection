import { useState } from 'react';
import { ModelTrainer } from './components/ModelTrainer';
import { NewsClassifier } from './components/NewsClassifier';
import { PredictionHistory } from './components/PredictionHistory';
import { Shield, Sparkles } from 'lucide-react';
import { LSTMModel } from './models/LSTMModel';
import { BiLSTMModel } from './models/BiLSTMModel';
import { CNNModel } from './models/CNNModel';
import { TrainingService } from './services/trainingService';

function App() {
  const [models, setModels] = useState<{
    lstm: LSTMModel;
    bilstm: BiLSTMModel;
    cnn: CNNModel;
    trainingService: TrainingService;
  } | null>(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-emerald-50">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <header className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Shield className="w-12 h-12 text-emerald-600" />
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 via-emerald-600 to-teal-600 bg-clip-text text-transparent">
              Fake News Detection System
            </h1>
            <Sparkles className="w-12 h-12 text-blue-600" />
          </div>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Advanced deep learning system using LSTM, BiLSTM, and CNN models to classify news articles as real or fake
          </p>
        </header>

        <div className="space-y-8">
          <ModelTrainer onModelsReady={setModels} />

          {models && (
            <>
              <NewsClassifier models={models} />
              <PredictionHistory />
            </>
          )}
        </div>

        <footer className="mt-16 text-center text-sm text-gray-500">
          <p>Built with TensorFlow-inspired architecture, React, TypeScript, and Supabase</p>
          <p className="mt-2">Models: LSTM, BiLSTM, CNN | Preprocessing: TF-IDF, Text Cleaning, Tokenization</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
