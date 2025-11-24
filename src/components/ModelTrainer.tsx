import { useState } from 'react';
import { LSTMModel } from '../models/LSTMModel';
import { BiLSTMModel } from '../models/BiLSTMModel';
import { CNNModel } from '../models/CNNModel';
import { TrainingService, sampleDataset } from '../services/trainingService';
import { Brain, TrendingUp, CheckCircle } from 'lucide-react';

interface ModelTrainerProps {
  onModelsReady: (models: {
    lstm: LSTMModel;
    bilstm: BiLSTMModel;
    cnn: CNNModel;
    trainingService: TrainingService;
  }) => void;
}

export function ModelTrainer({ onModelsReady }: ModelTrainerProps) {
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState('');
  const [metrics, setMetrics] = useState<any>(null);
  const [trained, setTrained] = useState(false);

  const trainModels = async () => {
    setIsTraining(true);
    setProgress('Preparing dataset...');

    await new Promise(resolve => setTimeout(resolve, 500));

    const trainingService = new TrainingService();
    const { sequences, labels } = trainingService.prepareDataset(sampleDataset);
    const { train, test } = trainingService.splitDataset(sequences, labels, 0.2);

    const vocabSize = trainingService.getVocabulary()?.size || 1000;
    const maxLength = trainingService.getMaxLength();

    setProgress('Training LSTM model...');
    await new Promise(resolve => setTimeout(resolve, 500));

    const lstm = new LSTMModel({
      vocabSize: vocabSize + 1,
      embeddingDim: 50,
      lstmUnits: 64,
      maxLength: maxLength
    });
    lstm.train(train, 5);

    const lstmMetrics = trainingService.evaluateModel(lstm, test.sequences, test.labels);

    setProgress('Training BiLSTM model...');
    await new Promise(resolve => setTimeout(resolve, 500));

    const bilstm = new BiLSTMModel({
      vocabSize: vocabSize + 1,
      embeddingDim: 50,
      lstmUnits: 64,
      maxLength: maxLength
    });
    bilstm.train(train, 5);

    const bilstmMetrics = trainingService.evaluateModel(bilstm, test.sequences, test.labels);

    setProgress('Training CNN model...');
    await new Promise(resolve => setTimeout(resolve, 500));

    const cnn = new CNNModel({
      vocabSize: vocabSize + 1,
      embeddingDim: 50,
      numFilters: 100,
      filterSizes: [3, 4, 5],
      maxLength: maxLength
    });
    cnn.train(train, 5);

    const cnnMetrics = trainingService.evaluateModel(cnn, test.sequences, test.labels);

    setMetrics({
      lstm: lstmMetrics,
      bilstm: bilstmMetrics,
      cnn: cnnMetrics
    });

    setProgress('Training complete!');
    setTrained(true);
    setIsTraining(false);

    onModelsReady({ lstm, bilstm, cnn, trainingService });
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
      <div className="flex items-center gap-3 mb-6">
        <Brain className="w-8 h-8 text-blue-600" />
        <h2 className="text-2xl font-bold text-gray-800">Model Training</h2>
      </div>

      {!trained && (
        <div className="text-center">
          <p className="text-gray-600 mb-6">
            Train the deep learning models on sample fake news data to enable classification.
          </p>
          <button
            onClick={trainModels}
            disabled={isTraining}
            className="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {isTraining ? 'Training...' : 'Train Models'}
          </button>
          {isTraining && (
            <p className="mt-4 text-blue-600 font-medium">{progress}</p>
          )}
        </div>
      )}

      {trained && metrics && (
        <div className="space-y-6">
          <div className="flex items-center justify-center gap-2 text-green-600 font-semibold">
            <CheckCircle className="w-6 h-6" />
            <span>Models trained successfully!</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {['lstm', 'bilstm', 'cnn'].map(modelType => (
              <div key={modelType} className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-6 border border-blue-200">
                <div className="flex items-center gap-2 mb-4">
                  <TrendingUp className="w-5 h-5 text-blue-600" />
                  <h3 className="font-bold text-gray-800 uppercase">{modelType}</h3>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Accuracy:</span>
                    <span className="font-semibold text-gray-800">
                      {(metrics[modelType].accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Precision:</span>
                    <span className="font-semibold text-gray-800">
                      {(metrics[modelType].precision * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Recall:</span>
                    <span className="font-semibold text-gray-800">
                      {(metrics[modelType].recall * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">F1-Score:</span>
                    <span className="font-semibold text-gray-800">
                      {(metrics[modelType].f1Score * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
