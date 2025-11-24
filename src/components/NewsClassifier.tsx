import { useState } from 'react';
import { LSTMModel } from '../models/LSTMModel';
import { BiLSTMModel } from '../models/BiLSTMModel';
import { CNNModel } from '../models/CNNModel';
import { TrainingService } from '../services/trainingService';
import { TextPreprocessor } from '../utils/textPreprocessing';
import { savePrediction } from '../services/supabaseClient';
import { FileText, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';

interface NewsClassifierProps {
  models: {
    lstm: LSTMModel;
    bilstm: BiLSTMModel;
    cnn: CNNModel;
    trainingService: TrainingService;
  };
}

export function NewsClassifier({ models }: NewsClassifierProps) {
  const [text, setText] = useState('');
  const [selectedModel, setSelectedModel] = useState<'lstm' | 'bilstm' | 'cnn'>('bilstm');
  const [result, setResult] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const analyzeNews = async () => {
    if (!text.trim()) return;

    setIsAnalyzing(true);
    await new Promise(resolve => setTimeout(resolve, 800));

    const preprocessed = TextPreprocessor.preprocess(text);
    const vocabulary = models.trainingService.getVocabulary();
    const maxLength = models.trainingService.getMaxLength();

    if (!vocabulary) {
      setIsAnalyzing(false);
      return;
    }

    const sequence = TextPreprocessor.textToSequence(preprocessed, vocabulary, maxLength);

    let prediction;
    switch (selectedModel) {
      case 'lstm':
        prediction = models.lstm.predict(sequence);
        break;
      case 'bilstm':
        prediction = models.bilstm.predict(sequence);
        break;
      case 'cnn':
        prediction = models.cnn.predict(sequence);
        break;
    }

    setResult(prediction);

    try {
      await savePrediction({
        text: text.substring(0, 500),
        prediction: prediction.label,
        confidence: prediction.confidence,
        model_type: selectedModel.toUpperCase() as 'LSTM' | 'BiLSTM' | 'CNN',
        probabilities: prediction.probabilities
      });
    } catch (error) {
      console.error('Failed to save prediction:', error);
    }

    setIsAnalyzing(false);
  };

  const exampleNews = [
    {
      label: "Real News Example",
      text: "Scientists at MIT have developed a new renewable energy technology that could increase solar panel efficiency by 40%. The breakthrough was published in Nature Energy journal."
    },
    {
      label: "Fake News Example",
      text: "BREAKING: Government admits to hiding aliens in secret underground base. Whistleblower reveals shocking truth that mainstream media won't report. Share before deleted!"
    }
  ];

  return (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <div className="flex items-center gap-3 mb-6">
        <FileText className="w-8 h-8 text-emerald-600" />
        <h2 className="text-2xl font-bold text-gray-800">News Classifier</h2>
      </div>

      <div className="mb-6">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Select Model
        </label>
        <div className="flex gap-4">
          {[
            { value: 'lstm', label: 'LSTM' },
            { value: 'bilstm', label: 'BiLSTM' },
            { value: 'cnn', label: 'CNN' }
          ].map(option => (
            <button
              key={option.value}
              onClick={() => setSelectedModel(option.value as any)}
              className={`px-6 py-2 rounded-lg font-semibold transition-colors ${
                selectedModel === option.value
                  ? 'bg-emerald-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mb-6">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Enter News Text or Headline
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a news headline or article here..."
          className="w-full h-40 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-none"
        />
      </div>

      <div className="mb-6">
        <p className="text-sm font-semibold text-gray-700 mb-3">Try an example:</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {exampleNews.map((example, idx) => (
            <button
              key={idx}
              onClick={() => setText(example.text)}
              className="text-left p-4 bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors"
            >
              <p className="text-xs font-semibold text-emerald-600 mb-1">{example.label}</p>
              <p className="text-sm text-gray-700 line-clamp-2">{example.text}</p>
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={analyzeNews}
        disabled={!text.trim() || isAnalyzing}
        className="w-full bg-emerald-600 text-white py-3 rounded-lg font-semibold hover:bg-emerald-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
      >
        {isAnalyzing ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Analyzing...
          </>
        ) : (
          'Analyze News'
        )}
      </button>

      {result && (
        <div className={`mt-6 p-6 rounded-xl border-2 ${
          result.label === 'FAKE'
            ? 'bg-red-50 border-red-300'
            : 'bg-green-50 border-green-300'
        }`}>
          <div className="flex items-center gap-3 mb-4">
            {result.label === 'FAKE' ? (
              <AlertCircle className="w-8 h-8 text-red-600" />
            ) : (
              <CheckCircle2 className="w-8 h-8 text-green-600" />
            )}
            <div>
              <h3 className={`text-2xl font-bold ${
                result.label === 'FAKE' ? 'text-red-700' : 'text-green-700'
              }`}>
                {result.label} NEWS
              </h3>
              <p className="text-sm text-gray-600">
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 space-y-2">
            <h4 className="font-semibold text-gray-700 mb-3">Probability Distribution</h4>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="font-medium text-red-600">Fake</span>
                  <span className="font-semibold text-gray-700">
                    {(result.probabilities.fake * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-red-500 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${result.probabilities.fake * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="font-medium text-green-600">Real</span>
                  <span className="font-semibold text-gray-700">
                    {(result.probabilities.real * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-green-500 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${result.probabilities.real * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 text-sm text-gray-600 bg-white rounded-lg p-3">
            <p className="font-semibold mb-1">Model Used: {selectedModel.toUpperCase()}</p>
            <p className="text-xs">
              {result.label === 'FAKE'
                ? 'This content shows patterns commonly found in misleading or fabricated news.'
                : 'This content appears to follow patterns of legitimate news reporting.'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
