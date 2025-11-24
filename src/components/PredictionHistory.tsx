import { useState, useEffect } from 'react';
import { getPredictionHistory, getPredictionStats } from '../services/supabaseClient';
import { History, TrendingUp, Shield, AlertTriangle } from 'lucide-react';

export function PredictionHistory() {
  const [history, setHistory] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [historyData, statsData] = await Promise.all([
        getPredictionHistory(10),
        getPredictionStats()
      ]);
      setHistory(historyData || []);
      setStats(statsData);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-8">
        <p className="text-center text-gray-500">Loading analytics...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {stats && stats.total > 0 && (
        <div className="bg-white rounded-xl shadow-lg p-8">
          <div className="flex items-center gap-3 mb-6">
            <TrendingUp className="w-8 h-8 text-orange-600" />
            <h2 className="text-2xl font-bold text-gray-800">Analytics</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6 border border-blue-200">
              <p className="text-sm text-blue-600 font-semibold mb-1">Total Predictions</p>
              <p className="text-3xl font-bold text-gray-800">{stats.total}</p>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-6 border border-green-200">
              <div className="flex items-center gap-2 mb-1">
                <Shield className="w-4 h-4 text-green-600" />
                <p className="text-sm text-green-600 font-semibold">Real News</p>
              </div>
              <p className="text-3xl font-bold text-gray-800">{stats.real}</p>
              <p className="text-xs text-gray-500 mt-1">
                {((stats.real / stats.total) * 100).toFixed(1)}%
              </p>
            </div>

            <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-lg p-6 border border-red-200">
              <div className="flex items-center gap-2 mb-1">
                <AlertTriangle className="w-4 h-4 text-red-600" />
                <p className="text-sm text-red-600 font-semibold">Fake News</p>
              </div>
              <p className="text-3xl font-bold text-gray-800">{stats.fake}</p>
              <p className="text-xs text-gray-500 mt-1">
                {((stats.fake / stats.total) * 100).toFixed(1)}%
              </p>
            </div>

            <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-6 border border-purple-200">
              <p className="text-sm text-purple-600 font-semibold mb-2">Models Used</p>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">LSTM:</span>
                  <span className="font-semibold text-gray-800">{stats.byModel.LSTM}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">BiLSTM:</span>
                  <span className="font-semibold text-gray-800">{stats.byModel.BiLSTM}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">CNN:</span>
                  <span className="font-semibold text-gray-800">{stats.byModel.CNN}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {history.length > 0 && (
        <div className="bg-white rounded-xl shadow-lg p-8">
          <div className="flex items-center gap-3 mb-6">
            <History className="w-8 h-8 text-gray-700" />
            <h2 className="text-2xl font-bold text-gray-800">Recent Predictions</h2>
          </div>

          <div className="space-y-3">
            {history.map((item) => (
              <div
                key={item.id}
                className={`p-4 rounded-lg border-l-4 ${
                  item.prediction === 'FAKE'
                    ? 'bg-red-50 border-red-500'
                    : 'bg-green-50 border-green-500'
                }`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <p className="text-sm text-gray-700 line-clamp-2 mb-2">{item.text}</p>
                    <div className="flex items-center gap-4 text-xs text-gray-500">
                      <span className="font-semibold">
                        Model: {item.model_type}
                      </span>
                      <span>
                        {new Date(item.created_at).toLocaleDateString()} {' '}
                        {new Date(item.created_at).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className={`text-lg font-bold ${
                      item.prediction === 'FAKE' ? 'text-red-700' : 'text-green-700'
                    }`}>
                      {item.prediction}
                    </p>
                    <p className="text-xs text-gray-500">
                      {(item.confidence * 100).toFixed(1)}%
                    </p>
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
