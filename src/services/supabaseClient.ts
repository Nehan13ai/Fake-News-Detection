import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export interface PredictionRecord {
  id?: string;
  text: string;
  prediction: 'REAL' | 'FAKE';
  confidence: number;
  model_type: 'LSTM' | 'BiLSTM' | 'CNN';
  probabilities: {
    real: number;
    fake: number;
  };
  created_at?: string;
}

export async function savePrediction(record: PredictionRecord) {
  const { data, error } = await supabase
    .from('predictions')
    .insert([record])
    .select()
    .maybeSingle();

  if (error) {
    console.error('Error saving prediction:', error);
    throw error;
  }

  return data;
}

export async function getPredictionHistory(limit: number = 50) {
  const { data, error } = await supabase
    .from('predictions')
    .select('*')
    .order('created_at', { ascending: false })
    .limit(limit);

  if (error) {
    console.error('Error fetching predictions:', error);
    throw error;
  }

  return data;
}

export async function getPredictionStats() {
  const { data, error } = await supabase
    .from('predictions')
    .select('prediction, model_type');

  if (error) {
    console.error('Error fetching stats:', error);
    throw error;
  }

  const stats = {
    total: data.length,
    fake: data.filter(p => p.prediction === 'FAKE').length,
    real: data.filter(p => p.prediction === 'REAL').length,
    byModel: {
      LSTM: data.filter(p => p.model_type === 'LSTM').length,
      BiLSTM: data.filter(p => p.model_type === 'BiLSTM').length,
      CNN: data.filter(p => p.model_type === 'CNN').length
    }
  };

  return stats;
}
