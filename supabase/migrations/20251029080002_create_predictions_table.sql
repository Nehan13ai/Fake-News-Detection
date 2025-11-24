/*
  # Create predictions table for Fake News Detection System

  1. New Tables
    - `predictions`
      - `id` (uuid, primary key) - Unique identifier for each prediction
      - `text` (text) - The news text that was analyzed
      - `prediction` (text) - Classification result: 'REAL' or 'FAKE'
      - `confidence` (real) - Confidence score (0-1)
      - `model_type` (text) - Model used: 'LSTM', 'BiLSTM', or 'CNN'
      - `probabilities` (jsonb) - Detailed probability breakdown
      - `created_at` (timestamptz) - Timestamp of prediction

  2. Security
    - Enable RLS on `predictions` table
    - Add policy allowing anyone to insert predictions
    - Add policy allowing anyone to read predictions
    
  3. Important Notes
    - This table stores all prediction history for analytics
    - JSONB field stores both fake and real probabilities
    - Timestamps enable time-series analysis of predictions
*/

CREATE TABLE IF NOT EXISTS predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  text text NOT NULL,
  prediction text NOT NULL CHECK (prediction IN ('REAL', 'FAKE')),
  confidence real NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  model_type text NOT NULL CHECK (model_type IN ('LSTM', 'BiLSTM', 'CNN')),
  probabilities jsonb NOT NULL DEFAULT '{"real": 0, "fake": 0}'::jsonb,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can insert predictions"
  ON predictions
  FOR INSERT
  TO anon
  WITH CHECK (true);

CREATE POLICY "Anyone can read predictions"
  ON predictions
  FOR SELECT
  TO anon
  USING (true);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model_type ON predictions(model_type);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions(prediction);