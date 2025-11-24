export interface ModelConfig {
  vocabSize: number;
  embeddingDim: number;
  lstmUnits: number;
  maxLength: number;
  dropout?: number;
}

export interface TrainingData {
  sequences: number[][];
  labels: number[];
}

export interface PredictionResult {
  label: 'REAL' | 'FAKE';
  confidence: number;
  probabilities: {
    real: number;
    fake: number;
  };
}

export class LSTMModel {
  private config: ModelConfig;
  private weights: {
    embedding: number[][];
    lstmWeights: any;
    denseWeights: number[][];
    denseBias: number[];
  } | null = null;

  constructor(config: ModelConfig) {
    this.config = config;
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  private tanh(x: number): number {
    return Math.tanh(x);
  }

  private initializeWeights() {
    const { vocabSize, embeddingDim, lstmUnits } = this.config;

    const embedding = Array(vocabSize).fill(0).map(() =>
      Array(embeddingDim).fill(0).map(() => (Math.random() - 0.5) * 0.1)
    );

    const denseWeights = Array(lstmUnits).fill(0).map(() =>
      Array(1).fill(0).map(() => (Math.random() - 0.5) * 0.1)
    );

    const denseBias = [0];

    this.weights = {
      embedding,
      lstmWeights: this.initializeLSTMWeights(embeddingDim, lstmUnits),
      denseWeights,
      denseBias
    };
  }

  private initializeLSTMWeights(inputDim: number, units: number) {
    return {
      inputGate: this.randomMatrix(inputDim + units, units),
      forgetGate: this.randomMatrix(inputDim + units, units),
      cellGate: this.randomMatrix(inputDim + units, units),
      outputGate: this.randomMatrix(inputDim + units, units),
      bias: Array(units * 4).fill(0)
    };
  }

  private randomMatrix(rows: number, cols: number): number[][] {
    return Array(rows).fill(0).map(() =>
      Array(cols).fill(0).map(() => (Math.random() - 0.5) * 0.1)
    );
  }

  private embed(sequence: number[]): number[][] {
    if (!this.weights) throw new Error('Model not initialized');

    return sequence.map(idx => {
      if (idx === 0 || idx >= this.weights!.embedding.length) {
        return Array(this.config.embeddingDim).fill(0);
      }
      return this.weights!.embedding[idx];
    });
  }

  private lstmCell(input: number[], prevHidden: number[], prevCell: number[]): [number[], number[]] {
    if (!this.weights) throw new Error('Model not initialized');

    const combined = [...input, ...prevHidden];
    const { lstmWeights } = this.weights;
    const units = this.config.lstmUnits;

    const forgetGate = this.matMul(combined, lstmWeights.forgetGate).map(v => this.sigmoid(v));
    const inputGate = this.matMul(combined, lstmWeights.inputGate).map(v => this.sigmoid(v));
    const cellGate = this.matMul(combined, lstmWeights.cellGate).map(v => this.tanh(v));
    const outputGate = this.matMul(combined, lstmWeights.outputGate).map(v => this.sigmoid(v));

    const newCell = prevCell.map((c, i) =>
      forgetGate[i] * c + inputGate[i] * cellGate[i]
    );

    const newHidden = newCell.map((c, i) =>
      outputGate[i] * this.tanh(c)
    );

    return [newHidden, newCell];
  }

  private matMul(vector: number[], matrix: number[][]): number[] {
    const result = Array(matrix[0].length).fill(0);

    for (let i = 0; i < matrix[0].length; i++) {
      for (let j = 0; j < vector.length; j++) {
        result[i] += vector[j] * (matrix[j]?.[i] || 0);
      }
    }

    return result;
  }

  private forward(sequence: number[]): number {
    if (!this.weights) throw new Error('Model not initialized');

    const embedded = this.embed(sequence);
    let hidden = Array(this.config.lstmUnits).fill(0);
    let cell = Array(this.config.lstmUnits).fill(0);

    for (const input of embedded) {
      [hidden, cell] = this.lstmCell(input, hidden, cell);
    }

    const output = this.matMul(hidden, this.weights.denseWeights);
    return this.sigmoid(output[0] + this.weights.denseBias[0]);
  }

  train(data: TrainingData, epochs: number = 10, learningRate: number = 0.001): void {
    this.initializeWeights();

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;

      for (let i = 0; i < data.sequences.length; i++) {
        const prediction = this.forward(data.sequences[i]);
        const loss = Math.pow(prediction - data.labels[i], 2);
        totalLoss += loss;
      }

      console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${(totalLoss / data.sequences.length).toFixed(4)}`);
    }
  }

  predict(sequence: number[]): PredictionResult {
    if (!this.weights) {
      throw new Error('Model not trained. Please train the model first.');
    }

    const probability = this.forward(sequence);
    const isFake = probability > 0.5;

    return {
      label: isFake ? 'FAKE' : 'REAL',
      confidence: isFake ? probability : 1 - probability,
      probabilities: {
        fake: probability,
        real: 1 - probability
      }
    };
  }

  save(): string {
    if (!this.weights) throw new Error('No weights to save');
    return JSON.stringify({
      config: this.config,
      weights: this.weights
    });
  }

  load(data: string): void {
    const parsed = JSON.parse(data);
    this.config = parsed.config;
    this.weights = parsed.weights;
  }
}
