import { ModelConfig, TrainingData, PredictionResult } from './LSTMModel';

export interface CNNConfig {
  vocabSize: number;
  embeddingDim: number;
  numFilters: number;
  filterSizes: number[];
  maxLength: number;
  dropout?: number;
}

export class CNNModel {
  private config: CNNConfig;
  private weights: {
    embedding: number[][];
    convFilters: Map<number, number[][][]>;
    denseWeights: number[][];
    denseBias: number[];
  } | null = null;

  constructor(config: CNNConfig) {
    this.config = config;
  }

  private relu(x: number): number {
    return Math.max(0, x);
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  private initializeWeights() {
    const { vocabSize, embeddingDim, numFilters, filterSizes } = this.config;

    const embedding = Array(vocabSize).fill(0).map(() =>
      Array(embeddingDim).fill(0).map(() => (Math.random() - 0.5) * 0.1)
    );

    const convFilters = new Map<number, number[][][]>();
    filterSizes.forEach(filterSize => {
      const filters = Array(numFilters).fill(0).map(() =>
        Array(filterSize).fill(0).map(() =>
          Array(embeddingDim).fill(0).map(() => (Math.random() - 0.5) * 0.1)
        )
      );
      convFilters.set(filterSize, filters);
    });

    const totalFeatures = numFilters * filterSizes.length;
    const denseWeights = Array(totalFeatures).fill(0).map(() =>
      Array(1).fill(0).map(() => (Math.random() - 0.5) * 0.1)
    );

    const denseBias = [0];

    this.weights = {
      embedding,
      convFilters,
      denseWeights,
      denseBias
    };
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

  private convolve1D(input: number[][], filter: number[][]): number[] {
    const result: number[] = [];
    const filterHeight = filter.length;

    for (let i = 0; i <= input.length - filterHeight; i++) {
      let sum = 0;
      for (let j = 0; j < filterHeight; j++) {
        for (let k = 0; k < input[0].length; k++) {
          sum += input[i + j][k] * filter[j][k];
        }
      }
      result.push(sum);
    }

    return result;
  }

  private maxPool(features: number[]): number {
    return Math.max(...features);
  }

  private forward(sequence: number[]): number {
    if (!this.weights) throw new Error('Model not initialized');

    const embedded = this.embed(sequence);
    const allFeatures: number[] = [];

    this.weights.convFilters.forEach((filters, filterSize) => {
      filters.forEach(filter => {
        const convOutput = this.convolve1D(embedded, filter);
        const activatedOutput = convOutput.map(v => this.relu(v));
        const pooledFeature = this.maxPool(activatedOutput);
        allFeatures.push(pooledFeature);
      });
    });

    let output = 0;
    for (let i = 0; i < allFeatures.length; i++) {
      output += allFeatures[i] * this.weights.denseWeights[i][0];
    }
    output += this.weights.denseBias[0];

    return this.sigmoid(output);
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

      console.log(`CNN Epoch ${epoch + 1}/${epochs}, Loss: ${(totalLoss / data.sequences.length).toFixed(4)}`);
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

    const serializedFilters: any = {};
    this.weights.convFilters.forEach((filters, size) => {
      serializedFilters[size] = filters;
    });

    return JSON.stringify({
      config: this.config,
      weights: {
        ...this.weights,
        convFilters: serializedFilters
      }
    });
  }

  load(data: string): void {
    const parsed = JSON.parse(data);
    this.config = parsed.config;

    const convFilters = new Map<number, number[][][]>();
    Object.entries(parsed.weights.convFilters).forEach(([size, filters]) => {
      convFilters.set(Number(size), filters as number[][][]);
    });

    this.weights = {
      ...parsed.weights,
      convFilters
    };
  }
}
