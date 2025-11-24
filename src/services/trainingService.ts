import { LSTMModel } from '../models/LSTMModel';
import { BiLSTMModel } from '../models/BiLSTMModel';
import { CNNModel } from '../models/CNNModel';
import { TextPreprocessor } from '../utils/textPreprocessing';

export interface NewsArticle {
  title: string;
  text: string;
  label: 'real' | 'fake';
}

export interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: {
    truePositive: number;
    trueNegative: number;
    falsePositive: number;
    falseNegative: number;
  };
}

export class TrainingService {
  private vocabulary: Map<string, number> | null = null;
  private maxLength = 100;

  prepareDataset(articles: NewsArticle[]) {
    const processedTexts = articles.map(article => {
      const combinedText = `${article.title} ${article.text}`;
      return TextPreprocessor.preprocess(combinedText);
    });

    this.vocabulary = TextPreprocessor.createVocabulary(processedTexts);

    const sequences = processedTexts.map(text =>
      TextPreprocessor.textToSequence(text, this.vocabulary!, this.maxLength)
    );

    const labels = articles.map(article => article.label === 'fake' ? 1 : 0);

    return { sequences, labels };
  }

  splitDataset(sequences: number[][], labels: number[], testSize: number = 0.2) {
    const totalSize = sequences.length;
    const testLength = Math.floor(totalSize * testSize);
    const trainLength = totalSize - testLength;

    const indices = Array.from({ length: totalSize }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const trainIndices = indices.slice(0, trainLength);
    const testIndices = indices.slice(trainLength);

    return {
      train: {
        sequences: trainIndices.map(i => sequences[i]),
        labels: trainIndices.map(i => labels[i])
      },
      test: {
        sequences: testIndices.map(i => sequences[i]),
        labels: testIndices.map(i => labels[i])
      }
    };
  }

  evaluateModel(
    model: LSTMModel | BiLSTMModel | CNNModel,
    testSequences: number[][],
    testLabels: number[]
  ): EvaluationMetrics {
    let truePositive = 0;
    let trueNegative = 0;
    let falsePositive = 0;
    let falseNegative = 0;

    for (let i = 0; i < testSequences.length; i++) {
      const prediction = model.predict(testSequences[i]);
      const predictedLabel = prediction.label === 'FAKE' ? 1 : 0;
      const actualLabel = testLabels[i];

      if (predictedLabel === 1 && actualLabel === 1) truePositive++;
      else if (predictedLabel === 0 && actualLabel === 0) trueNegative++;
      else if (predictedLabel === 1 && actualLabel === 0) falsePositive++;
      else if (predictedLabel === 0 && actualLabel === 1) falseNegative++;
    }

    const accuracy = (truePositive + trueNegative) / testSequences.length;
    const precision = truePositive / (truePositive + falsePositive) || 0;
    const recall = truePositive / (truePositive + falseNegative) || 0;
    const f1Score = 2 * (precision * recall) / (precision + recall) || 0;

    return {
      accuracy,
      precision,
      recall,
      f1Score,
      confusionMatrix: {
        truePositive,
        trueNegative,
        falsePositive,
        falseNegative
      }
    };
  }

  getVocabulary(): Map<string, number> | null {
    return this.vocabulary;
  }

  getMaxLength(): number {
    return this.maxLength;
  }
}

export const sampleDataset: NewsArticle[] = [
  {
    title: "Scientists Discover New Planet in Solar System",
    text: "Astronomers have discovered a new planet beyond Neptune. The planet, temporarily named Planet X, is approximately twice the size of Earth and orbits the sun once every 10,000 years.",
    label: "real"
  },
  {
    title: "BREAKING: Aliens Land on White House Lawn",
    text: "Extraterrestrial beings arrived in Washington D.C. today demanding to speak with world leaders. The government is covering up this historic event. Share this before it gets deleted!",
    label: "fake"
  },
  {
    title: "New Study Shows Benefits of Regular Exercise",
    text: "A comprehensive study published in the Journal of Medicine reveals that regular physical activity significantly reduces the risk of cardiovascular disease and improves mental health outcomes.",
    label: "real"
  },
  {
    title: "Doctors Hate This One Weird Trick to Lose Weight",
    text: "This miracle berry from the Amazon rainforest will make you lose 50 pounds in one week without any diet or exercise. Big Pharma doesn't want you to know about this secret.",
    label: "fake"
  },
  {
    title: "Global Climate Summit Reaches Historic Agreement",
    text: "World leaders have reached a landmark agreement on climate action at the international summit. The treaty includes commitments to reduce carbon emissions by 50% by 2030.",
    label: "real"
  },
  {
    title: "Shocking Truth About Vaccines That Will Change Everything",
    text: "Secret government documents reveal that vaccines contain mind control chips. The mainstream media refuses to report this but thousands of doctors have come forward to expose the truth.",
    label: "fake"
  },
  {
    title: "Tech Company Announces Revolutionary Battery Technology",
    text: "A leading technology firm has developed a new lithium-air battery that could triple the range of electric vehicles. The innovation is expected to be commercially available within three years.",
    label: "real"
  },
  {
    title: "Celebrity Dies and Comes Back to Life with Important Message",
    text: "Famous actor was clinically dead for 20 minutes and returned with a warning about the end of the world. Doctors are baffled and can't explain what happened. Click to see the shocking video.",
    label: "fake"
  },
  {
    title: "Economic Report Shows Steady Growth in Manufacturing Sector",
    text: "The latest economic indicators show that the manufacturing sector has experienced consistent growth over the past quarter. Employment in the sector has increased by 2.3% according to government statistics.",
    label: "real"
  },
  {
    title: "Pope Declares Support for Radical Political Movement",
    text: "In a shocking announcement, the Pope has endorsed a controversial political ideology. The Vatican denies these claims but leaked documents prove otherwise. This is what they don't want you to know.",
    label: "fake"
  },
  {
    title: "Archaeological Team Uncovers Ancient Ruins in Peru",
    text: "Researchers have discovered well-preserved ruins of a pre-Incan civilization in the mountains of Peru. The site includes temples and residential structures dating back over 3,000 years.",
    label: "real"
  },
  {
    title: "5G Towers Confirmed to Control Weather and Cause Earthquakes",
    text: "Whistleblower reveals that 5G technology is being used by governments to manipulate weather patterns and trigger natural disasters. Scientists are silenced when they try to speak out.",
    label: "fake"
  }
];
