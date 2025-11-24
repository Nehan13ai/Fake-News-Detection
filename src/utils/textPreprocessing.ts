export class TextPreprocessor {
  private static stopWords = new Set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
    'off', 'over', 'under', 'again', 'further', 'then', 'once'
  ]);

  static cleanText(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .replace(/\d+/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  static removeStopWords(text: string): string {
    const words = text.split(' ');
    return words
      .filter(word => !this.stopWords.has(word) && word.length > 2)
      .join(' ');
  }

  static tokenize(text: string): string[] {
    return text.split(' ').filter(word => word.length > 0);
  }

  static preprocess(text: string): string {
    const cleaned = this.cleanText(text);
    return this.removeStopWords(cleaned);
  }

  static createVocabulary(texts: string[]): Map<string, number> {
    const vocabulary = new Map<string, number>();
    let index = 1;

    texts.forEach(text => {
      const tokens = this.tokenize(text);
      tokens.forEach(token => {
        if (!vocabulary.has(token)) {
          vocabulary.set(token, index++);
        }
      });
    });

    return vocabulary;
  }

  static textToSequence(text: string, vocabulary: Map<string, number>, maxLen: number = 100): number[] {
    const tokens = this.tokenize(text);
    const sequence = tokens
      .map(token => vocabulary.get(token) || 0)
      .slice(0, maxLen);

    while (sequence.length < maxLen) {
      sequence.push(0);
    }

    return sequence;
  }

  static calculateTfIdf(texts: string[]): { vectors: number[][], vocabulary: string[] } {
    const processedTexts = texts.map(text => this.preprocess(text));
    const allWords = new Set<string>();

    processedTexts.forEach(text => {
      const tokens = this.tokenize(text);
      tokens.forEach(token => allWords.add(token));
    });

    const vocabulary = Array.from(allWords);
    const documentFrequency = new Map<string, number>();

    vocabulary.forEach(word => {
      let count = 0;
      processedTexts.forEach(text => {
        if (text.includes(word)) count++;
      });
      documentFrequency.set(word, count);
    });

    const vectors = processedTexts.map(text => {
      const tokens = this.tokenize(text);
      const termFrequency = new Map<string, number>();

      tokens.forEach(token => {
        termFrequency.set(token, (termFrequency.get(token) || 0) + 1);
      });

      const maxFreq = Math.max(...Array.from(termFrequency.values()));

      return vocabulary.map(word => {
        const tf = (termFrequency.get(word) || 0) / maxFreq;
        const idf = Math.log(processedTexts.length / (documentFrequency.get(word) || 1));
        return tf * idf;
      });
    });

    return { vectors, vocabulary };
  }
}
