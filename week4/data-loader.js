import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/+esm';

export class DataLoader {
  constructor() {
    this.data = null;
    this.symbols = [];
    this.dates = [];
    this.X_train = null;
    this.y_train = null;
    this.X_test = null;
    this.y_test = null;
    this.featureScalers = [];
    this.trainTestSplit = 0.8;
  }

  async loadCSV(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const csv = e.target.result;
          this.parseCSV(csv);
          resolve();
        } catch (error) {
          reject(error);
        }
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  }

  parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',');
    
    const rawData = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      if (values.length !== headers.length) continue;
      
      const row = {};
      headers.forEach((header, index) => {
        row[header.trim()] = values[index].trim();
      });
      rawData.push(row);
    }

    this.processRawData(rawData);
  }

  processRawData(rawData) {
    // Group by symbol and sort by date
    const bySymbol = {};
    const allDates = new Set();
    
    rawData.forEach(row => {
      const symbol = row.Symbol;
      if (!bySymbol[symbol]) {
        bySymbol[symbol] = [];
      }
      bySymbol[symbol].push({
        date: row.Date,
        open: parseFloat(row.Open),
        close: parseFloat(row.Close),
        high: parseFloat(row.High),
        low: parseFloat(row.Low),
        volume: parseFloat(row.Volume)
      });
      allDates.add(row.Date);
    });

    this.symbols = Object.keys(bySymbol);
    this.dates = Array.from(allDates).sort();
    
    // Sort each symbol's data by date
    this.symbols.forEach(symbol => {
      bySymbol[symbol].sort((a, b) => new Date(a.date) - new Date(b.date));
    });

    this.data = bySymbol;
  }

  prepareFeatures() {
    const sequenceLength = 12;
    const predictionHorizon = 3;
    const featuresPerStock = 2; // Open, Close
    
    // Normalize data per stock
    this.normalizeData();
    
    const samples = [];
    const labels = [];
    
    // Create sliding window samples
    for (let i = sequenceLength; i < this.dates.length - predictionHorizon; i++) {
      const currentDate = this.dates[i];
      const sequenceData = [];
      
      // Get 12-day sequence for all stocks
      for (let j = i - sequenceLength; j < i; j++) {
        const date = this.dates[j];
        const dayFeatures = [];
        
        this.symbols.forEach(symbol => {
          const stockData = this.data[symbol].find(d => d.date === date);
          if (stockData) {
            dayFeatures.push(stockData.normalizedOpen, stockData.normalizedClose);
          } else {
            dayFeatures.push(0, 0); // Padding for missing data
          }
        });
        
        sequenceData.push(dayFeatures);
      }
      
      // Create labels for each stock for next 3 days
      const currentLabels = [];
      this.symbols.forEach(symbol => {
        const currentData = this.data[symbol].find(d => d.date === currentDate);
        if (!currentData) {
          currentLabels.push(...Array(predictionHorizon).fill(0));
          return;
        }
        
        const currentClose = currentData.close;
        
        for (let offset = 1; offset <= predictionHorizon; offset++) {
          const futureDate = this.dates[i + offset];
          const futureData = this.data[symbol].find(d => d.date === futureDate);
          
          if (futureData && futureData.close > currentClose) {
            currentLabels.push(1);
          } else {
            currentLabels.push(0);
          }
        }
      });
      
      samples.push(sequenceData);
      labels.push(currentLabels);
    }
    
    // Split into train/test
    const splitIndex = Math.floor(samples.length * this.trainTestSplit);
    
    this.X_train = tf.tensor3d(samples.slice(0, splitIndex));
    this.X_test = tf.tensor3d(samples.slice(splitIndex));
    this.y_train = tf.tensor2d(labels.slice(0, splitIndex));
    this.y_test = tf.tensor2d(labels.slice(splitIndex));
    
    return {
      X_train: this.X_train,
      y_train: this.y_train,
      X_test: this.X_test,
      y_test: this.y_test,
      symbols: this.symbols
    };
  }

  normalizeData() {
    this.featureScalers = [];
    
    this.symbols.forEach(symbol => {
      const stockData = this.data[symbol];
      const opens = stockData.map(d => d.open);
      const closes = stockData.map(d => d.close);
      const highs = stockData.map(d => d.high);
      const lows = stockData.map(d => d.low);
      const volumes = stockData.map(d => d.volume);
      
      const openMin = Math.min(...opens);
      const openMax = Math.max(...opens);
      const closeMin = Math.min(...closes);
      const closeMax = Math.max(...closes);
      
      stockData.forEach(day => {
        day.normalizedOpen = (day.open - openMin) / (openMax - openMin);
        day.normalizedClose = (day.close - closeMin) / (closeMax - closeMin);
      });
      
      this.featureScalers.push({
        symbol,
        open: { min: openMin, max: openMax },
        close: { min: closeMin, max: closeMax }
      });
    });
  }

  dispose() {
    if (this.X_train) this.X_train.dispose();
    if (this.y_train) this.y_train.dispose();
    if (this.X_test) this.X_test.dispose();
    if (this.y_test) this.y_test.dispose();
  }
}
