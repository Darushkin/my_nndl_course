import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/+esm';
import { DataLoader } from './data-loader.js';
import { GRUModel } from './gru.js';

export class StockPredictorApp {
  constructor() {
    this.dataLoader = new DataLoader();
    this.model = null;
    this.isTraining = false;
    this.results = null;
    
    this.initializeUI();
  }

  initializeUI() {
    const fileInput = document.getElementById('csvFile');
    const trainBtn = document.getElementById('trainBtn');
    const progress = document.getElementById('progress');
    const resultsDiv = document.getElementById('results');
    
    fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
    trainBtn.addEventListener('click', () => this.startTraining());
  }

  async handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    try {
      document.getElementById('status').textContent = 'Loading CSV...';
      await this.dataLoader.loadCSV(file);
      document.getElementById('status').textContent = 'CSV loaded successfully!';
    } catch (error) {
      document.getElementById('status').textContent = `Error: ${error.message}`;
    }
  }

  async startTraining() {
    if (this.isTraining) return;
    
    try {
      this.isTraining = true;
      document.getElementById('status').textContent = 'Preparing data...';
      
      const { X_train, y_train, X_test, y_test, symbols } = this.dataLoader.prepareFeatures();
      
      document.getElementById('status').textContent = 'Building model...';
      this.model = new GRUModel([12, symbols.length * 2], symbols.length * 3);
      this.model.buildModel();
      
      document.getElementById('status').textContent = 'Training model...';
      await this.model.train(X_train, y_train, X_test, y_test, 30, 32);
      
      document.getElementById('status').textContent = 'Evaluating model...';
      const predictions = await this.model.predict(X_test);
      const stockAccuracies = this.model.calculateStockAccuracies(predictions, y_test, symbols);
      
      this.results = { stockAccuracies, symbols, predictions, y_test };
      this.displayResults();
      
      predictions.dispose();
      
    } catch (error) {
      document.getElementById('status').textContent = `Training failed: ${error.message}`;
    } finally {
      this.isTraining = false;
    }
  }

  displayResults() {
    if (!this.results) return;
    
    this.createAccuracyChart();
    this.createPredictionTimelines();
  }

  createAccuracyChart() {
    const { stockAccuracies, symbols } = this.results;
    
    // Sort stocks by accuracy
    const sortedStocks = symbols.sort((a, b) => stockAccuracies[b] - stockAccuracies[a]);
    const sortedAccuracies = sortedStocks.map(symbol => stockAccuracies[symbol]);
    
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    
    // Clear previous chart
    if (this.accuracyChart) {
      this.accuracyChart.destroy();
    }
    
    this.accuracyChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: sortedStocks,
        datasets: [{
          label: 'Prediction Accuracy',
          data: sortedAccuracies,
          backgroundColor: sortedAccuracies.map(acc => 
            acc > 0.6 ? '#4CAF50' : acc > 0.5 ? '#FFC107' : '#F44336'
          ),
          borderColor: '#333',
          borderWidth: 1
        }]
      },
      options: {
        indexAxis: 'y',
        scales: {
          x: {
            beginAtZero: true,
            max: 1.0,
            title: {
              display: true,
              text: 'Accuracy'
            }
          }
        },
        plugins: {
          title: {
            display: true,
            text: 'Stock Prediction Accuracy Ranking'
          },
          tooltip: {
            callbacks: {
              label: (context) => `Accuracy: ${(context.raw * 100).toFixed(2)}%`
            }
          }
        }
      }
    });
  }

  createPredictionTimelines() {
    const { symbols, predictions, y_test } = this.results;
    const predData = predictions.arraySync();
    const trueData = y_test.arraySync();
    
    const container = document.getElementById('timelineContainer');
    container.innerHTML = '';
    
    // Show top 5 stocks for timeline visualization
    const topStocks = symbols
      .map(symbol => ({ symbol, accuracy: this.results.stockAccuracies[symbol] }))
      .sort((a, b) => b.accuracy - a.accuracy)
      .slice(0, 5);
    
    topStocks.forEach(({ symbol }, stockIdx) => {
      const stockDiv = document.createElement('div');
      stockDiv.className = 'stock-timeline';
      stockDiv.innerHTML = `<h4>${symbol} Predictions</h4><div class="timeline" id="timeline-${symbol}"></div>`;
      container.appendChild(stockDiv);
      
      this.createSingleTimeline(symbol, stockIdx, predData, trueData);
    });
  }

  createSingleTimeline(symbol, stockIdx, predData, trueData) {
    const daysAhead = 3;
    const stockStartIdx = stockIdx * daysAhead;
    const sampleCount = Math.min(50, predData.length); // Show first 50 samples
    
    const timeline = document.getElementById(`timeline-${symbol}`);
    timeline.innerHTML = '';
    
    for (let sample = 0; sample < sampleCount; sample++) {
      const sampleDiv = document.createElement('div');
      sampleDiv.className = 'timeline-sample';
      
      for (let day = 0; day < daysAhead; day++) {
        const pred = predData[sample][stockStartIdx + day] > 0.5 ? 1 : 0;
        const trueVal = trueData[sample][stockStartIdx + day];
        const isCorrect = pred === trueVal;
        
        const dayDiv = document.createElement('div');
        dayDiv.className = `timeline-day ${isCorrect ? 'correct' : 'wrong'}`;
        dayDiv.title = `Sample ${sample}, Day ${day + 1}: Predicted ${pred}, Actual ${trueVal}`;
        sampleDiv.appendChild(dayDiv);
      }
      
      timeline.appendChild(sampleDiv);
    }
  }

  dispose() {
    if (this.dataLoader) {
      this.dataLoader.dispose();
    }
    if (this.model) {
      this.model.dispose();
    }
    tf.disposeVariables();
  }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.app = new StockPredictorApp();
});
