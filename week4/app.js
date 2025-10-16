import DataLoader from './data-loader.js';
import GRUModel from './gru.js';

class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = null;
        this.currentPredictions = null;
        this.accuracyChart = null;
        this.isTraining = false;
        
        this.initializeEventListeners();
        console.log('App initialized');
    }

    initializeEventListeners() {
        const fileInput = document.getElementById('csvFile');
        const trainBtn = document.getElementById('trainBtn');
        const predictBtn = document.getElementById('predictBtn');

        fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        trainBtn.addEventListener('click', () => this.trainModel());
        predictBtn.addEventListener('click', () => this.runPrediction());
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        try {
            document.getElementById('status').textContent = 'Loading CSV...';
            await this.dataLoader.loadCSV(file);
            
            document.getElementById('status').textContent = 'Preprocessing data...';
            const result = this.dataLoader.createSequences();
            
            document.getElementById('trainBtn').disabled = false;
            document.getElementById('status').textContent = 
                `Data loaded: ${result.symbols.length} stocks, ${result.X_train.shape[0]} training sequences. Click Train Model to begin training.`;
            
        } catch (error) {
            document.getElementById('status').textContent = `Error: ${error.message}`;
            console.error('File upload error:', error);
        }
    }

    async trainModel() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        const trainBtn = document.getElementById('trainBtn');
        const predictBtn = document.getElementById('predictBtn');
        
        trainBtn.disabled = true;
        predictBtn.disabled = true;

        try {
            const { X_train, y_train, X_test, y_test, symbols } = this.dataLoader;
            
            if (!X_train || X_train.shape[0] === 0) {
                throw new Error('No training data available. Please check your CSV file.');
            }

            console.log('Training data shape:', X_train.shape);
            console.log('Training labels shape:', y_train.shape);

            // Create model with correct input shape
            const sequenceLength = X_train.shape[1];
            const featuresPerTimestep = X_train.shape[2];
            this.model = new GRUModel([sequenceLength, featuresPerTimestep], y_train.shape[1]);
            
            document.getElementById('status').textContent = 'Training model...';
            await this.model.train(X_train, y_train, X_test, y_test, 20, 32); // Reduced epochs for testing
            
            predictBtn.disabled = false;
            document.getElementById('status').textContent = 'Training completed. Click Run Prediction to evaluate.';
            
        } catch (error) {
            document.getElementById('status').textContent = `Training failed: ${error.message}`;
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
            trainBtn.disabled = false;
        }
    }

    async runPrediction() {
        if (!this.model) {
            alert('Please train the model first');
            return;
        }

        try {
            document.getElementById('status').textContent = 'Running predictions...';
            const { X_test, y_test, symbols } = this.dataLoader;
            
            const predictions = await this.model.predict(X_test);
            const evaluation = this.model.evaluatePerStock(y_test, predictions, symbols);
            
            this.currentPredictions = evaluation;
            this.visualizeResults(evaluation, symbols);
            
            document.getElementById('status').textContent = 'Prediction completed. Results displayed below.';
            
            // Clean up
            predictions.dispose();
            
        } catch (error) {
            document.getElementById('status').textContent = `Prediction error: ${error.message}`;
            console.error('Prediction error:', error);
        }
    }

    visualizeResults(evaluation, symbols) {
        this.createAccuracyChart(evaluation.stockAccuracies, symbols);
        this.createTimelineCharts(evaluation.stockPredictions, symbols);
    }

    createAccuracyChart(accuracies, symbols) {
        const ctx = document.getElementById('accuracyChart');
        if (!ctx) {
            console.error('Accuracy chart canvas not found');
            return;
        }
        
        const chartContext = ctx.getContext('2d');
        
        // Sort stocks by accuracy
        const sortedEntries = Object.entries(accuracies)
            .sort(([,a], [,b]) => b - a);
        
        const sortedSymbols = sortedEntries.map(([symbol]) => symbol);
        const sortedAccuracies = sortedEntries.map(([, accuracy]) => accuracy * 100);

        if (this.accuracyChart) {
            this.accuracyChart.destroy();
        }

        this.accuracyChart = new Chart(chartContext, {
            type: 'bar',
            data: {
                labels: sortedSymbols,
                datasets: [{
                    label: 'Prediction Accuracy (%)',
                    data: sortedAccuracies,
                    backgroundColor: sortedAccuracies.map(acc => 
                        acc > 60 ? 'rgba(75, 192, 192, 0.8)' : 
                        acc > 50 ? 'rgba(255, 205, 86, 0.8)' : 
                        'rgba(255, 99, 132, 0.8)'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => `Accuracy: ${context.raw.toFixed(2)}%`
                        }
                    }
                }
            }
        });
    }

    createTimelineCharts(predictions, symbols) {
        const container = document.getElementById('timelineContainer');
        if (!container) {
            console.error('Timeline container not found');
            return;
        }
        
        container.innerHTML = '';

        // Show top 3 stocks by accuracy for timeline visualization
        const topStocks = Object.keys(predictions).slice(0, 3);

        topStocks.forEach(symbol => {
            const stockPredictions = predictions[symbol];
            const chartContainer = document.createElement('div');
            chartContainer.className = 'stock-chart';
            chartContainer.innerHTML = `<h4>${symbol} Prediction Timeline</h4><canvas id="timeline-${symbol}"></canvas>`;
            container.appendChild(chartContainer);

            const canvas = document.getElementById(`timeline-${symbol}`);
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            
            // Sample first 30 predictions for cleaner visualization
            const sampleSize = Math.min(30, stockPredictions.length);
            const sampleData = stockPredictions.slice(0, sampleSize);
            
            const correctData = sampleData.map(p => p.correct ? 1 : 0);
            const labels = sampleData.map((_, i) => `Pred ${i + 1}`);

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Correct Predictions',
                        data: correctData,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: sampleData.map(p => 
                            p.correct ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)'
                        )
                    }]
                },
                options: {
                    scales: {
                        y: {
                            min: 0,
                            max: 1,
                            ticks: {
                                callback: (value) => value === 1 ? 'Correct' : value === 0 ? 'Wrong' : ''
                            }
                        }
                    }
                }
            });
        });
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new StockPredictionApp();
});
