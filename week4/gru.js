class GRUModel {
    constructor(inputShape, outputSize) {
        this.model = null;
        this.inputShape = inputShape;
        this.outputSize = outputSize;
        this.history = null;
    }

    buildModel() {
        try {
            this.model = tf.sequential();
            
            // First GRU layer
            this.model.add(tf.layers.gru({
                units: 32,
                returnSequences: true,
                inputShape: this.inputShape
            }));
            
            // Dropout for regularization
            this.model.add(tf.layers.dropout({rate: 0.2}));
            
            // Second GRU layer
            this.model.add(tf.layers.gru({
                units: 16,
                returnSequences: false
            }));
            
            // Dropout for regularization
            this.model.add(tf.layers.dropout({rate: 0.2}));
            
            // Output layer
            this.model.add(tf.layers.dense({
                units: this.outputSize,
                activation: 'sigmoid'
            }));

            // Use simpler optimizer configuration
            this.model.compile({
                optimizer: 'adam',
                loss: 'binaryCrossentropy',
                metrics: ['accuracy']
            });

            console.log('Model built successfully');
            console.log('Input shape:', this.inputShape);
            console.log('Output size:', this.outputSize);
            
            return this.model;
        } catch (error) {
            console.error('Error building model:', error);
            throw error;
        }
    }

    async train(X_train, y_train, X_test, y_test, epochs = 10, batchSize = 16) {
        if (!this.model) {
            this.buildModel();
        }

        try {
            // Reset progress
            const progressElement = document.getElementById('trainingProgress');
            if (progressElement) {
                progressElement.value = 0;
            }

            console.log('Starting training...');
            console.log('X_train shape:', X_train.shape);
            console.log('y_train shape:', y_train.shape);
            
            this.history = await this.model.fit(X_train, y_train, {
                epochs: epochs,
                batchSize: batchSize,
                validationData: [X_test, y_test],
                verbose: 0, // Reduce console noise
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        const progress = ((epoch + 1) / epochs) * 100;
                        const status = `Epoch ${epoch + 1}/${epochs} - loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`;
                        
                        // Update UI
                        const progressElement = document.getElementById('trainingProgress');
                        const statusElement = document.getElementById('status');
                        if (progressElement) progressElement.value = progress;
                        if (statusElement) statusElement.textContent = status;
                        
                        console.log(status);
                    }
                }
            });

            console.log('Training completed successfully');
            return this.history;
        } catch (error) {
            console.error('Training error:', error);
            throw error;
        }
    }

    async predict(X) {
        if (!this.model) throw new Error('Model not trained');
        return this.model.predict(X);
    }

    evaluatePerStock(yTrue, yPred, symbols, horizon = 2) {
        try {
            const yTrueArray = yTrue.arraySync();
            const yPredArray = yPred.arraySync();
            
            const stockAccuracies = {};
            const stockPredictions = {};

            symbols.forEach((symbol, stockIdx) => {
                let correct = 0;
                let total = 0;
                const predictions = [];

                for (let i = 0; i < yTrueArray.length; i++) {
                    for (let offset = 0; offset < horizon; offset++) {
                        const targetIdx = stockIdx * horizon + offset;
                        if (targetIdx < yTrueArray[i].length && targetIdx < yPredArray[i].length) {
                            const trueVal = yTrueArray[i][targetIdx];
                            const predVal = yPredArray[i][targetIdx] > 0.5 ? 1 : 0;
                            
                            if (trueVal === predVal) correct++;
                            total++;
                            
                            predictions.push({
                                true: trueVal,
                                pred: predVal,
                                correct: trueVal === predVal
                            });
                        }
                    }
                }

                stockAccuracies[symbol] = total > 0 ? correct / total : 0;
                stockPredictions[symbol] = predictions;
            });

            return { stockAccuracies, stockPredictions };
        } catch (error) {
            console.error('Evaluation error:', error);
            throw error;
        }
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}

export default GRUModel;
