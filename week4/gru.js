class GRUModel {
    constructor(inputShape, outputSize) {
        this.model = null;
        this.inputShape = inputShape;
        this.outputSize = outputSize;
        this.history = null;
    }

    buildModel() {
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 64,
                    returnSequences: true,
                    inputShape: this.inputShape
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.gru({
                    units: 32,
                    returnSequences: false
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({
                    units: this.outputSize,
                    activation: 'sigmoid'
                })
            ]
        });

        this.model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        console.log('Model built successfully');
        return this.model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 50, batchSize = 32) {
        if (!this.model) {
            this.buildModel();
        }

        // Reset progress
        const progressElement = document.getElementById('trainingProgress');
        if (progressElement) {
            progressElement.value = 0;
        }

        try {
            this.history = await this.model.fit(X_train, y_train, {
                epochs: epochs,
                batchSize: batchSize,
                validationData: [X_test, y_test],
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

    evaluatePerStock(yTrue, yPred, symbols, horizon = 3) {
        // Convert tensors to arrays
        const yTrueArray = Array.isArray(yTrue) ? yTrue : yTrue.arraySync();
        const yPredArray = Array.isArray(yPred) ? yPred : yPred.arraySync();
        
        const stockAccuracies = {};
        const stockPredictions = {};

        symbols.forEach((symbol, stockIdx) => {
            let correct = 0;
            let total = 0;
            const predictions = [];

            for (let i = 0; i < yTrueArray.length; i++) {
                for (let offset = 0; offset < horizon; offset++) {
                    const targetIdx = stockIdx * horizon + offset;
                    if (targetIdx >= yTrueArray[i].length) continue;
                    
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

            stockAccuracies[symbol] = total > 0 ? correct / total : 0;
            stockPredictions[symbol] = predictions;
        });

        return { stockAccuracies, stockPredictions };
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}

export default GRUModel;
