import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/+esm';

export class GRUModel {
  constructor(inputShape, outputSize) {
    this.model = null;
    this.inputShape = inputShape;
    this.outputSize = outputSize;
    this.history = null;
  }

  buildModel() {
    const model = tf.sequential();
    
    // CNN layer for feature extraction
    model.add(tf.layers.conv1d({
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
      inputShape: this.inputShape
    }));
    
    // Bidirectional GRU layers
    model.add(tf.layers.bidirectional({
      layer: tf.layers.gru({ units: 64, returnSequences: true })
    }));
    
    model.add(tf.layers.bidirectional({
      layer: tf.layers.gru({ units: 32 })
    }));
    
    // Dense layers
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    
    // Output layer - 10 stocks × 3 days = 30 binary outputs
    model.add(tf.layers.dense({
      units: this.outputSize,
      activation: 'sigmoid'
    }));
    
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });
    
    this.model = model;
    return model;
  }

  async train(X_train, y_train, X_test, y_test, epochs = 50, batchSize = 32) {
    this.history = await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      validationData: [X_test, y_test],
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, acc = ${logs.binaryAccuracy.toFixed(4)}`);
        }
      }
    });
    
    return this.history;
  }

  async predict(X) {
    return this.model.predict(X);
  }

  evaluate(X_test, y_test) {
    return this.model.evaluate(X_test, y_test);
  }

  calculateStockAccuracies(predictions, y_true, symbols, daysAhead = 3) {
    const predData = predictions.arraySync();
    const trueData = y_true.arraySync();
    
    const stockAccuracies = {};
    symbols.forEach((symbol, stockIdx) => {
      const stockStartIdx = stockIdx * daysAhead;
      let correct = 0;
      let total = 0;
      
      for (let sample = 0; sample < predData.length; sample++) {
        for (let day = 0; day < daysAhead; day++) {
          const pred = predData[sample][stockStartIdx + day] > 0.5 ? 1 : 0;
          const trueVal = trueData[sample][stockStartIdx + day];
          
          if (pred === trueVal) {
            correct++;
          }
          total++;
        }
      }
      
      stockAccuracies[symbol] = correct / total;
    });
    
    return stockAccuracies;
  }

  async saveModel(name = 'gru-model') {
    await this.model.save(`indexeddb://${name}`);
  }

  async loadModel(name = 'gru-model') {
    this.model = await tf.loadLayersModel(`indexeddb://${name}`);
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
}
