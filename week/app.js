// Titanic Binary Classifier using TensorFlow.js
// Target: Survived (0/1)
// Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
// Identifier: PassengerId (excluded from features)

// Global variables
let trainData = null;
let testData = null;
let processedTrainData = null;
let processedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let rocData = null;
let auc = 0;

// Data Load Function
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    const statusDiv = document.getElementById('data-status');
    const previewDiv = document.getElementById('data-preview');

    if (!trainFile || !testFile) {
        statusDiv.innerHTML = '<div class="status error">Please select both training and test files</div>';
        return;
    }

    statusDiv.innerHTML = '<div class="status info">Loading data...</div>';

    try {
        // Load training data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        // Load test data  
        const testText = await readFile(testFile);
        testData = parseCSV(testText);

        statusDiv.innerHTML = '<div class="status success">Data loaded successfully!</div>';
        
        // Show data preview
        showDataPreview();
        
        // Enable preprocessing button
        document.getElementById('preprocess-btn').disabled = false;
        
    } catch (error) {
        statusDiv.innerHTML = `<div class="status error">Error loading data: ${error.message}</div>`;
    }
}

// File reader helper
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('File reading failed'));
        reader.readAsText(file);
    });
}

// CSV parser
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => v.trim());
        const row = {};
        headers.forEach((header, index) => {
            row[header] = values[index] || null;
        });
        data.push(row);
    }
    
    return data;
}

// Show data preview
function showDataPreview() {
    const previewDiv = document.getElementById('data-preview');
    
    if (!trainData || !testData) return;
    
    const trainShape = `${trainData.length} rows × ${Object.keys(trainData[0]).length} columns`;
    const testShape = `${testData.length} rows × ${Object.keys(testData[0]).length} columns`;
    
    // Calculate missing values
    const trainMissing = calculateMissingPercentage(trainData);
    const testMissing = calculateMissingPercentage(testData);
    
    // Show survival distribution
    showSurvivalCharts();
    
    previewDiv.innerHTML = `
        <h3>Data Overview</h3>
        <p><strong>Training Data:</strong> ${trainShape}</p>
        <p><strong>Test Data:</strong> ${testShape}</p>
        
        <h4>Missing Values (%) - Training Data</h4>
        <ul>
            ${Object.entries(trainMissing).map(([key, value]) => 
                `<li>${key}: ${value.toFixed(2)}%</li>`
            ).join('')}
        </ul>
        
        <h4>First 5 Rows - Training Data</h4>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <thead>
                <tr>
                    ${Object.keys(trainData[0]).map(h => `<th>${h}</th>`).join('')}
                </tr>
            </thead>
            <tbody>
                ${trainData.slice(0, 5).map(row => `
                    <tr>
                        ${Object.values(row).map(val => `<td>${val}</td>`).join('')}
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

// Calculate missing percentage
function calculateMissingPercentage(data) {
    const missing = {};
    const totalRows = data.length;
    
    if (totalRows === 0) return missing;
    
    Object.keys(data[0]).forEach(key => {
        const missingCount = data.filter(row => row[key] === null || row[key] === '' || row[key] === undefined).length;
        missing[key] = (missingCount / totalRows) * 100;
    });
    
    return missing;
}

// Show survival charts using tfjs-vis
function showSurvivalCharts() {
    if (!trainData) return;
    
    // Survival by Sex
    const sexSurvival = {};
    trainData.forEach(row => {
        if (row.Sex && row.Survived !== undefined) {
            const sex = row.Sex;
            const survived = parseInt(row.Survived);
            if (!sexSurvival[sex]) {
                sexSurvival[sex] = { survived: 0, total: 0 };
            }
            sexSurvival[sex].total++;
            if (survived === 1) sexSurvival[sex].survived++;
        }
    });
    
    const sexData = Object.entries(sexSurvival).map(([sex, stats]) => ({
        sex,
        survival_rate: (stats.survived / stats.total) * 100
    }));
    
    // Survival by Pclass
    const classSurvival = {};
    trainData.forEach(row => {
        if (row.Pclass && row.Survived !== undefined) {
            const pclass = `Class ${row.Pclass}`;
            const survived = parseInt(row.Survived);
            if (!classSurvival[pclass]) {
                classSurvival[pclass] = { survived: 0, total: 0 };
            }
            classSurvival[pclass].total++;
            if (survived === 1) classSurvival[pclass].survived++;
        }
    });
    
    const classData = Object.entries(classSurvival).map(([pclass, stats]) => ({
        pclass,
        survival_rate: (stats.survived / stats.total) * 100
    }));
    
    // Create charts
    const surface = { name: 'Survival Analysis', tab: 'Data Exploration' };
    
    tfvis.render.barchart(surface, {
        values: sexData.map(d => d.survival_rate),
        labels: sexData.map(d => d.sex)
    }, {
        xLabel: 'Sex',
        yLabel: 'Survival Rate (%)',
        width: 400,
        height: 300
    });
    
    tfvis.render.barchart(surface, {
        values: classData.map(d => d.survival_rate),
        labels: classData.map(d => d.pclass)
    }, {
        xLabel: 'Passenger Class',
        yLabel: 'Survival Rate (%)',
        width: 400,
        height: 300
    });
}

// Preprocessing Function
function preprocessData() {
    const statusDiv = document.getElementById('preprocess-status');
    const addFamilyFeatures = document.getElementById('family-features').checked;
    
    statusDiv.innerHTML = '<div class="status info">Preprocessing data...</div>';
    
    try {
        // Process training data
        processedTrainData = preprocessDataset(trainData, true, addFamilyFeatures);
        
        // Process test data
        processedTestData = preprocessDataset(testData, false, addFamilyFeatures);
        
        statusDiv.innerHTML = `
            <div class="status success">
                Preprocessing completed!<br>
                Training Features: ${processedTrainData.features.shape}<br>
                Training Labels: ${processedTrainData.labels.shape}<br>
                Test Features: ${processedTestData.features.shape}
            </div>
        `;
        
        // Enable model creation button
        document.getElementById('create-model-btn').disabled = false;
        
    } catch (error) {
        statusDiv.innerHTML = `<div class="status error">Preprocessing error: ${error.message}</div>`;
    }
}

// Dataset preprocessing
function preprocessDataset(data, isTraining, addFamilyFeatures) {
    const features = [];
    const labels = [];
    const passengerIds = [];
    
    // Calculate medians/modes from training data only
    let ageMedian = 28;
    let fareMedian = 14.45;
    let embarkedMode = 'S';
    
    if (isTraining) {
        const ages = data.map(row => parseFloat(row.Age)).filter(age => !isNaN(age));
        const fares = data.map(row => parseFloat(row.Fare)).filter(fare => !isNaN(fare));
        const embarked = data.map(row => row.Embarked).filter(e => e);
        
        ageMedian = ages.length > 0 ? tf.median(tf.tensor1d(ages)).dataSync()[0] : 28;
        fareMedian = fares.length > 0 ? tf.median(tf.tensor1d(fares)).dataSync()[0] : 14.45;
        
        // Mode calculation for Embarked
        const embarkedCount = {};
        embarked.forEach(e => {
            embarkedCount[e] = (embarkedCount[e] || 0) + 1;
        });
        embarkedMode = Object.keys(embarkedCount).reduce((a, b) => 
            embarkedCount[a] > embarkedCount[b] ? a : b, 'S'
        );
    }
    
    data.forEach(row => {
        const featureVector = [];
        passengerIds.push(row.PassengerId);
        
        // Pclass (one-hot encoding)
        const pclass = parseInt(row.Pclass) || 3;
        featureVector.push(pclass === 1 ? 1 : 0);
        featureVector.push(pclass === 2 ? 1 : 0);
        featureVector.push(pclass === 3 ? 1 : 0);
        
        // Sex (one-hot: male=1, female=0)
        featureVector.push(row.Sex === 'male' ? 1 : 0);
        
        // Age (impute with median, then standardize)
        let age = parseFloat(row.Age);
        if (isNaN(age)) age = ageMedian;
        featureVector.push((age - ageMedian) / 20); // Rough standardization
        
        // SibSp and Parch
        featureVector.push(parseInt(row.SibSp) || 0);
        featureVector.push(parseInt(row.Parch) || 0);
        
        // Fare (impute with median, then standardize)
        let fare = parseFloat(row.Fare);
        if (isNaN(fare)) fare = fareMedian;
        featureVector.push((fare - fareMedian) / 30); // Rough standardization
        
        // Embarked (one-hot encoding)
        let embarked = row.Embarked || embarkedMode;
        featureVector.push(embarked === 'C' ? 1 : 0);
        featureVector.push(embarked === 'Q' ? 1 : 0);
        featureVector.push(embarked === 'S' ? 1 : 0);
        
        // Optional family features
        if (addFamilyFeatures) {
            const sibSp = parseInt(row.SibSp) || 0;
            const parch = parseInt(row.Parch) || 0;
            const familySize = sibSp + parch + 1;
            const isAlone = familySize === 1 ? 1 : 0;
            
            featureVector.push(familySize);
            featureVector.push(isAlone);
        }
        
        features.push(featureVector);
        
        // Labels for training data only
        if (isTraining && row.Survived !== undefined) {
            labels.push(parseInt(row.Survived));
        }
    });
    
    return {
        features: tf.tensor2d(features),
        labels: isTraining ? tf.tensor1d(labels) : null,
        passengerIds: passengerIds,
        featureNames: getFeatureNames(addFamilyFeatures)
    };
}

// Get feature names for interpretation
function getFeatureNames(addFamilyFeatures) {
    const baseFeatures = [
        'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Age_std',
        'SibSp', 'Parch', 'Fare_std', 'Embarked_C', 'Embarked_Q', 'Embarked_S'
    ];
    
    if (addFamilyFeatures) {
        baseFeatures.push('FamilySize', 'IsAlone');
    }
    
    return baseFeatures;
}

// Model Creation Function
function createModel() {
    const statusDiv = document.getElementById('model-status');
    const summaryDiv = document.getElementById('model-summary');
    
    statusDiv.innerHTML = '<div class="status info">Creating model...</div>';
    
    try {
        const inputShape = processedTrainData.features.shape[1];
        
        model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputShape],
                    units: 16,
                    activation: 'relu',
                    name: 'hidden_layer'
                }),
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid',
                    name: 'output_layer'
                })
            ]
        });
        
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        statusDiv.innerHTML = '<div class="status success">Model created successfully!</div>';
        
        // Print model summary
        summaryDiv.innerHTML = `
            <h4>Model Summary</h4>
            <pre>${model.summary()}</pre>
            <p><strong>Architecture:</strong> Input(${inputShape}) → Dense(16, relu) → Dense(1, sigmoid)</p>
        `;
        
        // Enable training button
        document.getElementById('train-btn').disabled = false;
        
    } catch (error) {
        statusDiv.innerHTML = `<div class="status error">Model creation error: ${error.message}</div>`;
    }
}

// Training Function
async function trainModel() {
    const statusDiv = document.getElementById('training-status');
    const plotsDiv = document.getElementById('training-plots');
    
    statusDiv.innerHTML = '<div class="status info">Training model...</div>';
    
    try {
        // Create validation split (80/20)
        const splitIndex = Math.floor(processedTrainData.features.shape[0] * 0.8);
        
        const trainFeatures = processedTrainData.features.slice(0, splitIndex);
        const trainLabels = processedTrainData.labels.slice(0, splitIndex);
        validationData = processedTrainData.features.slice(splitIndex);
        validationLabels = processedTrainData.labels.slice(splitIndex);
        
        // Training callbacks
        const callbacks = tfvis.show.fitCallbacks(
            { name: 'Training Performance', tab: 'Training' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { 
                callbacks: ['onEpochEnd'],
                height: 300,
                width: 600 
            }
        );
        
        // Add early stopping
        callbacks.push({
            onEpochEnd: async (epoch, logs) => {
                if (epoch > 5 && logs.val_loss > trainingHistory.val_loss[epoch-1]) {
                    // Simple early stopping logic
                    console.log(`Early stopping considered at epoch ${epoch}`);
                }
            }
        });
        
        // Train model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationSplit: 0.2,
            callbacks: callbacks,
            verbose: 1
        });
        
        statusDiv.innerHTML = '<div class="status success">Training completed!</div>';
        
        // Enable evaluation button
        document.getElementById('evaluate-btn').disabled = false;
        document.getElementById('predict-btn').disabled = false;
        document.getElementById('export-model-btn').disabled = false;
        
    } catch (error) {
        statusDiv.innerHTML = `<div class="status error">Training error: ${error.message}</div>`;
    }
}

// Model Evaluation Function
async function evaluateModel() {
    const statusDiv = document.getElementById('metrics-status');
    
    statusDiv.innerHTML = '<div class="status info">Evaluating model...</div>';
    
    try {
        // Get predictions on validation set
        validationPredictions = model.predict(validationData);
        const probs = await validationPredictions.data();
        const trueLabels = await validationLabels.data();
        
        // Calculate ROC curve and AUC
        rocData = calculateROC(trueLabels, probs);
        auc = calculateAUC(rocData);
        
        // Plot ROC curve
        plotROC(rocData, auc);
        
        // Update metrics with default threshold
        updateThreshold(0.5);
        
        statusDiv.innerHTML = `<div class="status success">Evaluation completed! AUC: ${auc.toFixed(4)}</div>`;
        
    } catch (error) {
        statusDiv.innerHTML = `<div class="status error">Evaluation error: ${error.message}</div>`;
    }
}

// Calculate ROC curve
function calculateROC(trueLabels, probabilities) {
    const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
    const rocPoints = [];
    
    thresholds.forEach(threshold => {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (let i = 0; i < trueLabels.length; i++) {
            const prediction = probabilities[i] >= threshold ? 1 : 0;
            const actual = trueLabels[i];
            
            if (actual === 1 && prediction === 1) tp++;
            else if (actual === 0 && prediction === 1) fp++;
            else if (actual === 0 && prediction === 0) tn++;
            else if (actual === 1 && prediction === 0) fn++;
        }
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        rocPoints.push({ threshold, fpr, tpr, tp, fp, tn, fn });
    });
    
    return rocPoints;
}

// Calculate AUC (Area Under Curve)
function calculateAUC(rocData) {
    let auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        const prev = rocData[i - 1];
        const curr = rocData[i];
        auc += (curr.fpr - prev.fpr) * (curr.tpr + prev.tpr) / 2;
    }
    return auc;
}

// Plot ROC curve
function plotROC(rocData, auc) {
    const rocDiv = document.getElementById('roc-plot');
    
    const surface = { name: 'ROC Curve', tab: 'Metrics' };
    
    const rocValues = rocData.map(point => ({ x: point.fpr, y: point.tpr }));
    
    tfvis.render.scatterplot(surface, {
        values: rocValues,
        series: ['ROC Curve']
    }, {
        xLabel: 'False Positive Rate',
        yLabel: 'True Positive Rate',
        width: 500,
        height: 400,
        seriesColors: ['blue'],
        zoomToFit: true
    });
    
    // Add AUC to plot area
    rocDiv.innerHTML = `<p><strong>AUC: ${auc.toFixed(4)}</strong></p>`;
}

// Update metrics based on threshold
function updateThreshold(threshold) {
    const thresholdValue = document.getElementById('threshold-value');
    const metricsDisplay = document.getElementById('metrics-display');
    
    thresholdValue.textContent = threshold;
    
    if (!rocData) return;
    
    // Find closest ROC point to threshold
    const point = rocData.reduce((closest, current) => 
        Math.abs(current.threshold - threshold) < Math.abs(closest.threshold - threshold) ? current : closest
    );
    
    const { tp, fp, tn, fn, tpr, fpr } = point;
    
    const precision = tp / (tp + fp) || 0;
    const recall = tpr; // Same as TPR
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
    
    metricsDisplay.innerHTML = `
        <div class="metric-card">
            <h3>Confusion Matrix</h3>
            <table style="margin: 0 auto;">
                <tr><td></td><td><strong>Predicted 0</strong></td><td><strong>Predicted 1</strong></td></tr>
                <tr><td><strong>Actual 0</strong></td><td>${tn}</td><td>${fp}</td></tr>
                <tr><td><strong>Actual 1</strong></td><td>${fn}</td><td>${tp}</td></tr>
            </table>
        </div>
        <div class="metric-card">
            <h3>Accuracy</h3>
            <p style="font-size: 24px; color: #4CAF50;">${(accuracy * 100).toFixed(2)}%</p>
        </div>
        <div class="metric-card">
            <h3>Precision</h3>
            <p style="font-size: 24px; color: #2196F3;">${(precision * 100).toFixed(2)}%</p>
        </div>
        <div class="metric-card">
            <h3>Recall</h3>
            <p style="font-size: 24px; color: #FF9800;">${(recall * 100).toFixed(2)}%</p>
        </div>
        <div class="metric-card">
            <h3>F1-Score</h3>
            <p style="font-size: 24px; color: #9C27B0;">${(f1 * 100).toFixed(2)}%</p>
        </div>
    `;
}

// Prediction Function
async function predictTestData() {
    const statusDiv = document.getElementById('prediction-status');
    
    statusDiv.innerHTML = '<div class="status info">Generating predictions...</div>';
    
    try {
        const threshold = parseFloat(document.getElementById('threshold-slider').value);
        const testFeatures = processedTestData.features;
        
        // Get probabilities
        const probabilities = await model.predict(testFeatures).data();
        
        // Convert to binary predictions
        const predictions = probabilities.map(prob => prob >= threshold ? 1 : 0);
        
        // Create submission file
        const submissionContent = createSubmissionCSV(processedTestData.passengerIds, predictions);
        const probabilitiesContent = createProbabilitiesCSV(processedTestData.passengerIds, probabilities);
        
        // Download files
        downloadFile('submission.csv', submissionContent);
        downloadFile('probabilities.csv', probabilitiesContent);
        
        statusDiv.innerHTML = `
            <div class="status success">
                Predictions generated!<br>
                Files downloaded: submission.csv, probabilities.csv<br>
                Positive predictions: ${predictions.filter(p => p === 1).length}/${predictions.length}
            </div>
        `;
        
    } catch (error) {
        statusDiv.innerHTML = `<div class="status error">Prediction error: ${error.message}</div>`;
    }
}

// Create submission CSV
function createSubmissionCSV(passengerIds, predictions) {
    let csv = 'PassengerId,Survived\n';
    passengerIds.forEach((id, index) => {
        csv += `${id},${predictions[index]}\n`;
    });
    return csv;
}

// Create probabilities CSV
function createProbabilitiesCSV(passengerIds, probabilities) {
    let csv = 'PassengerId,Probability\n';
    passengerIds.forEach((id, index) => {
        csv += `${id},${probabilities[index].toFixed(4)}\n`;
    });
    return csv;
}

// Download file helper
function downloadFile(filename, content) {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// Model Export Function
async function exportModel() {
    const statusDiv = document.getElementById('prediction-status');
    
    statusDiv.innerHTML = '<div class="status info">Exporting model...</div>';
    
    try {
        await model.save('downloads://titanic-tfjs-model');
        statusDiv.innerHTML = '<div class="status success">Model exported successfully!</div>';
    } catch (error) {
        statusDiv.innerHTML = `<div class="status error">Model export error: ${error.message}</div>`;
    }
}

// Note: To adapt for other datasets, modify:
// 1. Data schema in preprocessDataset() function
// 2. Feature engineering logic
// 3. Model architecture if needed
// 4. Evaluation metrics as required
