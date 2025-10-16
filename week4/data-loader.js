class DataLoader {
    constructor() {
        this.stocksData = null;
        this.normalizedData = null;
        this.symbols = [];
        this.dates = [];
        this.X_train = null;
        this.y_train = null;
        this.X_test = null;
        this.y_test = null;
        this.testDates = [];
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csv = e.target.result;
                    this.parseCSV(csv);
                    resolve(this.stocksData);
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
        const headers = lines[0].split(',').map(h => h.trim());
        
        console.log('CSV Headers:', headers);
        
        const data = {};
        const symbols = new Set();
        const dates = new Set();

        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;

            const values = line.split(',');
            if (values.length < headers.length) {
                console.warn(`Skipping line ${i}: insufficient columns`);
                continue;
            }

            const row = {};
            headers.forEach((header, index) => {
                row[header] = values[index] ? values[index].trim() : '';
            });

            const symbol = row.Symbol || row.symbol;
            const date = row.Date || row.date;
            
            if (!symbol || !date) {
                console.warn(`Skipping row ${i}: missing symbol or date`);
                continue;
            }

            // Parse numeric values safely
            const open = parseFloat(row.Open || row.open);
            const close = parseFloat(row.Close || row.close);
            const high = parseFloat(row.High || row.high);
            const low = parseFloat(row.Low || row.low);
            const volume = parseFloat(row.Volume || row.volume);

            if (isNaN(open) || isNaN(close)) {
                console.warn(`Skipping row ${i}: invalid numeric data`);
                continue;
            }

            symbols.add(symbol);
            dates.add(date);

            if (!data[symbol]) data[symbol] = {};
            
            data[symbol][date] = {
                Open: open,
                Close: close,
                High: high || open,
                Low: low || open,
                Volume: volume || 0
            };
        }

        this.symbols = Array.from(symbols).sort();
        this.dates = Array.from(dates).sort();
        this.stocksData = data;

        console.log(`Loaded ${this.symbols.length} stocks with ${this.dates.length} trading days`);
        if (this.symbols.length > 0) {
            console.log('Sample data:', this.stocksData[this.symbols[0]][this.dates[0]]);
        }
    }

    normalizeData() {
        if (!this.stocksData) throw new Error('No data loaded');
        
        this.normalizedData = {};
        const minMax = {};

        // Calculate min-max per stock
        this.symbols.forEach(symbol => {
            minMax[symbol] = {
                Open: { min: Infinity, max: -Infinity },
                Close: { min: Infinity, max: -Infinity }
            };

            this.dates.forEach(date => {
                if (this.stocksData[symbol][date]) {
                    const point = this.stocksData[symbol][date];
                    minMax[symbol].Open.min = Math.min(minMax[symbol].Open.min, point.Open);
                    minMax[symbol].Open.max = Math.max(minMax[symbol].Open.max, point.Open);
                    minMax[symbol].Close.min = Math.min(minMax[symbol].Close.min, point.Close);
                    minMax[symbol].Close.max = Math.max(minMax[symbol].Close.max, point.Close);
                }
            });
        });

        // Normalize data
        this.symbols.forEach(symbol => {
            this.normalizedData[symbol] = {};
            this.dates.forEach(date => {
                if (this.stocksData[symbol][date]) {
                    const point = this.stocksData[symbol][date];
                    const openRange = minMax[symbol].Open.max - minMax[symbol].Open.min;
                    const closeRange = minMax[symbol].Close.max - minMax[symbol].Close.min;
                    
                    this.normalizedData[symbol][date] = {
                        Open: openRange > 0 ? (point.Open - minMax[symbol].Open.min) / openRange : 0.5,
                        Close: closeRange > 0 ? (point.Close - minMax[symbol].Close.min) / closeRange : 0.5
                    };
                }
            });
        });

        return this.normalizedData;
    }

    createSequences(sequenceLength = 10, predictionHorizon = 2) {
        if (!this.stocksData) throw new Error('No data loaded');
        if (!this.normalizedData) this.normalizeData();

        const sequences = [];
        const targets = [];
        const validDates = [];

        // Use fewer sequences for stability
        const maxSequences = Math.min(1000, this.dates.length - sequenceLength - predictionHorizon);

        for (let i = sequenceLength; i < Math.min(this.dates.length - predictionHorizon, sequenceLength + maxSequences); i++) {
            const currentDate = this.dates[i];
            const sequenceData = [];
            let validSequence = true;

            // Build sequence
            for (let j = 0; j < sequenceLength; j++) {
                const seqDate = this.dates[i - sequenceLength + j];
                const timeStepData = [];

                this.symbols.forEach(symbol => {
                    if (this.normalizedData[symbol] && this.normalizedData[symbol][seqDate]) {
                        timeStepData.push(
                            this.normalizedData[symbol][seqDate].Open,
                            this.normalizedData[symbol][seqDate].Close
                        );
                    } else {
                        validSequence = false;
                    }
                });

                if (validSequence) sequenceData.push(timeStepData);
            }

            // Create target
            if (validSequence) {
                const target = [];
                const baseClosePrices = [];

                // Get current prices
                this.symbols.forEach(symbol => {
                    baseClosePrices.push(this.stocksData[symbol][currentDate].Close);
                });

                // Create binary targets
                for (let offset = 1; offset <= predictionHorizon; offset++) {
                    const futureDate = this.dates[i + offset];
                    this.symbols.forEach((symbol, idx) => {
                        if (this.stocksData[symbol] && this.stocksData[symbol][futureDate]) {
                            const futureClose = this.stocksData[symbol][futureDate].Close;
                            target.push(futureClose > baseClosePrices[idx] ? 1 : 0);
                        } else {
                            validSequence = false;
                        }
                    });
                }

                if (validSequence) {
                    sequences.push(sequenceData);
                    targets.push(target);
                    validDates.push(currentDate);
                }
            }
        }

        if (sequences.length === 0) {
            throw new Error('No valid sequences created. Check data consistency.');
        }

        // Split data
        const splitIndex = Math.floor(sequences.length * 0.8);
        
        this.X_train = tf.tensor3d(sequences.slice(0, splitIndex));
        this.y_train = tf.tensor2d(targets.slice(0, splitIndex));
        this.X_test = tf.tensor3d(sequences.slice(splitIndex));
        this.y_test = tf.tensor2d(targets.slice(splitIndex));
        this.testDates = validDates.slice(splitIndex);

        console.log(`Created ${sequences.length} sequences`);
        console.log(`Training shape: ${this.X_train.shape}, Test shape: ${this.X_test.shape}`);
        
        return {
            X_train: this.X_train,
            y_train: this.y_train,
            X_test: this.X_test,
            y_test: this.y_test,
            symbols: this.symbols,
            testDates: this.testDates
        };
    }

    dispose() {
        if (this.X_train) this.X_train.dispose();
        if (this.y_train) this.y_train.dispose();
        if (this.X_test) this.X_test.dispose();
        if (this.y_test) this.y_test.dispose();
    }
}

export default DataLoader;
