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
        
        console.log('CSV Headers:', headers); // Debug log
        
        const data = {};
        const symbols = new Set();
        const dates = new Set();

        // Parse all rows
        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;

            // Handle CSV with potential quotes and commas within values
            const values = this.parseCSVLine(line);
            if (values.length !== headers.length) {
                console.warn(`Skipping line ${i}: expected ${headers.length} columns, got ${values.length}`);
                continue;
            }

            const row = {};
            headers.forEach((header, index) => {
                row[header] = values[index].trim();
            });

            const symbol = row.Symbol || row.symbol;
            const date = row.Date || row.date;
            
            if (!symbol || !date) {
                console.warn(`Skipping row ${i}: missing symbol or date`, row);
                continue;
            }

            symbols.add(symbol);
            dates.add(date);

            if (!data[symbol]) data[symbol] = {};
            
            // Parse numeric values - handle your specific column names
            data[symbol][date] = {
                Open: parseFloat(row.Open || row.open || 0),
                Close: parseFloat(row.Close || row.close || 0),
                High: parseFloat(row.High || row.high || 0),
                Low: parseFloat(row.Low || row.low || 0),
                Volume: parseFloat(row.Volume || row.volume || 0),
                AdjClose: parseFloat(row['Adj Close'] || row['Adj Close'] || row.Close || row.close || 0)
            };
        }

        this.symbols = Array.from(symbols).sort();
        this.dates = Array.from(dates).sort();
        this.stocksData = data;

        console.log(`Loaded ${this.symbols.length} stocks with ${this.dates.length} trading days`);
        console.log('Sample data:', this.stocksData[this.symbols[0]][this.dates[0]]);
    }

    parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                result.push(current);
                current = '';
            } else {
                current += char;
            }
        }
        
        result.push(current);
        return result;
    }

    normalizeData() {
        if (!this.stocksData) throw new Error('No data loaded');
        
        this.normalizedData = {};
        const minMax = {};

        // Calculate min-max per stock for Open and Close
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

    createSequences(sequenceLength = 12, predictionHorizon = 3) {
        if (!this.stocksData) throw new Error('No data loaded');
        if (!this.normalizedData) this.normalizeData();

        const sequences = [];
        const targets = [];
        const validDates = [];

        // Create aligned data matrix
        for (let i = sequenceLength; i < this.dates.length - predictionHorizon; i++) {
            const currentDate = this.dates[i];
            const sequenceData = [];
            let validSequence = true;

            // Get sequence for all symbols
            for (let j = sequenceLength - 1; j >= 0; j--) {
                const seqDate = this.dates[i - j];
                const timeStepData = [];

                this.symbols.forEach(symbol => {
                    if (this.normalizedData[symbol] && this.normalizedData[symbol][seqDate]) {
                        timeStepData.push(
                            this.normalizedData[symbol][seqDate].Open,
                            this.normalizedData[symbol][seqDate].Close
                        );
                    } else {
                        console.warn(`Missing data for ${symbol} on ${seqDate}`);
                        validSequence = false;
                    }
                });

                if (validSequence) sequenceData.push(timeStepData);
            }

            // Create target labels
            if (validSequence) {
                const target = [];
                const baseClosePrices = [];

                // Get base close prices (current date)
                this.symbols.forEach(symbol => {
                    baseClosePrices.push(this.stocksData[symbol][currentDate].Close);
                });

                // Calculate binary labels for prediction horizon
                for (let offset = 1; offset <= predictionHorizon; offset++) {
                    const futureDate = this.dates[i + offset];
                    this.symbols.forEach((symbol, idx) => {
                        if (this.stocksData[symbol] && this.stocksData[symbol][futureDate]) {
                            const futureClose = this.stocksData[symbol][futureDate].Close;
                            target.push(futureClose > baseClosePrices[idx] ? 1 : 0);
                        } else {
                            console.warn(`Missing future data for ${symbol} on ${futureDate}`);
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
            throw new Error('No valid sequences created. Check your data format and ensure all stocks have data for all dates.');
        }

        // Split into train/test (80/20 chronological split)
        const splitIndex = Math.floor(sequences.length * 0.8);
        
        this.X_train = tf.tensor3d(sequences.slice(0, splitIndex));
        this.y_train = tf.tensor2d(targets.slice(0, splitIndex));
        this.X_test = tf.tensor3d(sequences.slice(splitIndex));
        this.y_test = tf.tensor2d(targets.slice(splitIndex));
        this.testDates = validDates.slice(splitIndex);

        console.log(`Created ${sequences.length} sequences`);
        console.log(`Training: ${this.X_train.shape} sequences, Test: ${this.X_test.shape} sequences`);
        console.log(`Features per timestep: ${this.X_train.shape[2]}`);
        
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
