# ETF Portfolio Weight Optimization

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

![Optuna](https://img.shields.io/badge/Optuna-3.x-green.svg)

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
## Overview

This project develops a data-driven approach to optimize Exchange-Traded Fund (ETF) portfolio weights using machine learning, sentiment analysis, and macroeconomic indicators. Focusing on five ETF sectors—Financials, Real Estate, Technology, Energy, and Healthcare—we implement Long Short-Term Memory (LSTM), Transformer, and Recurrent Neural Network (RNN) models to predict daily ETF returns and optimize portfolio weights for risk-averse, risk-neutral, and risk-loving investors. The models incorporate sentiment scores from financial news (processed with FinBERT) and macroeconomic indicators from Federal Reserve Economic Data (FRED). The goal is to maximize the Sharpe Ratio, benchmarked against an equal-weight portfolio (EQW).

Key highlights:

- **LSTM Model 2 (Sentiment + Volatility)**: Best for risk-averse investors with a Sharpe Ratio of 0.74, low volatility (1.19%), and an Alpha of 0.70%.
- **Transformer Model 2**: Suitable for risk-neutral investors with a balanced return (4.65%) and Sharpe Ratio (0.69).
- **RNN Model 2**: A viable alternative for risk-averse investors with low volatility (1.40%) and diversified sector weights.

The project covers data from 2014 to 2024, sourced from Yahoo Finance, Bloomberg, Reuters, Financial Times, and FRED.

## Table of Contents

- Features
- Installation
- Usage
- Data Sources
- Model Architectures
- Results
- Limitations
- Future Improvements
- Contributing
- License
- References

## Features

- **Sentiment Analysis**: Uses FinBERT to derive sentiment scores from financial news, achieving 81-83% confidence across sectors.
- **Macroeconomic Integration**: Incorporates indicators like CPI, Unemployment Rate, GDP, VIX, and Treasury yields.
- **Machine Learning Models**:
  - **LSTM**: Stacked layers with dropout, optimized for time-series prediction.
  - **Transformer**: Attention-based model for high-alpha strategies.
  - **RNN**: Simple architecture for baseline comparison.
- **Portfolio Optimization**: Maximizes Sharpe Ratio using predicted returns and historical covariance (Ledoit-Wolf shrinkage).
- **Visualization**: Includes bar charts for sector weights, performance metrics, and cumulative returns.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/etf-portfolio-optimization.git
   cd etf-portfolio-optimization
   ```

2. **Install Dependencies**: Ensure Python 3.8+ is installed. Then, install required packages:

   ```bash
   pip install -r requirements.txt
   ```

   Sample `requirements.txt`:

   ```
   pandas>=1.5.0
   numpy>=1.23.0
   tensorflow>=2.10.0
   optuna>=3.0.0
   scikit-learn>=1.2.0
   matplotlib>=3.5.0
   scipy>=1.9.0
   beautifulsoup4>=4.11.0
   yfinance>=0.2.0
   ```

## Usage

1. **Run Data Preparation**: Execute the data loading and cleaning script:

   ```bash
   python scripts/data_preparation.py
   ```

   This script loads ETF prices, sentiment scores, and macro data, standardizes dates, and merges them into a combined DataFrame.

2. **Train Models**: Train LSTM, Transformer, and RNN models with hyperparameter tuning:

   ```bash
   python scripts/train_models.py
   ```

   This script uses Optuna to optimize hyperparameters and saves trained models (`model1_*.h5`, `model2_*.h5`, `model3_*.h5`).

3. **Generate Predictions and Optimize Portfolio**: Run predictions and portfolio optimization:

   ```bash
   python scripts/predict_optimize.py
   ```

   Outputs include predicted returns, optimized weights, and performance metrics (Sharpe Ratio, Volatility, Alpha).

4. **Visualize Results**: Generate visualizations for sector weights and performance metrics:

   ```bash
   python scripts/visualize_results.py
   ```

   Produces bar charts, cumulative return plots, and more.

## Data Sources

- **ETF Price Data**: Yahoo Finance (2014-2024) for XLK (Technology), XLV (Healthcare), XLE (Energy), VNQ (Real Estate), and XLF (Financials).
- **Sentiment Scores**: Derived from Bloomberg, Reuters, and Financial Times using BeautifulSoup and FinBERT.
- **Macroeconomic Indicators**: FRED, including CPI, Unemployment Rate, GDP, VIX, WTI Crude Oil Prices, and Treasury yields.

## Model Architectures

- **LSTM**:
  - Stacked LSTM layers with dropout (0.2-0.5).
  - Hyperparameters: units (64-256), learning rate (1e-4 to 1e-2), batch size (16-64).
  - Optimized with Optuna for minimal MSE.
- **Transformer**:
  - Multi-head attention mechanism with 2-8 heads, feed-forward dimensions (64-256).
  - Trained for up to 150 epochs with early stopping.
- **RNN**:
  - SimpleRNN layers with higher dropout (0.3-0.6) to mitigate overfitting.
  - Suitable for baseline comparisons but struggles with long-term dependencies.

## Results

| Model | Return (%) | Volatility (%) | Sharpe Ratio | VaR (5%) | ES (5%) | Alpha (%) |
| --- | --- | --- | --- | --- | --- | --- |
| **LSTM Model 2** | 4.42 | 1.19 | 0.74 | \-0.11 | \-0.14 | 0.70 |
| **Transformer Model 2** | 4.65 | 1.61 | 0.69 | \-0.16 | \-0.21 | 0.86 |
| **RNN Model 2** | 4.04 | 1.40 | 0.36 | \-0.12 | \-0.18 | 0.23 |
| **Benchmark (EQW)** | \- | 15.24 | \- | \- | \-2.23 | \- |

- **LSTM Model 2**: Best for risk-averse investors, with $7,000 additional annual return and $140,500 less risk on a $1M investment compared to EQW.
- **Transformer Model 2**: Balanced for risk-neutral investors, offering $8,600 additional return and $136,300 less risk.
- **RNN Model 2**: Diversified weights (45.81% Financials, 43.44% Energy), reducing sector-specific risk.

## Limitations

- **Short-Term Focus**: Models use 20-day sequences, potentially missing longer economic cycles.
- **Computational Intensity**: Transformers and FinBERT require significant resources (e.g., 45 hours for training on CPU).
- **Macro Data Noise**: Interpolated daily macro indicators may reduce Model 3 performance.
- **RNN Constraints**: Vanishing gradient issues limit long-term dependency capture.

## Future Improvements

- **Longer Time Horizons**: Experiment with 60-day sequences or hierarchical embeddings to capture extended trends.
- **Hybrid Models**: Combine LSTM for short-term predictions with Transformer attention for long-term trends.
- **Advanced Macro Engineering**: Use PCA or factor analysis to reduce noise in macroeconomic indicators.
- **Market Regime Testing**: Evaluate model robustness across bull and bear markets (e.g., 2020 COVID-19 downturn).
- **Hardware Optimization**: Use GPUs (e.g., AWS EC2 g4dn.xlarge) to reduce training time.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Report issues or suggest improvements via the Issues tab.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## References

- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv preprint arXiv:1908.10063*. Link.
- Federal Reserve Bank of St. Louis (2023). Federal Reserve Economic Data (FRED). Link.
- Katsuya, A., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *Proceedings of KDD '19*, pp. 2623-2631. Link.
- Richardson, L. (2023). Beautiful Soup Documentation. Link.
- Yahoo Finance (2023). Historical ETF Data. Link.
