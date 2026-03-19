# NHL Game Prediction Model

A machine learning pipeline that predicts NHL game outcomes using historical data and advanced feature engineering techniques.

## 📊 Performance Metrics
- **Accuracy**: 56.7%
- **Precision**: 57.0%
- **Data Size**: 12,000+ games across 5 seasons (2021-2025)
- **Improvement**: 12% above baseline random prediction

## 🚀 Features

### Data Collection
- Automated web scraping from Hockey-Reference.com
- Handles complex multi-level HTML tables
- Processes team relocations (Arizona → Utah)
- Collects 32 NHL teams across multiple seasons

### Feature Engineering
- **40+ predictive features** including:
  - Rolling averages (goals, shots, penalties, advanced stats)
  - Opponent strength metrics
  - Efficiency ratios (shooting %, save %, power play conversion)
  - Home/away performance differentials
  - Recent form indicators

### Model Architecture
- Random Forest classifier with optimized hyperparameters
- Binary classification (Win/Loss)
- Cross-validation for model selection
- Feature importance analysis

## 🛠️ Tech Stack
- **Python**: Core programming language
- **Scikit-learn**: Machine learning framework
- **Pandas**: Data manipulation and analysis
- **BeautifulSoup**: Web scraping
- **Requests**: HTTP requests for data collection

## 📁 Project Structure
```
├── scraper.py              # Web scraping pipeline
├── prediction.ipynb       # Model training and evaluation
├── nhl_matches_2021_2025.csv  # Processed dataset
└── README.md
```

## 🔧 Installation & Usage

### Prerequisites
```bash
pip install pandas scikit-learn beautifulsoup4 requests numpy
```

### Running the Scraper
```python
python scraper.py
```
This will:
- Scrape NHL standings and team URLs
- Collect game logs for all teams
- Process and clean the data
- Export to CSV format

### Training the Model
Open `prediction.ipynb` and run all cells to:
- Load and preprocess data
- Engineer features
- Train multiple model configurations
- Evaluate performance and feature importance

## 📈 Model Performance

### Key Insights
- **Most Important Features**: Opponent strength, recent form, efficiency metrics
- **Home Advantage**: Significant predictor in model
- **Rolling Windows**: 3-game averages provide optimal signal-to-noise ratio

### Feature Importance (Top 5)
1. Opponent strength differential
2. Recent win percentage (last 10 games)
3. Shot efficiency trends
4. Home/away status
5. Power play conversion rate

## 🎯 Future Improvements
- [ ] Add player injury data
- [ ] Incorporate betting odds for calibration
- [ ] Real-time prediction API
- [ ] Expand to playoff predictions
- [ ] Add confidence intervals

## 📝 Data Sources
- [Hockey-Reference.com](https://www.hockey-reference.com/) - Historical game data and statistics
