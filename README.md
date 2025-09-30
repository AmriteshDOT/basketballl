
## Game Prediction

- The project focusses on predicting outcomes of NCAA basketball games using machine learning by estimating the win probability of a team based on historical game and team statistics.




## Pipeline Overview
#### Built a two-stage model:
- XGBoost for predicting score margin
- Spline calibration to convert margins into win probabilities

#### Designed features using:
- Elo ratings
- Seed differentials
- Team-specific season averages from historical data
- Used Leave-One-Season-Out cross-validation to generalize across different seasons.
#### Results
- Achieved a Brier score of 0.1724.

## Tech Stack
- Python, XGBoost, pandas, numpy, scikit-learn