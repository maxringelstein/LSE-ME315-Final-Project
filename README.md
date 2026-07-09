# Machine Learning Project: Predicting Track Popularity & Genre from Spotify Audio Features

A machine learning project analysing 176,773 unique Spotify tracks to answer two questions: how well can a track's popularity be predicted from its audio features alone, and how well can its genre be identified from the same features? Built for the ME315 Machine Learning in Practice course at LSE Summer School.

## Overview

The project compares two families of models across two supervised learning tasks using a consistent train/test and cross-validation protocol:

**Task 1 — Regression:** Predicting track `popularity` (0–100) using OLS, Ridge, Lasso, Random Forest, and XGBoost.

**Task 2 — Classification:** Predicting track `genre` (6 classes: Classical, Hip-Hop, Electronic, Jazz, Country, Comedy) using Logistic Regression, LDA, QDA, and Random Forest.

An initial classification target — musical mode (Major/Minor) — was tested and abandoned after EDA showed it was almost entirely unrelated to the available audio features (max correlation of 0.07). Genre was chosen instead after eta-squared analysis confirmed audio features carry strong discriminative signal for it.

## Key Findings

- **Regression:** Linear models plateau at R² ≈ 0.20. Non-linear ensembles (Random Forest, XGBoost) both converge at R² ≈ 0.34, revealing a genuine ceiling in the feature set rather than a linear-model limitation — popularity is substantially driven by factors (recency, artist prominence, marketing, playlist placement) not captured by audio features alone.
- **Classification:** Random Forest achieves 82.4% test accuracy against a 17.9% majority-class baseline. Misclassifications are musically coherent — Jazz is most often confused with Electronic, Classical, and Country, genres that genuinely overlap acoustically.

## Data

Source: [Ultimate Spotify Tracks DB](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db) (Kaggle, zaheenhamidani), 232,725 raw records.

- **Regression subset:** 176,773 unique tracks after deduplication on `track_id`
- **Classification subset:** 54,070 tracks across 6 acoustically distinct, evenly-distributed genres

**Features used (12):** `acousticness`, `danceability`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`, `duration_min`, `mode_enc`, `is_4_4`

## Repository Structure

```
├── SpotifyFeatures.csv        # Raw dataset (not included — see Data Access below)
├── notebook.ipynb             # Full analysis: EDA, preprocessing, modelling, evaluation
├── ME315_Final_Project.pdf    # Final written report
└── README.md
```

## Data Access

`SpotifyFeatures.csv` is not included in this repository due to size. Download it directly from [Kaggle](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db) and place it in the repository root before running the notebook.

## Running the Notebook

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
jupyter notebook notebook.ipynb
```

## Methods

All continuous features are standardised (fit on training data only, via pipeline) for linear and discriminant models; tree-based models use raw features. Regularisation parameters for linear models (Ridge, Lasso, Logistic Regression) are tuned via 5-fold cross-validation. A fixed random seed ensures reproducibility across runs.

## Author

Max Ringelstein — July 2026

## License

For academic use as part of ME315 coursework at LSE.
