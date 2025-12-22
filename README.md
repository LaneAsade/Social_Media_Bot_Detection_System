# Social Media Bot Detection

This project implements a machine learning pipeline to classify social media tweets as bot-generated or human-generated using tweet-level linguistic and structural features.

## Problem Statement
Automated bot accounts pose challenges on social media platforms by spreading spam and misinformation. The goal of this project is to detect bot-generated tweets using machine learning techniques without relying on temporal metadata.

## Dataset
- Source: Public bot detection dataset
- Each row represents a single tweet
- Labels: 
  - 0 → Human
  - 1 → Bot

## Approach
1. Performed exploratory data analysis to understand class distribution and feature variance
2. Cleaned tweet text and removed noise such as URLs, mentions, and hashtags
3. Engineered tweet-level features including:
   - Tweet length
   - Hashtag count
   - Follower-based metrics
   - Sentiment polarity
4. Trained an XGBoost classifier and evaluated performance using ROC-AUC and F1-score

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- TextBlob
- Matplotlib

## Results
The model demonstrates the effectiveness of combining text-based features with numerical metadata for bot detection.
