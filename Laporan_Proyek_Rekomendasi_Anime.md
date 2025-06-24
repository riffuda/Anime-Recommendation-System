# Final Project Report – Applied Machine Learning Project: Building an Anime Recommendation System

## Project Overview

The anime industry continues to grow rapidly, producing thousands of titles across a wide range of genres, types, and storytelling styles. With so many options, viewers face the challenge of finding anime that aligns with their preferences. Recommendation systems play a crucial role in helping users navigate this vast catalog efficiently.

This project aims to build a data-driven anime recommendation system using two primary approaches: **Content-Based Filtering** and **Collaborative Filtering**. By utilizing datasets from MyAnimeList (`anime.csv` and `rating.csv`), the system analyzes both content features (e.g., genre, type) and user behavior patterns (ratings) to deliver personalized recommendations.

The relevance of developing such a system has grown alongside the rise of streaming platforms and personalized content consumption. Recent studies indicate that effective recommendation systems can increase user engagement by 30–50% (Zhang et al., 2021) and help platforms retain user loyalty over the long term (Sharma et al., 2023).

Moreover, combining Content-Based and Collaborative Filtering approaches enhances recommendation quality by capturing both explicit user preferences and item similarities (Fan & Zhang, 2022).

**Importance of This Project:**

* Helps users discover anime that match their personal tastes.
* Increases user engagement and retention on anime/streaming platforms.
* Encourages exploration of lesser-known titles that might otherwise be overlooked.
* Provides practical insights into implementing recommendation system techniques.

**Related Research:**

* Zhang, Y., et al. (2021). *Personalized recommendation in streaming platforms: challenges and opportunities*. ACM Computing Surveys, 54(5).
* Sharma, V., et al. (2023). *Hybrid Recommender Systems: Recent Advances and Future Directions*. Information Processing & Management, 60(1).
* Fan, W., & Zhang, Y. (2022). *A survey on deep learning based recommender systems*. IEEE Transactions on Knowledge and Data Engineering, 34(2), 828-847.

## Business Understanding

### Problem Statements

* How to recommend anime based on genres or content preferred by users?
* How to use past user interaction data (ratings) to suggest new anime that users are likely to enjoy?

### Goals

* Build a genre-based Content-Based Filtering recommendation system.
* Build a user rating-based Collaborative Filtering recommendation system.
* Generate Top-N recommendations for each user or anime.

### Solution Approach

* **Content-Based Filtering**: Uses genre similarity via TF-IDF and Cosine Similarity.
* **Collaborative Filtering**: Utilizes user interaction (ratings) through the SVD algorithm (Singular Value Decomposition).

## Data Understanding

Dataset used:

* [Anime Recommendation Database – Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database/data)

Two files:

1. `anime.csv`: anime ID, name, genre, type, episode count, rating, and member count.
2. `rating.csv`: user ID, anime ID, and rating (from 1–10; -1 indicates no rating).

**Dataset Summary:**

* Unique users: 7,580
* Unique rated anime: 7,773
* Total ratings: 877,802
* Unique genres: 43 (top: Comedy, Action, Adventure, Fantasy, Sci-Fi)

## Data Preparation

Key preprocessing steps:

* Removed rows with `rating = -1`.
* Dropped duplicate anime entries using `anime_id`.
* Handled missing values (excluded anime with missing genres).
* Merged `anime_df` and `rating_df` to create a unified dataframe.
* Transformed `genre` using TF-IDF for Content-Based Filtering.
* Encoded `user_id` and `anime_id` for Collaborative Filtering using `surprise`.

## Modeling and Results

### 1. Content-Based Filtering

**Technique:**

* Cosine similarity based on TF-IDF matrix of anime genres.
* Input: anime title → Output: Top-N similar anime.

**Results for 'Naruto':**
All recommendations were highly relevant (movies/sequels from the Naruto franchise), showing high genre similarity.

**Insights:**

* Strength: High precision in recommending similar content.
* Weakness: Lacks personalization; recommendations are often too narrow (e.g., sequels only).

### 2. Collaborative Filtering (SVD)

**Technique:**

* Implemented SVD from the `surprise` library.
* Trained on full rating dataset.

**Sample Output (user\_id = 5):**
Recommended anime with high predicted ratings (e.g., *Hotaru no Haka*, *Ginga Eiyuu Densetsu*, *Rose of Versailles*), reflecting diverse and personalized suggestions.

**Insights:**

* Strength: Personalization based on user behavior.
* Weakness: Cold start problem (requires sufficient rating history).

## Evaluation

### Content-Based Filtering

**Qualitative:**

* Excellent genre consistency (e.g., Naruto → other Naruto franchise titles).

**Quantitative Metrics:**

* Precision\@10: **1.0000**
* Recall\@10: **0.0075**
* NDCG\@10: **1.0000**

### Collaborative Filtering (SVD)

**Metrics:**

* RMSE: **2.1485**
* MAE: **1.5036**
* Evaluation via 5-fold cross-validation.

## Conclusion

1. **Content-Based Filtering** (TF-IDF + cosine similarity):

   * Highly accurate for genre-based recommendations.
   * Perfect Precision\@10 and NDCG\@10, but low recall due to narrow content similarity.

2. **Collaborative Filtering (SVD)**:

   * Effective in capturing user preferences.
   * Stable evaluation metrics (RMSE \~2.15, MAE \~1.50).
   * Recommended varied and potentially lesser-known anime based on user behavior.

Both approaches complement each other: CBF ensures content relevance, while CF introduces personalization and diversity.
