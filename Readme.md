
# Anime Recommendation System

This project builds a machine learning-based anime recommendation system using two main approaches: Content-Based Filtering and Collaborative Filtering (SVD). The system leverages anime metadata and user rating data from MyAnimeList to generate personalized recommendations.

## Project Description

The anime industry produces thousands of titles with diverse genres and storytelling styles, making it difficult for users to discover content that matches their preferences. This project addresses that issue by implementing a recommendation system that uses both genre similarity and user behavior to provide high-quality suggestions.

## Features

- Content-Based Filtering using TF-IDF and Cosine Similarity
- Collaborative Filtering using Singular Value Decomposition (SVD)
- Top-N Recommendations for anime or users
- Data preprocessing pipeline for cleaning and merging datasets
- Evaluation metrics: RMSE, MAE, Precision@K, Recall@K, NDCG@K

## Dataset

Source: [Kaggle - Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

- anime.csv: Metadata including title, genre, type, episodes, rating, and member count
- rating.csv: User ratings (-1 means no rating, 1–10 are valid ratings)

## Technologies Used

- Python 3.x
- Pandas, Numpy, Scikit-learn
- Surprise (for SVD)
- TF-IDF Vectorizer, Cosine Similarity
- Jupyter Notebook
- Matplotlib / Seaborn (for visualization)

## Installation

1. Clone the repository:
   git clone https://github.com/yourusername/anime-recommendation-system.git

2. Navigate into the directory:
   cd anime-recommendation-system

3. Install the required packages:
   pip install -r requirements.txt

If requirements.txt is not available, install manually:
   pip install pandas numpy scikit-learn scikit-surprise matplotlib seaborn

4. Run the Jupyter notebooks located in the `notebooks/` directory.

## Usage

- Run `content_based_filtering.ipynb` to generate genre-based recommendations.
- Run `collaborative_filtering_svd.ipynb` to get personalized user recommendations.
- Change parameters such as `top_n`, `user_id`, or `anime_title` to explore different outcomes.

## Evaluation

- Content-Based Filtering: Precision@10 = 1.0, Recall@10 ≈ 0.0075, NDCG@10 = 1.0
- Collaborative Filtering (SVD): RMSE ≈ 2.15, MAE ≈ 1.50 (based on 5-fold cross-validation)

## Example Output

Content-Based (input: Naruto):
1. Boruto: Naruto the Movie
2. Naruto Shippuden Movie 4
3. Naruto: Road to Ninja

Collaborative Filtering (input: user_id = 5):
1. Hotaru no Haka (predicted rating: 9.05)
2. Ginga Eiyuu Densetsu (8.45)
3. Rose of Versailles (8.14)

## Acknowledgments

- Kaggle Dataset: Anime Recommendations Database
- Surprise Library
- MyAnimeList

## License

This project is licensed under the MIT License.
