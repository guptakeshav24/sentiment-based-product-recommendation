# Sentiment-Based Product Recommendation System

A hybrid recommendation system that combines collaborative filtering with sentiment analysis to provide personalized product recommendations. The system analyzes user preferences and product review sentiments to recommend products that users are likely to enjoy.

## Features

- **Hybrid Recommendation Engine**: Combines user-based collaborative filtering (40%) with sentiment analysis (60%)
- **Sentiment Classification**: Uses machine learning models to classify product reviews as positive or negative
- **Multiple ML Models**: Evaluated Logistic Regression, XGBoost, and Random Forest for sentiment analysis
- **Web Interface**: User-friendly Flask-based web application for getting recommendations
- **RESTful API**: JSON API endpoint for programmatic access to recommendations

## Architecture

The system consists of two main components:

1. **Sentiment Analysis Model**: Classifies product reviews using Word2Vec embeddings and machine learning classifiers
2. **Recommendation Engine**: 
   - User-based collaborative filtering to find similar users
   - Sentiment analysis to filter out products with negative reviews
   - Hybrid scoring to combine both approaches

### Model Selection

- **Sentiment Model**: Logistic Regression (selected for better recall on negative reviews - 0.76)
- **Recommendation Method**: User-based Collaborative Filtering (RMSE: 2.15 vs 3.57 for item-based)

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd /path/to/Capstone
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r Deployment/requirements.txt
   ```

4. **Download additional dependencies** (if not already installed):
   ```bash
   pip install gensim xgboost nltk
   ```

5. **Download NLTK data** (run once):
   ```python
   import nltk
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Project Structure

```
Capstone/
├── Deployment/                         # Deployment files
│   ├── app.py                          # Flask web application
│   ├── model.py                        # Recommendation model and prediction logic
│   ├── index.html                      # Web interface
│   ├── requirements.txt                # Python dependencies
│   ├── model.pkl                       # Trained sentiment classification model
│   ├── user_final_rating.pkl           # User-based collaborative filtering matrix
│   └── df_final.csv                    # Dataset with review vectors
├── SentimentBasedProductRecommendation.ipynb  # Jupyter notebook with full analysis
├── Sentiment_Based_Product_Recommendation_Report.docx  # Project report
├── Sentiment_Based_Product_Recommendation_Report.pdf   # Project report (PDF)
├── README.md                           # This file
└── venv/                               # Virtual environment (not included in git)
```

## Usage

### Running the Web Application

**Navigate to Deployment folder and run:**
```bash
cd Deployment
source ../venv/bin/activate  # Activate virtual environment
python app.py
```

The application will start on `http://localhost:50000`

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:50000`
2. Enter a username (e.g., `00dog3`, `00sab00`, `01impala`, `0325home`)
3. Click "Get Recommendations"
4. View the top 5 recommended products

### Using the API

**Get recommendations for a user:**
```bash
curl http://localhost:50000/recommendations/00dog3
```

**Response:**
```json
{
  "user_id": "00dog3",
  "recommendations": [
    {"product_name": "Product Name 1", "rank": 1},
    {"product_name": "Product Name 2", "rank": 2},
    ...
  ],
  "total_recommendations": 5
}
```

### Using the Model Directly

```python
import sys
sys.path.append('Deployment')
import model

# Get recommendations for a user
recommendations = model.recommend_products_hybrid('00dog3')
print(recommendations)
```

## Model Details

### Sentiment Analysis Pipeline

1. **Text Preprocessing**:
   - Lowercasing
   - Punctuation removal
   - Number removal
   - Stopword removal
   - Lemmatization

2. **Feature Extraction**:
   - Word2Vec embeddings (100 dimensions)
   - Document vectors created by averaging word embeddings

3. **Classification Models**:
   - **Logistic Regression**: Selected model (Recall: 0.76 for negative class)
   - **XGBoost**: Accuracy: 90.7%, ROC-AUC: 0.85
   - **Random Forest**: Accuracy: 90.5%, ROC-AUC: 0.85

### Recommendation Algorithm

1. **User-Based Collaborative Filtering**:
   - Computes user-user similarity using cosine similarity
   - Predicts ratings based on similar users' preferences
   - Filters out items already rated by the user

2. **Sentiment Filtering**:
   - Analyzes reviews for top CF-recommended products
   - Predicts sentiment using trained classifier
   - Filters out products with negative sentiment

3. **Hybrid Scoring**:
   - Normalizes CF ratings and sentiment scores
   - Combines: `hybrid_score = 0.4 * CF_score + 0.6 * sentiment_score`
   - Returns top 5 products

## Dataset

The system uses a product review dataset containing:
- User reviews and ratings
- Product information
- Review vectors (Word2Vec embeddings)
- User sentiment labels (Positive/Negative)

The processed dataset (`df_final.csv`) is included in the `Deployment/` folder.

## Technologies Used

- **Python**: Core programming language
- **Flask**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning models
- **Gensim**: Word2Vec implementation
- **XGBoost**: Gradient boosting classifier
- **NLTK**: Natural language processing
- **Matplotlib/Seaborn**: Data visualization (in notebook)

## Performance Metrics

### Sentiment Classification
- **Logistic Regression**: 
  - Accuracy: 71.2%
  - Recall (Negative): 0.76
  - ROC-AUC: 0.81

### Recommendation System
- **User-Based CF**: RMSE = 2.15
- **Item-Based CF**: RMSE = 3.57

## Jupyter Notebook

The `SentimentBasedProductRecommendation.ipynb` notebook contains:
- Complete data analysis and exploration
- Text preprocessing pipeline
- Model training and evaluation
- Recommendation system implementation
- Model comparison and selection
- Detailed comments explaining each step

## Notes

- The application requires the pre-trained models (`model.pkl` and `user_final_rating.pkl`) in the `Deployment/` folder
- Dataset file `df_final.csv` must be present in the `Deployment/` folder
- Ensure sufficient memory for loading large pickle files and CSV dataset
- User IDs must exist in the training data for recommendations
- Sample user IDs for testing: `00dog3`, `00sab00`, `01impala`, `0325home`



## Author

Keshav Gupta

