import pickle
import pandas as pd
import numpy as np

with open('rf_classifier.pkl', 'rb') as f:
    rf_classifier = pickle.load(f)

with open('user_final_rating.pkl', 'rb') as f:
    user_final_rating = pickle.load(f)

reviews = pd.read_csv('https://drive.google.com/uc?export=download&id=1scp8OUht3ikqshXchS8Juyn_Yz2hUagx', encoding='latin-1')

def recommend_products_hybrid(user_input):
    # Step 1: Get user CF predictions
    user_cf = user_final_rating.loc[user_input]
    user_cf_nonzero = user_cf[user_cf > 0].sort_values(ascending=False)

    # Step 2: Get sentiment predictions using RF classifier
    product_sentiment = {}
    items_to_process = user_cf_nonzero.head(20).index.tolist()

    print(f"\nProcessing {len(items_to_process)} items for sentiment...")

    for item_id in items_to_process:
        item_reviews = reviews[reviews['id'] == item_id]
        
        if len(item_reviews) == 0:
            product_sentiment[item_id] = 2.5  # Default neutral
            continue
        
        # Check if review_vectors column exists
        if 'review_vectors' not in item_reviews.columns:
            # Fallback: use actual ratings if available
            if 'reviews_rating' in item_reviews.columns:
                product_sentiment[item_id] = item_reviews['reviews_rating'].mean()
            else:
                product_sentiment[item_id] = 2.5
            continue
        
        predictions = []
        for idx, review in item_reviews.iterrows():
            if pd.notna(review.get('review_vectors')):
                try:
                    vector = np.array(review['review_vectors']).reshape(1, -1)
                    pred = rf_classifier.predict(vector)[0]
                    predictions.append(pred)
                except Exception as e:
                    continue
        
        if predictions:
            product_sentiment[item_id] = np.mean(predictions)
        else:
            # Fallback to actual rating if available
            if 'reviews_rating' in item_reviews.columns:
                product_sentiment[item_id] = item_reviews['reviews_rating'].mean()
            else:
                product_sentiment[item_id] = 2.5

    print(f"Sentiment scores calculated for {len(product_sentiment)} items")

    # Step 3: Create hybrid scores
    hybrid_scores = []

    if len(user_cf_nonzero) > 0:
        max_cf = user_cf_nonzero.max()
        
        for item_id in user_cf_nonzero.head(20).index:
            cf_rating = user_cf_nonzero[item_id]
            sentiment_pred = product_sentiment.get(item_id, 2.5)
            
            # Normalize CF rating
            cf_norm = cf_rating / max_cf if max_cf > 0 else 0
            
            # Normalize sentiment prediction (assuming 1-5 scale)
            sentiment_norm = (sentiment_pred - 1) / 4.0 if sentiment_pred else 0.5
            
            # Hybrid score: 60% CF + 40% Sentiment
            hybrid_score = 0.6 * cf_norm + 0.4 * sentiment_norm
            
            hybrid_scores.append({
                'id': item_id,
                'cf_rating': float(cf_rating),
                'sentiment_pred': float(sentiment_pred),
                'hybrid_score': float(hybrid_score)
            })

    # Step 4: Create DataFrame
    if len(hybrid_scores) > 0:
        d_hybrid = pd.DataFrame(hybrid_scores)
        d_hybrid = d_hybrid.sort_values('hybrid_score', ascending=False).head(5)
        
        # Merge with names
        d_hybrid = pd.merge(
            d_hybrid,
            reviews[['id', 'name']].drop_duplicates(),
            on='id',
            how='left'
        )
        
        print("\nHybrid Model Recommendations:")
    else:
        print("ERROR: No hybrid scores generated!")
        print("Falling back to CF-only recommendations...")
        
        # Fallback: Use CF recommendations only
        d_hybrid = pd.DataFrame({
            'id': user_cf_nonzero.head(5).index.tolist(),
            'cf_rating': user_cf_nonzero.head(5).values.tolist(),
            'sentiment_pred': [product_sentiment.get(id, 2.5) for id in user_cf_nonzero.head(5).index],
            'hybrid_score': (user_cf_nonzero.head(5).values / user_cf_nonzero.max()).tolist()
        })
        
        d_hybrid = pd.merge(
            d_hybrid,
            reviews[['id', 'name']].drop_duplicates(),
            on='id',
            how='left'
        )
        
    return d_hybrid['name'].tolist()


if __name__ == "__main__":
    user_id = input("Enter your user ID: ")
    print(recommend_products_hybrid(user_id))
