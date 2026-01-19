from flask import Flask, jsonify, send_from_directory
import model
import os

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/recommendations/<user_id>', methods=['GET'])
def get_recommendations(user_id):
    """Get recommendations for a user by calling model.py function"""
    try:
        # Call the function from model.py
        product_names = model.recommend_products_hybrid(user_id)
        
        # Format response
        recommendations = [
            {'product_name': name, 'rank': idx + 1}
            for idx, name in enumerate(product_names)
        ]
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations)
        })
    
    except KeyError:
        return jsonify({
            'error': f'User "{user_id}" not found',
            'available_users': list(model.user_final_rating.index[:10])
        }), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=50000)
