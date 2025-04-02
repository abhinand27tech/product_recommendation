from flask import Flask, request, jsonify
from flask_cors import CORS
from ml_services import RecommendationSystem
import os
import threading
import atexit
import time
import pandas as pd

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Global variables to track initialization state
initialization_complete = False
initialization_error = None
initialization_progress = "Starting initialization..."
initialization_lock = threading.Lock()
init_thread = None
start_time = None

# Initialize recommendation system
recommendation_system = RecommendationSystem()

def update_progress(message):
    global initialization_progress
    initialization_progress = message
    print(f"Progress: {message}")

def initialize_system():
    global initialization_complete, initialization_error, initialization_progress, recommendation_system, start_time
    
    # Use a lock to prevent multiple initializations
    if initialization_lock.locked():
        print("Initialization already in progress")
        return
        
    with initialization_lock:
        if initialization_complete:
            print("System already initialized")
            return
            
        try:
            start_time = time.time()
            update_progress("Checking data files...")
            
            # Define data paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            detail_path = os.path.join(current_dir, 'data', 'Detail_comb.csv')
            header_path = os.path.join(current_dir, 'data', 'Header_comb.csv')

            # Check if data files exist
            if not os.path.exists(detail_path):
                raise FileNotFoundError(f"Detail file not found at {detail_path}")
            if not os.path.exists(header_path):
                raise FileNotFoundError(f"Header file not found at {header_path}")

            update_progress("Loading header file...")
            print(f"\nLooking for data files in: {os.path.join(current_dir, 'data')}")
            print(f"Detail file path: {detail_path}")
            print(f"Header file path: {header_path}")

            # Load data
            data_loaded = recommendation_system.load_data(
                detail_path, 
                header_path,
                progress_callback=update_progress
            )
            
            if not data_loaded:
                raise Exception("Failed to load data")
            
            elapsed_time = time.time() - start_time
            initialization_complete = True
            update_progress(f"Initialization complete! (Took {elapsed_time:.1f} seconds)")
            print("Data loaded successfully")
            
        except Exception as e:
            initialization_error = str(e)
            initialization_progress = f"Error: {str(e)}"
            print(f"Error during initialization: {str(e)}")

def cleanup_on_exit():
    """Cleanup function to run when the server shuts down"""
    global init_thread
    if init_thread and init_thread.is_alive():
        print("Shutting down initialization thread...")
        init_thread.join(timeout=5)  # Wait up to 5 seconds for the thread to finish

atexit.register(cleanup_on_exit)

@app.route('/api/status')
def get_status():
    """Get the current initialization status"""
    global start_time
    elapsed = time.time() - start_time if start_time else 0
    return jsonify({
        'initialized': initialization_complete,
        'error': initialization_error,
        'progress': initialization_progress,
        'elapsed_seconds': int(elapsed)
    })

@app.route('/api/godowns', methods=['GET'])
def get_godowns():
    """Get list of all godowns"""
    try:
        if not initialization_complete:
            return jsonify({
                'success': False,
                'error': 'System is still initializing',
                'progress': initialization_progress
            }), 503

        if initialization_error:
            return jsonify({
                'success': False,
                'error': f'System failed to initialize: {initialization_error}'
            }), 500

        godowns = recommendation_system.get_godowns()
        print(f"Retrieved godowns: {godowns}")  # Debug log
        
        if not godowns:
            return jsonify({
                'success': False,
                'error': 'No godowns found'
            }), 404
            
        return jsonify({
            'success': True,
            'godowns': godowns
        })
    except Exception as e:
        print(f"Error getting godowns: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/customers', methods=['POST'])
def get_customers():
    """Get customers for a given godown code"""
    try:
        if not initialization_complete:
            return jsonify({
                'success': False,
                'error': 'System is still initializing',
                'progress': initialization_progress
            }), 503

        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        godown_code = data.get('godown_code')
        if not godown_code:
            return jsonify({
                'success': False,
                'error': 'Godown code is required'
            }), 400
        
        print(f"Fetching customers for godown: {godown_code}")  # Debug log
        customers = recommendation_system.get_customers_by_godown(godown_code)
        
        if not customers:
            return jsonify({
                'success': False,
                'error': 'No customers found for this godown'
            }), 404
            
        print(f"Found {len(customers)} customers")  # Debug log
        return jsonify({
            'success': True,
            'customers': customers
        })
    except Exception as e:
        print(f"Error getting customers: {str(e)}")  # Debug log
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recommendations/<customer_id>')
def get_recommendations(customer_id):
    """Get recommendations for a specific customer in a godown"""
    try:
        if not initialization_complete:
            return jsonify({
                'success': False,
                'error': 'System is still initializing',
                'progress': initialization_progress
            }), 503

        # Get godown code from query parameters
        godown_code = request.args.get('godown_code')
        if not godown_code:
            return jsonify({
                'success': False,
                'error': 'Godown code is required'
            }), 400
            
        print(f"Generating recommendations for customer {customer_id} in godown {godown_code}")  # Debug log
            
        # Validate customer belongs to godown
        if not recommendation_system.validate_customer(godown_code, customer_id):
            return jsonify({
                'success': False,
                'error': 'Customer not found in the specified godown'
            }), 404
            
        # Get recommendations (increased to 15), passing godown_code
        recommendations = recommendation_system.get_recommendations(
            customer_id, 
            n_recommendations=15,
            godown_code=godown_code
        )
        print(f"Generated {len(recommendations) if recommendations else 0} recommendations")  # Debug log
        
        if not recommendations:
            return jsonify({
                'success': True,
                'recommendations': [],
                'message': 'No recommendations available for this customer'
            })
            
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'message': f'Found {len(recommendations)} recommendations'
        })
        
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")  # Debug log
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/product-names', methods=['GET'])
def get_product_names():
    """Get all product names and their IDs"""
    try:
        # Define data path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        product_names_path = os.path.join(current_dir, 'data', 'item_no_product_names.csv')

        # Check if file exists
        if not os.path.exists(product_names_path):
            return jsonify({
                'success': False,
                'error': 'Product names file not found'
            }), 404

        # Read the CSV file
        product_names_df = pd.read_csv(product_names_path)
        
        # Convert to list of dictionaries
        product_names = product_names_df.to_dict('records')
        
        return jsonify(product_names)
    except Exception as e:
        print(f"Error fetching product names: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\nStarting Recommendation System Server...")
    print("Note: Initial data loading may take several minutes.")
    print("Progress updates will be shown here and in the web interface.")
    
    # Start initialization in a background thread only if not already running
    if not initialization_complete and (not init_thread or not init_thread.is_alive()):
        init_thread = threading.Thread(target=initialize_system)
        init_thread.start()
    
    # Start the Flask server without debug mode to prevent double initialization
    app.run(host='0.0.0.0', port=5000) 