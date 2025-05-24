# processing.py
import tempfile
import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from utils.comparator import GeoJSONComparator

processing_bp = Blueprint('processing', __name__, url_prefix='/processing')

@processing_bp.route('/compare', methods=['POST'])
def compare_files():
    """Endpoint for comparing two GeoJSON files using temp files"""
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"error": "Both files required"}), 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save files to temp directory
            file1_path = os.path.join(temp_dir, secure_filename(file1.filename))
            file2_path = os.path.join(temp_dir, secure_filename(file2.filename))
            
            file1.save(file1_path)
            file2.save(file2_path)

            # Initialize comparator
            comparator = GeoJSONComparator(api_key="YOUR_API_KEY")  # Replace with actual key
            
            # Call the EXACT method name from your comparator.py
            result = comparator.compare_with_gemini(file1_path, file2_path)
            
            return jsonify({
                "status": "success",
                "comparison": result
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Comparison failed: {str(e)}"
        }), 500
