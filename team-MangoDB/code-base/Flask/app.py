# --- Standard Library Imports ---
import os
import json
import asyncio
import datetime

# --- Third-party Library Imports ---
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.utils import secure_filename
from groq import Groq
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# --- Langchain Component Imports ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from flask_cors import CORS

# --- Initial Application Setup ---
# Load environment variables from a .env file for configuration
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
CORS(app, origins="*", methods=["OPTIONS", "POST"])

# --- Configuration Variables ---
# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY # Explicitly set for Langchain

# File Upload Configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database Configuration
DB_NAME = os.getenv("DB_NAME", 'db123')
DB_USER = os.getenv("DB_USER", 'shrihari')
DB_PASSWORD = os.getenv("DB_PASSWORD", '5432')
DB_HOST = os.getenv("DB_HOST", 'localhost')
DB_PORT = os.getenv("DB_PORT", '5432')

# Allowed department table names for PostGIS lookup (security measure)
ALLOWED_DEPARTMENTS = ["police", "firebrigade", "hospital"]
SEARCH_RADIUS_METERS = 5000 # Default search radius for nearby services

# --- API Client Initializations ---
# Initialize Groq client for audio transcription
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    app.logger.error(f"Failed to initialize Groq client: {e}")
    groq_client = None # Allow app to run but /transcribe will fail gracefully

# Initialize Langchain LLM for transcript processing
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
except Exception as e:
    app.logger.error(f"Failed to initialize Google Generative AI model: {e}")
    llm = None # Allow app to run but /process_transcript LLM part will fail


# --- Langchain Chain Definition ---
# JSON format instructions for the LLM
JSON_FORMAT_INSTRUCTIONS = """
The output should be a JSON object with the following keys:
- depts: A list of strings, where each string is the name of a relevant department or organization the person should contact from the given list ("police", "firebrigade", "hospital"). Infer these from the transcript.
- person_name: A string representing the full name of the person speaking or being discussed in the transcript. Extract this directly if mentioned, otherwise "Unknown".
- summary: A concise string summarizing the main situation or problem.
- key_issues: A list of strings, highlighting the main problems or challenges.
- location (optional): A string representing the location mentioned, if any.
- timestamp (optional): A string representing a specific time or date mentioned, if any.
- suggestion (optional): A string with instructions for the person in an emergency (excluding recommendations to contact emergency services directly via this output).
"""

# Prompt template for the Langchain chain
langchain_prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant specializing in summarizing emergency-related transcripts.
Your goal is to extract key information and format it as a JSON object.
{json_format_instructions}
Transcript:
{transcript}
Provide output in the specified JSON format.
"""
)

# Output parser to convert LLM string output to JSON
json_output_parser = JsonOutputParser()

# Langchain processing chain
if llm: # Only define chain if LLM initialized successfully
    langchain_chain = (
        langchain_prompt.partial(json_format_instructions=JSON_FORMAT_INSTRUCTIONS)
        | llm
        | json_output_parser
    )
else:
    langchain_chain = None
    app.logger.error("Langchain chain could not be initialized due to LLM failure.")


# --- Database Helper Functions ---
def get_db_connection():
    """
    Establishes and returns a new PostgreSQL database connection.
    Returns:
        psycopg2.connection or None: The connection object or None if connection fails.
    """
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except psycopg2.OperationalError as e:
        app.logger.error(f"Database connection failed: {e}")
        return None

def find_places_within_radius(center_lat, center_lng, radius_meters, dept_table_name):
    """
    Finds the closest place from a specified department table within a given radius.
    Args:
        center_lat (float): Latitude of the center point.
        center_lng (float): Longitude of the center point.
        radius_meters (int): Search radius in meters.
        dept_table_name (str): Name of the department table (e.g., 'police').
                               This table name MUST be validated against ALLOWED_DEPARTMENTS
                               before calling this function to prevent SQL injection.
    Returns:
        dict or None: A dictionary with the closest place's details or None if not found or error.
    """
    if dept_table_name not in ALLOWED_DEPARTMENTS:
        app.logger.error(f"Invalid department table name '{dept_table_name}' for spatial lookup.")
        return None

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if not conn:
            app.logger.warning("Skipping spatial lookup due to database connection failure.")
            return None

        cursor = conn.cursor(cursor_factory=RealDictCursor)
        query = f"""
          SELECT
            id,
            name,
            ST_Distance(location, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography) AS distance_meters,
            ST_Y(location::geometry) AS latitude,
            ST_X(location::geometry) AS longitude,
            ST_AsGeoJSON(location)::json AS location_geojson
          FROM "{dept_table_name}"
          WHERE ST_DWithin(location, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)
          ORDER BY distance_meters
          LIMIT 1;
        """
        cursor.execute(query, (center_lng, center_lat, center_lng, center_lat, radius_meters))
        closest_place = cursor.fetchone()
        
        if closest_place:
            # Add formatted coordinates for easier frontend use
            closest_place['coordinates'] = {
                'lat': float(closest_place['latitude']),
                'lng': float(closest_place['longitude'])
            }
            
        return closest_place if closest_place else None

    except psycopg2.Error as e:
        app.logger.error(f"Error finding places for department '{dept_table_name}': {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# --- Synchronous Processing Helper ---
def process_transcript_with_langchain_sync(transcript_text: str):
    """
    Processes a transcript using the Langchain chain synchronously.
    Args:
        transcript_text (str): The transcript to process.
    Returns:
        dict or None: The processed data from Langchain or None if an error occurs or chain not available.
    """
    if not langchain_chain:
        app.logger.error("Langchain chain is not available for transcript processing.")
        return None
    try:
        # Use the synchronous invoke method instead of async
        result = langchain_chain.invoke({"transcript": transcript_text})
        return result
    except Exception as e:
        app.logger.error(f"Langchain chain execution error: {e}")
        return None

# --- Flask API Endpoints ---
@app.route('/transcribe', methods=['POST'])
def transcribe_audio_endpoint():
    """
    API endpoint to transcribe an audio file using Groq API.
    Expects a file upload with the key 'file'.
    Returns:
        JSON response with transcribed text or an error message.
    """
    app.logger.info("Received request for /transcribe")
    if not groq_client:
        app.logger.error("Groq client not initialized. Cannot transcribe.")
        return jsonify({"error": "Transcription service unavailable"}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
        with open(filepath, 'rb') as audio_file:
            # Groq API expects file as a tuple: (filename, file_content_bytes)
            transcription = groq_client.audio.transcriptions.create(
                file=(filename, audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json" # "json" for just text, "verbose_json" for more details
            )
        app.logger.info(f"Transcription successful for {filename}")
        return jsonify({'text': transcription.text})

    except Exception as e:
        app.logger.error(f"Transcription error for {filename}: {e}")
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath) # Clean up the temporary file

@app.route('/process_transcript', methods=['POST'])
def process_transcript_endpoint():
    """
    API endpoint to process a transcript, find nearby services, and return analysis.
    Expects a JSON payload with 'transcript' (str), 'lat' (float), and 'lng' (float).
    Returns:
        JSON response with transcript analysis and nearby services, or an error message.
    """
    app.logger.info("Received request for /process_transcript")
    if not langchain_chain:
        app.logger.error("Langchain chain not initialized. Cannot process transcript.")
        return jsonify({"error": "Transcript processing service unavailable"}), 503

    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        transcript = request_data.get('transcript')
        lat = request_data.get('lat')
        lng = request_data.get('lng')

        if not all([transcript, isinstance(lat, (int, float)), isinstance(lng, (int, float))]):
            return jsonify({"error": "Missing or invalid 'transcript', 'lat', or 'lng' in request"}), 400

        # Perform Langchain processing using synchronous method
        processed_transcript_data = process_transcript_with_langchain_sync(transcript)
        print(processed_transcript_data)
        if processed_transcript_data is None:
            app.logger.warning("Transcript processing with Langchain failed or returned no data.")
            # Depending on requirements, you might want to proceed without LLM analysis
            # or return an error. Here, we'll indicate failure.
            return jsonify({"error": "Transcript analysis failed"}), 500
        app.logger.info("Langchain processing complete.")

        # Perform PostGIS spatial lookup for relevant departments
        depts_to_contact = processed_transcript_data.get('depts', [])
        closest_places_results = {}

        for dept_name in depts_to_contact:
            if dept_name in ALLOWED_DEPARTMENTS:
                closest_place = find_places_within_radius(lat, lng, SEARCH_RADIUS_METERS, dept_name)
                print(closest_place)
                closest_places_results[dept_name] = closest_place # Will be None if not found
                if closest_place:
                    print(closest_place)
                    app.logger.info(f"Found closest {dept_name}: {closest_place.get('name')}")
                else:
                    app.logger.info(f"No {dept_name} found within radius for lat/lng: {lat}/{lng}")
            else:
                app.logger.warning(f"LLM suggested department '{dept_name}' not in allowed list. Skipping.")


        final_result = {
            "transcript_analysis": processed_transcript_data,
            "closest_nearby_services": closest_places_results,
            "status": "completed",
            "request_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        return jsonify(final_result), 200

    except Exception as e:
        app.logger.error(f"Unexpected error in /process_transcript: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

# --- Main Execution Block ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)