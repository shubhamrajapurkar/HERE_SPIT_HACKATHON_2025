from flask import Flask, request, jsonify
from graph import graph
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

@app.route('/route', methods=['POST'])
def get_route():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Missing prompt in request'}), 400
    
    try:
        state = graph.invoke({"prompt": data['prompt']})
        if "final_route" in state:
            # Convert route locations to dict if they exist
            if "route_locations" in state["final_route"]:
                route_locations = []
                for loc in state["final_route"]["route_locations"]:
                    if "location" in loc:
                        loc["location"] = loc["location"].dict()
                    if "suggestions" in loc:
                        loc["suggestions"] = [s.dict() for s in loc["suggestions"]]
                    route_locations.append(loc)
                state["final_route"]["route_locations"] = route_locations
                
            return jsonify({
                "route": state["final_route"],
                "type": "transit" if state.get("transport_mode") == "transit" else "car"
            })
        return jsonify({'error': 'No route found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)