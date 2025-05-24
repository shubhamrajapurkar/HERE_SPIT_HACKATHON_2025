from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Literal

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
api_key = os.getenv("GEMINI_API_KEY")
here_key = os.getenv("HERE_API_KEY")
machine_ip = os.getenv("MACHINE_IP")
langsmith_key = os.getenv("LANGSMITH_API_KEY")

os.environ["LANGCHAIN_API_KEY"] = langsmith_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langchain-groq-demo"

# llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.3-70b-versatile")
llm = ChatGoogleGenerativeAI(  # Using chat model
    model="gemini-1.5-pro",
    google_api_key=api_key,
)

class State(TypedDict):
    prompt: str
    parsed_route: dict
    route1: dict
    route2: dict
    final_route: dict
    transport_mode: Literal["car", "transit"]

class Location(BaseModel):
    name: str
    type: Literal["accounting", "airport", "atm", "bakery", "bank", "bar", "beauty_salon", 
                  "book_store", "bus_station", "cafe", "car_dealer", "car_repair", "church", 
                  "clothing_store", "dentist", "doctor", "drugstore", "electronics_store", 
                  "embassy", "finance", "food", "furniture_store", "gas_station", 
                  "general_contractor", "gym", "hair_care", "hardware_store", "health", 
                  "hindu_temple", "home_goods_store", "hospital", "insurance_agency", 
                  "jewelry_store", "laundry", "lodging", "mosque", "movie_theater", "park", 
                  "parking", "pharmacy", "place_of_worship", "point_of_interest", "post_office", 
                  "primary_school", "real_estate_agency", "restaurant", "school", "shoe_store", 
                  "shopping_mall", "spa", "storage", "store", "subway_station", "supermarket", 
                  "tourist_attraction", "train_station", "transit_station", "travel_agency", 
                  "university"]

class Locations(BaseModel):
    locations: list[Location]

class GeocodedLocation(BaseModel):
    name: str
    type: Literal["accounting", "airport", "atm", "bakery", "bank", "bar", "beauty_salon", 
                  "book_store", "bus_station", "cafe", "car_dealer", "car_repair", "church", 
                  "clothing_store", "dentist", "doctor", "drugstore", "electronics_store", 
                  "embassy", "finance", "food", "furniture_store", "gas_station", 
                  "general_contractor", "gym", "hair_care", "hardware_store", "health", 
                  "hindu_temple", "home_goods_store", "hospital", "insurance_agency", 
                  "jewelry_store", "laundry", "lodging", "mosque", "movie_theater", "park", 
                  "parking", "pharmacy", "place_of_worship", "point_of_interest", "post_office", 
                  "primary_school", "real_estate_agency", "restaurant", "school", "shoe_store", 
                  "shopping_mall", "spa", "storage", "store", "subway_station", "supermarket", 
                  "tourist_attraction", "train_station", "transit_station", "travel_agency", 
                  "university"]
    latitude: float
    longitude: float

class GeocodedRoute(BaseModel):
    locations: list[GeocodedLocation]

def route(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that identifies the user input and extracts the desired path that the user wants to take. For each location, you must categorize it with one of these specific types: point_of_interest for main destinations, and for extra stops use the most appropriate type from: restaurant, bar, cafe, gas_station, parking, etc. The type must be one of the predefined values from the here Maps Places API types. IMPORTANT: You must maintain the exact order of locations as specified by the user. For example, if the user wants to go from A to B to C (main locations) and wants to stop at a restaurant after B (extra location), the locations in the response must be ordered as: A, B, restaurant, C. For each main location that is, point of interest, the user will specify the surrounding area, for example, if the user says I want to go from SPIT in andheri, then while adding the name, add the suffix ', Andheri''"),
        ("human", "{prompt}"),
    ])
    llm_with_structure = llm.with_structured_output(Locations)
    prompt = prompt.invoke({"prompt": state["prompt"]})
    response = llm_with_structure.invoke(prompt)
    return {"parsed_route": response.locations}

def check_transport_mode(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that determines if the user wants to travel by car or public transit. Only respond with 'car' or 'transit' based on the user's input. no additional text, only one word - 'car' or 'transit'."),
        ("human", "{prompt}"),
    ])
    prompt = prompt.invoke({"prompt": state["prompt"]})
    response = llm.invoke(prompt)
    return {"transport_mode": response.content.lower()}

def public_transit(state: State):
    import requests
    machine_ip = os.getenv("MACHINE_IP")
    route_locations = state["route1"]["route_locations"]
    
    # Process locations to get coordinates
    processed_locations = []
    for loc in route_locations:
        if loc["type"] == "main":
            processed_locations.append(loc["location"])
        elif loc["type"] == "extra" and loc["suggestions"]:
            # Take only the first suggestion for extra locations
            processed_locations.append(loc["suggestions"][0])
    
    if len(processed_locations) < 2:
        print("Need at least start and end locations for routing")
        return {"route2": None}
    
    # Extract start and end coordinates
    startLat, startLng = processed_locations[0].latitude, processed_locations[0].longitude
    endLat, endLng = processed_locations[-1].latitude, processed_locations[-1].longitude
    
    # Build URL with all locations
    base_url = f"http://{machine_ip}:8080/otp/routers/default/plan"
    url = f"{base_url}?fromPlace={startLat}%2C{startLng}"
    
    # Add intermediate locations if any
    if len(processed_locations) > 2:
        # Add each intermediate location (excluding start and end)
        for loc in processed_locations[1:-1]:
            url += f"&intermediatePlaces={loc.latitude}%2C{loc.longitude}"
    
    # Add end location and other parameters
    url += f"&toPlace={endLat}%2C{endLng}"
    url += "&date=2025-01-16&time=09:00:00&arriveBy=false&mode=TRANSIT%2CWALK&maxWalkDistance=1000&numItineraries=5"
    
    try:
        print("Making request to:", url)
        response = requests.get(url)
        print("response", response)
        if response.status_code == 200:
            return {"route2": response.json()}
        else:
            print(f"Error getting transit route: {response.status_code}")
            return {"route2": None}
    except Exception as e:
        print(f"Exception during transit request: {str(e)}")
        return {"route2": None}

def geocode(state: State):
    import uuid
    import requests

    here_api_key = os.getenv("HERE_API_KEY")
    route_locations = []  # List to maintain order of all locations

    for i, location in enumerate(state["parsed_route"]):
        headers = {
            'X-Request-Id': str(uuid.uuid4()),
            'X-Correlation-Id': str(uuid.uuid4()),
            "Origin": "http://localhost:8081",
        }

        if location.type == "point_of_interest":
            # HERE Geocoding & Search API: autocomplete for main locations
            params = {
                'q': location.name,
                'apiKey': here_api_key,
                'limit': 1,
            }
            response = requests.get(
                'https://autosuggest.search.hereapi.com/v1/autosuggest',
                headers=headers,
                params=params
            )
            if response.status_code == 200:
                data = response.json()
                # Take the first item with position
                items = data.get('items', [])
                found = False
                for item in items:
                    if 'position' in item:
                        found = True
                        route_locations.append({
                            "type": "main",
                            "location": GeocodedLocation(
                                name=location.name,
                                type=location.type,
                                latitude=item['position']['lat'],
                                longitude=item['position']['lng']
                            )
                        })
                        break
                if not found:
                    print(f"Warning: No valid position found for input: {location.name}")
                    route_locations.append({
                        "type": "main",
                        "location": GeocodedLocation(
                            name=location.name,
                            type=location.type,
                            latitude=0.0,
                            longitude=0.0
                        )
                    })
            else:
                print(f"Error calling HERE Autosuggest API for {location.name}: {response.status_code}")
                route_locations.append({
                    "type": "main",
                    "location": GeocodedLocation(
                        name=location.name,
                        type=location.type,
                        latitude=0.0,
                        longitude=0.0
                    )
                })
        else:
            # Extra locations: use HERE Places API to search nearby
            main_locations = [item for item in route_locations if item["type"] == "main"]
            if main_locations:
                last_main_location = main_locations[-1]["location"]
                lat = last_main_location.latitude
                lng = last_main_location.longitude

                # Use HERE Browse API for nearby places of a certain category
                params = {
                    'at': f"{lat},{lng}",
                    'categories': location.type,
                    'limit': 5,
                    'apiKey': here_api_key,
                }
                response = requests.get(
                    'https://browse.search.hereapi.com/v1/browse',
                    headers=headers,
                    params=params
                )
                suggestions = []
                if response.status_code == 200:
                    data = response.json()
                    items = data.get('items', [])
                    for item in items:
                        if 'position' in item:
                            suggestions.append(GeocodedLocation(
                                name=item.get('title', location.type),
                                type=location.type,
                                latitude=item['position']['lat'],
                                longitude=item['position']['lng']
                            ))
                else:
                    print(f"Error calling HERE Browse API for type {location.type}: {response.status_code}")

                if suggestions:
                    route_locations.append({
                        "type": "extra",
                        "location_type": location.type,
                        "suggestions": suggestions
                    })
                else:
                    print(f"Warning: No valid suggestions found for type: {location.type}")

    return {"route1": {"route_locations": route_locations}}

def should_continue(state: State):
    return "public_transit" if state.get("transport_mode") == "transit" else "extract_route"

def extract_route(state: State):
    if "route2" in state:
        return {"final_route": state["route2"]}
    elif "route1" in state:
        return {"final_route": state["route1"]}
    return {"final_route": None}

# First create the graph builder
graph_builder = StateGraph(State)

# Then add all the nodes
graph_builder.add_node("route", route)
graph_builder.add_node("check_transport_mode", check_transport_mode)
graph_builder.add_node("geocode", geocode)
graph_builder.add_node("public_transit", public_transit)
graph_builder.add_node("extract_route", extract_route)

# Then add all the edges
graph_builder.add_edge(START, "route")
graph_builder.add_edge("route", "geocode")
graph_builder.add_edge("geocode", "check_transport_mode")
graph_builder.add_conditional_edges("check_transport_mode", should_continue)
graph_builder.add_edge("public_transit", "extract_route")
graph_builder.add_edge("extract_route", END)
graph_builder.add_edge("public_transit", END)

# Finally compile the graph
graph = graph_builder.compile()
# state = graph.invoke({"prompt": "I want to go from kjsce in vidyavihar to greeshma residency in thane using car. I also want to stop at a restaurant in between"})
# print("parsed - ", state["parsed_route"])
# print("route - ", state["route1"])
# print("final - ", state["final_route"])
