# here_mcp_server.py

import httpx
import json
from typing import Any, Optional, Union, Literal, List, Dict
from mcp.server.fastmcp import FastMCP

# Keep this secure and do not commit it to public repositories if possible.
HERE_API_KEY = "4ZvgSs1gqgUkgDjB_sCR9YyE-cmBSx593Eus4Za2tSk" 

# Initialize FastMCP server
mcp = FastMCP(
    "here_geocoding_search_v7",
    title="HERE Geocoding & Search API v7 (Server-Side Key)",
    description="Tools for interacting with the HERE Geocoding and Search API v7, using a server-configured API key."
)

# Constants
USER_AGENT = "MCPHereGeocodingSearchServer/1.0"

# --- Helper Function for API Requests ---
async def make_api_request(
    method: Literal["GET", "POST"],
    base_url: str,
    query_params: Optional[Dict[str, Any]] = None,
    body_data: Optional[Union[Dict[str, Any], str]] = None, 
    extra_headers: Optional[Dict[str, str]] = None
) -> str:
    """
    Makes an API request to the HERE Geocoding and Search API using the server-configured API key.
    Returns the response text (usually JSON string) or an error message string.
    """
    if not HERE_API_KEY or HERE_API_KEY == "YOUR_HERE_API_KEY":
        return json.dumps({"error": "API key is not configured on the server. Please set HERE_API_KEY."})

    actual_headers = {"User-Agent": USER_AGENT}
    if extra_headers:
        actual_headers.update(extra_headers)

    all_query_params = query_params.copy() if query_params else {}
    all_query_params["apiKey"] = HERE_API_KEY
    
    processed_query_params = {}
    for k, v in all_query_params.items():
        if v is not None:
            if isinstance(v, list):
                processed_query_params[k] = ",".join(map(str, v))
            elif isinstance(v, bool):
                processed_query_params[k] = str(v).lower()
            else:
                processed_query_params[k] = v
    
    request_kwargs = {
        "params": processed_query_params,
        "headers": actual_headers,
        "timeout": 30.0
    }

    async with httpx.AsyncClient() as client:
        try:
            if method.upper() == "GET":
                response = await client.get(base_url, **request_kwargs)
            elif method.upper() == "POST":
                if isinstance(body_data, str) and actual_headers.get("Content-Type", "").startswith("text/plain"):
                    request_kwargs["content"] = body_data
                elif isinstance(body_data, dict): 
                    request_kwargs["data"] = body_data
                response = await client.post(base_url, **request_kwargs)
            else:
                return json.dumps({"error": f"Unsupported HTTP method: {method}"})

            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            error_content = e.response.text
            try:
                error_json_detail = json.loads(error_content)
                return json.dumps({"error": "API Error", "status_code": e.response.status_code, "details": error_json_detail})
            except json.JSONDecodeError:
                 return json.dumps({"error": "API Error", "status_code": e.response.status_code, "raw_content": error_content})
        except httpx.RequestError as e:
            return json.dumps({"error": "Request Error", "details": str(e)})
        except Exception as e:
            return json.dumps({"error": "An unexpected error occurred", "details": str(e)})

# --- Tool Definitions ---

@mcp.tool()
async def geocode_address(
    q: Optional[str] = None,
    qq: Optional[str] = None,
    limit: Optional[int] = None,
    at: Optional[str] = None,
    in_filter: Optional[str] = None, 
    lang: Optional[List[str]] = None,
    show: Optional[List[str]] = None,
    x_request_id: Optional[str] = None
) -> str:
    """
    Geocode an address. Finds geo-coordinates for addresses, places, etc.
    Either 'q' (free-text query) or 'qq' (qualified query) must be provided.

    Args:
        q: Free-text query (e.g., "125, Berliner, berlin").
        qq: Qualified query (e.g., "city=Berlin;country=Germany;street=Friedrichstr;houseNumber=20").
        limit: Maximum number of results to be returned (1-100).
        at: Search context center as coordinates (latitude,longitude).
        in_filter: Search within a geographic area (e.g., "countryCode:USA"). API calls this 'in'.
        lang: Preferred BCP 47 language codes for results (e.g., ['en-US', 'de-DE']).
        show: Additional fields to render in response (e.g., ['countryInfo', 'tz']).
        x_request_id: Optional custom request ID for correlation.
    """
    if not q and not qq:
        return json.dumps({"error": "Either 'q' or 'qq' parameter must be provided."})

    base_url = "https://geocode.search.hereapi.com/v1/geocode"
    params = {
        "q": q,
        "qq": qq,
        "limit": limit,
        "at": at,
        "in": in_filter,
        "lang": lang,
        "show": show,
    }
    extra_headers = {}
    if x_request_id:
        extra_headers["X-Request-ID"] = x_request_id
        
    return await make_api_request("GET", base_url, query_params=params, extra_headers=extra_headers)

@mcp.tool()
async def discover_places(
    q: str,
    at: Optional[str] = None,
    in_filter: Optional[str] = None,
    ranking: Optional[str] = None,
    route: Optional[str] = None,
    with_features: Optional[List[str]] = None,
    ev_station_filters: Optional[Dict[str, Any]] = None,
    lang: Optional[List[str]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    political_view: Optional[str] = None,
    show: Optional[List[str]] = None,
    x_request_id: Optional[str] = None
) -> str:
    """
    Discover places using GET request. Processes a free-form text query for an address or place.
    One of 'at', 'in_filter' (circle/bbox), or 'in_filter' (countryCode + at/circle/bbox) is required.

    Args:
        q: Free-text query (e.g., "Eismieze Berlin", "125, Berliner, berlin").
        at: Search context center (latitude,longitude), e.g., "52.5308,13.3856".
        in_filter: Search area. Options:
                  - Country: "countryCode:USA" or "countryCode:CAN,MEX,USA"
                  - Circle: "circle:52.53,13.38;r=10000" (lat,lon;radius_in_meters)
                  - Bbox: "bbox:13.08836,52.33812,13.761,52.6755" (west,south,east,north)
        ranking: Ranking mode. Currently supports "excursionDistance" (BETA, requires route).
        route: Geographic corridor (BETA). Format: "{flexible_polyline};w={width_meters}"
               Examples: "BFoz5xJ67i1B1B7PzIhaxL7Y" or "BFoz5xJ67i1B1B7PzIhaxL7Y;w=5000"
        with_features: Activate features, e.g., ["recommendPlaces"] (ALPHA, RESTRICTED).
        ev_station_filters: EV station constraints, e.g., {"minPower": 50.0, "connectorTypeIds": ["29","33"]}.
        lang: Preferred BCP 47 language codes, e.g., ["en-US", "de-DE"].
        limit: Maximum results (1-100, default 20).
        offset: Pagination offset (0-99, enables pagination mode).
        political_view: Political view as ISO 3166-1 alpha-3 code (e.g., "ARG", "IND").
        show: Additional fields: ["eMobilityServiceProviders", "ev", "phonemes", "streetInfo", "tripadvisor", "tz"].
        x_request_id: Optional request ID for correlation.
    """
    base_url = "https://discover.search.hereapi.com/v1/discover"
    
    params = {
        "q": q,
        "at": at,
        "in": in_filter,
        "ranking": ranking,
        "route": route,
        "lang": lang,
        "limit": limit,
        "offset": offset,
        "politicalView": political_view,
        "show": show,
    }
    
    # Handle 'with' parameter (reserved keyword in Python)
    if with_features:
        params["with"] = with_features
    
    # Handle EV station filters
    if ev_station_filters:
        for k, v in ev_station_filters.items():
            param_name = f"evStation[{k}]"
            if isinstance(v, list):
                params[param_name] = ",".join(map(str, v))
            else:
                params[param_name] = str(v)
    
    extra_headers = {}
    if x_request_id:
        extra_headers["X-Request-ID"] = x_request_id
        
    return await make_api_request("GET", base_url, query_params=params, extra_headers=extra_headers)

@mcp.tool()
async def lookup_by_id(
    here_id: str, 
    lang: Optional[List[str]] = None,
    show: Optional[List[str]] = None,
    x_request_id: Optional[str] = None
) -> str:
    """
    Lookup a known place by its HERE ID.

    Args:
        here_id: The HERE ID of the location (e.g., "here:pds:place:276u33db-8097f3194e4b411081b761ea9a366776"). API calls this 'id'.
        lang: Preferred language codes for results.
        show: Additional fields to render.
        x_request_id: Optional custom request ID.
    """
    base_url = "https://lookup.search.hereapi.com/v1/lookup"
    params = {
        "id": here_id,
        "lang": lang,
        "show": show,
    }
    extra_headers = {}
    if x_request_id:
        extra_headers["X-Request-ID"] = x_request_id
        
    return await make_api_request("GET", base_url, query_params=params, extra_headers=extra_headers)

@mcp.tool()
async def reverse_geocode(
    at: str,
    limit: Optional[int] = 1,
    types: Optional[List[str]] = None,
    lang: Optional[List[str]] = None,
    show: Optional[List[str]] = None,
    x_request_id: Optional[str] = None
) -> str:
    """
    Reverse geocode coordinates to get address information.

    Args:
        at: Coordinates to reverse geocode (latitude,longitude).
        limit: Maximum number of results (1-10).
        types: Result types to limit to (e.g., ['address', 'street']).
        lang: Preferred language codes for results.
        show: Additional fields to render.
        x_request_id: Optional custom request ID.
    """
    base_url = "https://revgeocode.search.hereapi.com/v1/revgeocode"
    params = {
        "at": at,
        "limit": limit,
        "types": types,
        "lang": lang,
        "show": show,
    }
    extra_headers = {}
    if x_request_id:
        extra_headers["X-Request-ID"] = x_request_id
        
    return await make_api_request("GET", base_url, query_params=params, extra_headers=extra_headers)

@mcp.tool()
async def multi_reverse_geocode_post(
    sub_requests_body: str,
    limit: Optional[int] = 1, 
    types: Optional[List[str]] = None,
    lang: Optional[List[str]] = None,
    show: Optional[List[str]] = None,
    x_request_id: Optional[str] = None
) -> str:
    """
    Multi-Reverse Geocode using POST. Returns addresses/places for a list of geo-coordinates.
    The request body must be a newline-separated list of sub-requests.
    Each sub-request is like "id=<unique_id>&at=<lat>,<lon>[&bearing=<val>]".

    Args:
        sub_requests_body: Newline-separated string of sub-requests.
                           Example: "id=req1&at=52.5308,13.3856\\nid=req2&at=40.7128,-74.0060&bearing=90"
        limit: Max results per sub-request (1-10, default 1).
        types: Result types to limit to (e.g., ['address', 'street']).
        lang: Preferred language codes for results.
        show: Additional fields to render.
        x_request_id: Optional custom request ID.
    """
    base_url = "https://multi-revgeocode.search.hereapi.com/v1/multi-revgeocode"
    query_params = {
        "limit": limit,
        "types": types,
        "lang": lang,
        "show": show,
    }
    
    extra_headers = {"Content-Type": "text/plain; charset=UTF-8"}
    if x_request_id:
        extra_headers["X-Request-ID"] = x_request_id

    return await make_api_request(
        "POST", 
        base_url, 
        query_params=query_params, 
        body_data=sub_requests_body, 
        extra_headers=extra_headers
    )

# --- Main Execution ---
if __name__ == "__main__":
    if not HERE_API_KEY or HERE_API_KEY == "YOUR_HERE_API_KEY":
        print("ERROR: HERE_API_KEY is not set in the script. Please edit here_mcp_server.py and add your API key.")
    else:
        print("Starting HERE Geocoding & Search API v7 MCP Server (with server-side API key)...")
        print("Available tools will be registered with the MCP host (e.g., Claude for Desktop).")
    mcp.run(transport='stdio')