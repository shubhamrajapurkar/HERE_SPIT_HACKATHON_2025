from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from config import RESTAURANT_INDEX_NAME
from es import get_es_client
from models import Restaurant, SearchRequest, SearchResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

es = get_es_client()


@app.get("/api/v1/hello")
def read_root():
    return {"Hello": "World"}


@app.post("/api/v1/search")
async def search(request: SearchRequest) -> SearchResponse:
    if not es:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Elasticsearch is not available",
        )

    query = {"bool": {"must": [], "filter": []}}

    if request.query:
        query["bool"]["must"].append(
            {
                "bool": {
                    "should": [
                        # Exact matches with high boost
                        {
                            "multi_match": {
                                "query": request.query,
                                "fields": [
                                    "name.keyword^4",
                                    "cuisines.keyword^3",
                                    "location.keyword^2",
                                ],
                                "type": "cross_fields",
                                "operator": "and",
                            }
                        },
                        # Analyzed text search with medium boost
                        {
                            "multi_match": {
                                "query": request.query,
                                "fields": ["name^3", "cuisines^2", "location^1.5"],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                                "operator": "and",
                            }
                        },
                        # Edge ngram for prefix matching
                        {
                            "multi_match": {
                                "query": request.query,
                                "fields": [
                                    "name.edge_ngram^2",
                                    "cuisines.edge_ngram^1.5",
                                    "location.edge_ngram",
                                ],
                                "type": "phrase_prefix",
                                "operator": "and",
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            }
        )

    if request.cuisine_types:
        query["bool"]["filter"].append({"terms": {"cuisines": request.cuisine_types}})

    if request.min_rating is not None:
        query["bool"]["filter"].append(
            {"range": {"rating": {"gte": request.min_rating}}}
        )

    if request.min_quality is not None:
        query["bool"]["filter"].append(
            {"range": {"quality": {"gte": request.min_quality}}}
        )

    if request.min_hygiene:
        hygiene_hierarchy = [
            "very good",
            "good",
            "acceptable",
            "needs improvement",
            "bad",
            "very bad",
        ]

        try:
            min_index = hygiene_hierarchy.index(request.min_hygiene)
            acceptable_levels = hygiene_hierarchy[: min_index + 1]
            query["bool"]["filter"].append({"terms": {"hygiene": acceptable_levels}})
        except ValueError:
            query["bool"]["filter"].append({"terms": {"hygiene": ["acceptable"]}})

    if request.dietary_preferences:
        for pref, value in request.dietary_preferences.dict(exclude_unset=True).items():
            if value:
                query["bool"]["filter"].append(
                    {"term": {f"dietary_options.{pref}": True}}
                )

    if request.location:
        query["bool"]["filter"].append(
            {"match": {"location": {"query": request.location, "operator": "and"}}}
        )

    if not query["bool"]["must"] and not query["bool"]["filter"]:
        query = {"match_all": {}}

    from_val = (request.page - 1) * request.page_size

    response = es.search(
        index=RESTAURANT_INDEX_NAME,
        query=query,
        from_=from_val,
        size=request.page_size,
        sort=[
            {"rating": {"order": "desc"}},
            {"quality": {"order": "desc"}},
            {"name.keyword": {"order": "asc"}},  # For consistent ordering
        ],
        timeout="30s",
        track_total_hits=True,
    )

    if "hits" not in response:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search service returned invalid response",
        )

    hits = response["hits"]
    total = hits["total"]["value"] if isinstance(hits["total"], dict) else hits["total"]

    restaurants = []
    for hit in hits["hits"]:
        restaurants.append(Restaurant(**hit["_source"]))

    return SearchResponse(
        total=total,
        page=request.page,
        page_size=request.page_size,
        results=restaurants,
    )
