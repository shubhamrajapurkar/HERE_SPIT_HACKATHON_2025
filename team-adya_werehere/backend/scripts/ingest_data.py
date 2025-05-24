import json
import sys

from elasticsearch import Elasticsearch
from tqdm import tqdm

sys.path.append("../backend")

from config import RESTAURANT_INDEX_NAME
from es import get_es_client

RESTAURANT_INDEX_SETTINGS = {
    "analysis": {
        "analyzer": {
            "cuisine_analyzer": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "trim_restaurant_suffix", "unique"],
            },
            "custom_text_analyzer": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": [
                    "lowercase",
                    "english_stop",
                    "english_stemmer",
                    "asciifolding",
                ],
            },
            "edge_ngram_analyzer": {
                "tokenizer": "standard",
                "filter": ["lowercase", "edge_ngram_filter"],
            },
        },
        "filter": {
            "trim_restaurant_suffix": {
                "type": "pattern_replace",
                "pattern": "\\s+restaurant$",
                "replacement": "",
            },
            "english_stop": {"type": "stop", "stopwords": "_english_"},
            "english_stemmer": {"type": "stemmer", "language": "english"},
            "edge_ngram_filter": {
                "type": "edge_ngram",
                "min_gram": 2,
                "max_gram": 15,
            },
        },
        "normalizer": {
            "lowercase_normalizer": {"type": "custom", "filter": ["lowercase"]}
        },
    }
}

RESTAURANT_INDEX_MAPPING = {
    "properties": {
        "name": {
            "type": "text",
            "analyzer": "custom_text_analyzer",
            "fields": {
                "keyword": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "edge_ngram": {
                    "type": "text",
                    "analyzer": "edge_ngram_analyzer",
                },
            },
        },
        "location": {
            "type": "text",
            "analyzer": "standard",
            "fields": {
                "keyword": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "edge_ngram": {
                    "type": "text",
                    "analyzer": "edge_ngram_analyzer",
                },
            },
        },
        "contact": {
            "properties": {
                "phone": {"type": "keyword", "null_value": ""},
                "email": {"type": "keyword", "null_value": ""},
                "website": {"type": "keyword", "null_value": ""},
            }
        },
        "cuisines": {
            "type": "text",
            "analyzer": "cuisine_analyzer",
            "fields": {
                "keyword": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "edge_ngram": {
                    "type": "text",
                    "analyzer": "edge_ngram_analyzer",
                },
            },
        },
        "dietary_options": {
            "properties": {
                "vegetarian": {"type": "boolean"},
                "jain": {"type": "boolean"},
                "vegan": {"type": "boolean"},
                "halal": {"type": "boolean"},
                "kosher": {"type": "boolean"},
                "gluten_free": {"type": "boolean"},
            }
        },
        "certification": {"type": "keyword"},
        "rating": {"type": "float"},
        "quality": {"type": "float"},
        "hygiene": {
            "type": "keyword",
            "ignore_above": 256,
            "normalizer": "lowercase",
            "null_value": "acceptable",
        },
        "coordinates": {
            "properties": {"lat": {"type": "float"}, "lng": {"type": "float"}},
        },
    }
}


def index_data(documents: list[dict]):
    es = get_es_client()

    if es is None:
        print("Indexing failed: no elasticsearch instance connected")
        return

    response = _create_index(es)
    if "acknowledged" not in response:
        print("Indexing failed: index creation error")
        print(response)
        return

    response = _index_documents(es, documents)
    if response["errors"]:
        print("Indexing failed: bulk insertion error")
        print(response)
        return

    print(
        f"Indexed {len(documents)} documents into Elasticsearch Index {RESTAURANT_INDEX_NAME}"
    )


def _create_index(es: Elasticsearch) -> dict:
    es.indices.delete(index=RESTAURANT_INDEX_NAME, ignore_unavailable=True)
    response = es.indices.create(
        index=RESTAURANT_INDEX_NAME,
        mappings=RESTAURANT_INDEX_MAPPING,
        settings=RESTAURANT_INDEX_SETTINGS,
    )
    return response.body


def _preprocess_document(doc: dict) -> dict:
    processed = doc.copy()

    if "contact" in processed:
        for field in ["phone", "email", "website"]:
            if processed["contact"].get(field) in ["None", None]:
                processed["contact"][field] = ""

    return processed


def _index_documents(es: Elasticsearch, documents: list[dict]) -> dict:
    operation = []
    indexed_names = set()

    for document in tqdm(documents, total=len(documents), desc="Processing Documents"):
        name = document.get("name", "").strip()
        if not name or name in indexed_names:
            continue

        exists_query = {
            "query": {
                "term": {"name.keyword": {"value": name, "case_insensitive": True}}
            },
            "size": 1,
        }

        search_result = es.search(index=RESTAURANT_INDEX_NAME, body=exists_query)

        if search_result["hits"]["total"]["value"] > 0:
            continue

        processed_doc = _preprocess_document(document)
        operation.append({"index": {}})
        operation.append(processed_doc)
        indexed_names.add(name)

    if not operation:
        return {"errors": False, "items": []}

    response = es.bulk(index=RESTAURANT_INDEX_NAME, operations=operation)
    return response.body


if __name__ == "__main__":
    with open("../data/restaurant_review.json") as f:
        sourced_documents = json.load(f)

    index_data(documents=sourced_documents)
