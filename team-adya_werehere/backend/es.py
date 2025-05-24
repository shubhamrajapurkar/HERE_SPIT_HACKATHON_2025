import os
from functools import lru_cache

from dotenv import load_dotenv
from elasticsearch import AuthenticationException, ConnectionError, Elasticsearch
from elasticsearch.exceptions import ConnectionTimeout

load_dotenv()


@lru_cache(maxsize=1)
def get_es_client() -> Elasticsearch:
    """
    Returns a singleton Elasticsearch client instance.
    Raises ValueError if connection cannot be established.
    """
    try:
        hosts = os.environ.get("ES_HOST")
        api_key = os.environ.get("ES_API_KEY")
        client = Elasticsearch(
            hosts,
            api_key=api_key,
            retry_on_timeout=True,
            max_retries=3,
        )

        print("Connected to Elasticsearch")

        return client
    except (ConnectionError, AuthenticationException, ConnectionTimeout) as e:
        raise ValueError(f"Failed to connect to Elasticsearch: {e}")
