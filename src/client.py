import httpx
from typing import Any, Dict, Optional
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class SpaceXClient:
    """
    Async HTTP client for interacting with the SpaceX API v4.
    
    Architectural Choices:
    - **Async/Await**: Uses `httpx` for non-blocking I/O, essential for high-performance agents.
    - **Tenacity Retry**: Implements exponential backoff for resilience against transient API failures.
    - **Data Cleaning**: Recursively strips nulls/unused fields to optimize token usage for the LLM.
    """
    BASE_URL = "https://api.spacexdata.com/v4"
    WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary"

    def __init__(self):
        self.client = httpx.AsyncClient(base_url=self.BASE_URL, timeout=10.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError))
    )
    async def _get(self, endpoint: str) -> Dict[str, Any]:
        """
        Generic GET request wrapper with automatic retries and error handling.
        Returns cleaned JSON data or an error dict.
        """
        try:
            response = await self.client.get(endpoint)
            response.raise_for_status()
            return self._clean_data(response.json())
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            return {"error": f"API returned {e.response.status_code}"}
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise e # Raise to trigger retry for connection errors

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError))
    )
    async def _post(self, endpoint: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generic POST request wrapper (used for Query endpoints) with retries.
        """
        try:
            response = await self.client.post(endpoint, json=json_data)
            response.raise_for_status()
            return self._clean_data(response.json())
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            return {"error": f"API returned {e.response.status_code}"}
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise e # Raise to trigger retry

    def _clean_data(self, data: Any) -> Any:
        """
        Recursively remove null values and excessive IDs to save tokens.
        
        The SpaceX API returns very verbose JSON with many null fields and 
        long lists of IDs (ships, payloads, etc.) that consume context window 
        without adding value for most user queries.
        """
        if isinstance(data, list):
            return [self._clean_data(item) for item in data if item is not None]
        elif isinstance(data, dict):
            cleaned = {}
            for k, v in data.items():
                # Skip large arrays of IDs or specific meta fields if necessary
                if v is not None and not k.endswith("_id") and k not in ["ships", "capsules", "payloads"]:
                    cleaned[k] = self._clean_data(v)
            return cleaned
        return data

    async def get_next_launch(self):
        return await self._get("/launches/next")

    async def get_latest_launch(self):
        return await self._get("/launches/latest")

    async def get_rocket(self, rocket_id: str):
        return await self._get(f"/rockets/{rocket_id}")

    async def get_company_info(self):
        return await self._get("/company")

    async def get_launchpad(self, launchpad_id: str):
        return await self._get(f"/launchpads/{launchpad_id}")
    
    async def get_wikipedia_summary(self, query: str):
        """
        Fallback method to get summary from Wikipedia.
        
        Logic:
        1. Uses OpenSearch API to find the canonical page title (handles capitalization/typos).
        2. Fetches the REST summary for that specific title.
        
        This two-step process is much more robust than guessing URLs.
        """
        headers = {
            "User-Agent": "SpaceXAgent/1.0 (devin@example.com)" # Wikipedia requires a User-Agent
        }
        async with httpx.AsyncClient(timeout=5.0, headers=headers) as wiki_client:
            try:
                # Step 1: Search for the page title
                search_url = "https://en.wikipedia.org/w/api.php"
                search_params = {
                    "action": "opensearch",
                    "search": query,
                    "limit": 1,
                    "namespace": 0,
                    "format": "json"
                }
                search_resp = await wiki_client.get(search_url, params=search_params)
                
                try:
                    search_data = search_resp.json()
                except Exception:
                    return {"error": f"Wikipedia Search API returned invalid JSON: {search_resp.text[:100]}"}
                
                # Opensearch returns [query, [titles], [descriptions], [urls]]
                if not search_data[1]:
                    return {"error": f"No Wikipedia page found for '{query}'."}
                
                correct_title = search_data[1][0]
                
                # Step 2: Get summary for the found title
                # We use the REST API for a clean summary
                summary_url = f"{self.WIKIPEDIA_API}/{correct_title.replace(' ', '_')}"
                response = await wiki_client.get(summary_url)
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "title": data.get("title"), 
                        "summary": data.get("extract"), 
                        "source_url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
                    }
                else:
                    return {"error": f"Failed to fetch summary for '{correct_title}'."}
                    
            except Exception as e:
                return {"error": f"Wikipedia fallback failed: {str(e)}"}

    async def query_launches(self, query: Dict[str, Any], options: Optional[Dict[str, Any]] = None):
        """
        Wraps the powerful POST /launches/query endpoint.
        Allows complex Mongo-style filtering (handled by the agent).
        """
        payload = {"query": query}
        if options:
            payload["options"] = options
        return await self._post("/launches/query", payload)

    async def close(self):
        await self.client.aclose()
