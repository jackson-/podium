import httpx
from typing import Any, Dict, Optional
import logging

class SpaceXClient:
    BASE_URL = "https://api.spacexdata.com/v4"

    def __init__(self):
        self.client = httpx.AsyncClient(base_url=self.BASE_URL, timeout=10.0)

    async def _get(self, endpoint: str) -> Dict[str, Any]:
        try:
            response = await self.client.get(endpoint)
            response.raise_for_status()
            return self._clean_data(response.json())
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            return {"error": f"API returned {e.response.status_code}"}
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return {"error": "Internal client error"}

    async def _post(self, endpoint: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = await self.client.post(endpoint, json=json_data)
            response.raise_for_status()
            return self._clean_data(response.json())
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            return {"error": f"API returned {e.response.status_code}"}
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return {"error": "Internal client error"}

    def _clean_data(self, data: Any) -> Any:
        """Recursively remove null values and excessive IDs to save tokens."""
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

    async def query_launches(self, query: Dict[str, Any], options: Optional[Dict[str, Any]] = None):
        payload = {"query": query}
        if options:
            payload["options"] = options
        return await self._post("/launches/query", payload)

    async def close(self):
        await self.client.aclose()
