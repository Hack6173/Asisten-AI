"""
asisten_ai.search_provider - Antarmuka Penyedia Pencarian Web yang Dapat Dikonfigurasi
Mendukung DuckDuckGo (bebas API key), SearXNG lokal/publik, dan mode Mock/Offline.
Aturan Penting: Jangan pernah berpura-pura telah melakukan pencarian jika internet tidak tersedia!
"""
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from pydantic import BaseModel
from asisten_ai.config import SYSTEM_CONFIG
from asisten_ai.offline_manager import OFFLINE_MANAGER

logger = logging.getLogger(__name__)


class SearchResultItem(BaseModel):
    title: str
    url: str
    snippet: str
    source_domain: str
    published_date: Optional[str] = None
    query: str


class BaseSearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[SearchResultItem]:
        pass


class DuckDuckGoSearchProvider(BaseSearchProvider):
    def __init__(self, timeout: int = 10, rate_limit_delay: float = 1.5):
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_call_time = 0.0

    def search(self, query: str, max_results: int = 5) -> List[SearchResultItem]:
        if not OFFLINE_MANAGER.is_online():
            raise ConnectionError(
                "Tidak dapat melakukan pencarian web: Koneksi internet tidak aktif (Mode Offline)."
            )

        # Rate limiting throttle
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

        results: List[SearchResultItem] = []
        try:
            from duckduckgo_search import DDGS

            with DDGS(timeout=self.timeout) as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))
                self._last_call_time = time.time()

                for r in raw_results:
                    title = r.get("title", "").strip()
                    url = r.get("href", "").strip()
                    body = r.get("body", "").strip()
                    if not url or not title:
                        continue

                    domain = urlparse(url).netloc
                    results.append(
                        SearchResultItem(
                            title=title,
                            url=url,
                            snippet=body,
                            source_domain=domain,
                            published_date=None,
                            query=query,
                        )
                    )
        except Exception as e:
            logger.error(f"DuckDuckGo search error untuk query '{query}': {e}")
            raise RuntimeError(f"Gagal melakukan pencarian via DuckDuckGo: {e}")

        return results


class SearXNGSearchProvider(BaseSearchProvider):
    def __init__(self, base_url: str, api_key: str = "", timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5) -> List[SearchResultItem]:
        if not OFFLINE_MANAGER.is_online():
            raise ConnectionError(
                "Tidak dapat melakukan pencarian web: Koneksi internet tidak aktif."
            )

        import requests

        params = {"q": query, "format": "json"}
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.get(
                f"{self.base_url}/search",
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_results = data.get("results", [])[:max_results]

            results = []
            for r in raw_results:
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")
                domain = urlparse(url).netloc
                results.append(
                    SearchResultItem(
                        title=title,
                        url=url,
                        snippet=content,
                        source_domain=domain,
                        published_date=r.get("publishedDate"),
                        query=query,
                    )
                )
            return results
        except Exception as e:
            raise RuntimeError(f"Gagal melakukan pencarian via SearXNG ({self.base_url}): {e}")


class MockSearchProvider(BaseSearchProvider):
    """
    Search provider untuk lingkungan pengujian terisolasi atau simulasi deterministik.
    TIDAK digunakan untuk memalsukan data pengguna nyata.
    """
    def __init__(self, canned_data: Optional[Dict[str, List[Dict[str, str]]]] = None):
        self.canned_data = canned_data or {}

    def search(self, query: str, max_results: int = 5) -> List[SearchResultItem]:
        items = self.canned_data.get(query, [])
        return [
            SearchResultItem(
                title=item.get("title", "Mock Title"),
                url=item.get("url", "https://example.org/doc"),
                snippet=item.get("snippet", "Mock snippet"),
                source_domain=item.get("domain", "example.org"),
                published_date="2026-01-01",
                query=query,
            )
            for item in items[:max_results]
        ]


def get_search_provider(override_name: Optional[str] = None) -> BaseSearchProvider:
    provider_name = override_name or SYSTEM_CONFIG.search.provider.lower()
    if provider_name == "searxng":
        return SearXNGSearchProvider(
            base_url=SYSTEM_CONFIG.search.searxng_url,
            api_key=SYSTEM_CONFIG.search.api_key,
            timeout=SYSTEM_CONFIG.search.timeout_seconds,
        )
    elif provider_name == "mock":
        return MockSearchProvider()
    else:
        return DuckDuckGoSearchProvider(
            timeout=SYSTEM_CONFIG.search.timeout_seconds,
            rate_limit_delay=SYSTEM_CONFIG.search.rate_limit_delay_seconds,
        )
