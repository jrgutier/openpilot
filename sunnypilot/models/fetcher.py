"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import time

import requests
from requests.exceptions import (SSLError, RequestException, HTTPError)
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.models.helpers import is_bundle_version_compatible

from cereal import custom


class ModelParser:
  """Handles parsing of model data into cereal objects"""

  @staticmethod
  def _parse_download_uri(download_uri_data) -> custom.ModelManagerSP.DownloadUri:
    download_uri = custom.ModelManagerSP.DownloadUri()
    download_uri.uri = download_uri_data.get("url")
    download_uri.sha256 = download_uri_data.get("sha256")
    return download_uri

  @staticmethod
  def _parse_artifact(artifact_data) -> custom.ModelManagerSP.Artifact:
    artifact = custom.ModelManagerSP.Artifact()
    artifact.fileName = artifact_data.get("file_name")
    artifact.downloadUri = ModelParser._parse_download_uri(artifact_data.get("download_uri", {}))
    return artifact

  @staticmethod
  def _parse_model(model_data) -> custom.ModelManagerSP.Model:
    model = custom.ModelManagerSP.Model()

    model.type = model_data.get("type")
    model.artifact = ModelParser._parse_artifact(model_data.get("artifact", {}))
    if metadata := model_data.get("metadata"):
      model.metadata = ModelParser._parse_artifact(metadata)
    return model

  @staticmethod
  def _parse_overrides(overrides_data: dict[str, str]) -> list[custom.ModelManagerSP.Override]:
    overrides = []
    for key, value in overrides_data.items():
      override = custom.ModelManagerSP.Override()
      override.key = key
      override.value = value
      overrides.append(override)
    return overrides

  @staticmethod
  def _parse_bundle(bundle) -> custom.ModelManagerSP.ModelBundle:
    model_bundle = custom.ModelManagerSP.ModelBundle()
    model_bundle.index = int(bundle["index"])
    model_bundle.internalName = bundle["short_name"]
    model_bundle.displayName = bundle["display_name"]
    model_bundle.models = [ModelParser._parse_model(model) for model in bundle.get("models",[])]
    model_bundle.status = 0
    model_bundle.generation = int(bundle["generation"])
    model_bundle.environment = bundle["environment"]
    model_bundle.runner = bundle.get("runner", custom.ModelManagerSP.Runner.snpe)
    model_bundle.is20hz = bundle.get("is_20hz", False)
    model_bundle.minimumSelectorVersion = int(bundle["minimum_selector_version"])
    model_bundle.overrides = ModelParser._parse_overrides(bundle.get("overrides", {}))
    model_bundle.ref = bundle.get("ref")

    return model_bundle

  @staticmethod
  def parse_models(json_data: dict) -> list[custom.ModelManagerSP.ModelBundle]:
    found_bundles = []
    for bundle in json_data.get("bundles", []):
      try:
        found_bundles.append(ModelParser._parse_bundle(bundle))
      except Exception as e:
        # Newer manifests may introduce enum values (e.g. new model types) our schema doesn't know.
        # Skip those bundles rather than failing the whole fetch.
        cloudlog.warning(f"Skipping bundle {bundle.get('short_name', '?')} (unparseable): {e}")
    return [bundle for bundle in found_bundles if is_bundle_version_compatible(bundle.to_dict())]


class ModelCache:
  """Handles caching of model data to avoid frequent remote fetches"""

  def __init__(self, params: Params, cache_timeout: int = int(3600 * 1e9)):
    self.params = params
    self.cache_timeout = cache_timeout
    self._LAST_SYNC_KEY = "ModelManager_LastSyncTime"
    self._CACHE_KEY = "ModelManager_ModelsCache"

  def _is_expired(self) -> bool:
    """Checks if the cache has expired"""
    current_time = int(time.monotonic() * 1e9)
    last_sync = self.params.get(self._LAST_SYNC_KEY) or 0
    return bool(last_sync == 0) or (current_time - last_sync) >= self.cache_timeout

  def get(self) -> tuple[dict, bool]:
    """
    Retrieves cached model data and expiration status atomically.
    Returns: Tuple of (cached_data, is_expired)
    If no cached data exists or on error, returns an empty dict
    """
    try:
      cached_data = self.params.get(self._CACHE_KEY)
      if not cached_data:
        cloudlog.warning("No cached model data available")
        return {}, True
      return cached_data, self._is_expired()
    except Exception as e:
      cloudlog.exception(f"Error retrieving cached model data: {str(e)}")
      return {}, True

  def set(self, data: dict) -> None:
    """Updates the cache with new model data"""
    self.params.put(self._CACHE_KEY, data)
    self.params.put(self._LAST_SYNC_KEY, int(time.monotonic() * 1e9))


class ModelFetcher:
  """Handles fetching and caching of model data from remote source"""
  MODEL_URL_TEMPLATE = "https://raw.githubusercontent.com/sunnypilot/sunnypilot-models/refs/heads/gh-pages/docs/driving_models_v{ver}.json"
  # Known-good floor — probing walks upward from here. Bump if upstream ever retires this.
  MODEL_URL_BASELINE = 15
  # Sanity cap on how far above the baseline we probe.
  MODEL_URL_PROBE_CEILING = 20

  _resolved_url: str | None = None

  @classmethod
  def get_model_url(cls) -> str:
    """Resolves the highest-numbered `driving_models_v{N}.json` that exists upstream.

    Walks upward from MODEL_URL_BASELINE via HEAD requests, stopping at the first 404.
    Result is memoized on the class so the probe runs once per process. On any network
    error we fall back to the baseline URL — this keeps offline boots working.
    """
    if cls._resolved_url is not None:
      return cls._resolved_url

    latest = cls.MODEL_URL_BASELINE
    for ver in range(cls.MODEL_URL_BASELINE + 1, cls.MODEL_URL_BASELINE + cls.MODEL_URL_PROBE_CEILING + 1):
      try:
        resp = requests.head(cls.MODEL_URL_TEMPLATE.format(ver=ver), timeout=3, allow_redirects=True)
      except RequestException:
        break
      if resp.status_code == 404:
        break
      if resp.status_code != 200:
        cloudlog.warning(f"Model URL probe for v{ver} returned HTTP {resp.status_code}; stopping")
        break
      latest = ver

    cls._resolved_url = cls.MODEL_URL_TEMPLATE.format(ver=latest)
    cloudlog.info(f"Resolved models manifest URL: {cls._resolved_url}")
    return cls._resolved_url

  def __init__(self, params: Params):
    self.params = params
    self.model_cache = ModelCache(params)
    self.model_parser = ModelParser()

  def _fetch_and_cache_models(self) -> list[custom.ModelManagerSP.ModelBundle] | None:
    """Fetches fresh model data from remote and updates cache.
    Returns None on transport errors. Raises on 404 and other fatal HTTP errors.
    """
    url = self.get_model_url()
    try:
      response = requests.get(url, timeout=10)

      # Explicitly handle 404 differently
      if response.status_code == 404:
        cloudlog.error(f"Models URL returned 404 Not Found: {url}")
        raise HTTPError(f"404 Not Found: {url}", response=response)

      # Raise for any other 4xx/5xx
      response.raise_for_status()

      json_data = response.json()
      self.model_cache.set(json_data)
      cloudlog.debug("Successfully updated models cache")
      return self.model_parser.parse_models(json_data)

    except ConnectionError as e:
      cloudlog.warning(f"DNS/connection error while fetching models: {e}")
    except SSLError as e:
      cloudlog.warning(f"SSL error while fetching models: {e}")
    except RequestException as e:
      cloudlog.warning(f"Request transport error while fetching models: {e}")
    except Exception as e:
      cloudlog.exception(f"Unexpected error fetching models: {e}")

    return None

  def get_available_bundles(self) -> list[custom.ModelManagerSP.ModelBundle]:
    """Gets the list of available models, with smart cache handling"""
    cached_data, is_expired = self.model_cache.get()

    if cached_data and not is_expired:
      cloudlog.debug("Using valid cached models data")
      return self.model_parser.parse_models(cached_data)

    fetched_bundles = self._fetch_and_cache_models()
    if fetched_bundles is not None:
      return fetched_bundles

    if not cached_data:
      cloudlog.warning("Failed to fetch fresh data and no cache available")

    cloudlog.warning("Failed to fetch fresh data. Using expired cache as fallback")
    return self.model_parser.parse_models(cached_data)

if __name__ == "__main__":
  params = Params()
  model_fetcher = ModelFetcher(params)
  bundles = model_fetcher.get_available_bundles()
  for bundle in bundles:
    for model in bundle.models:
      model_overrides = {override.key: override.value for override in bundle.overrides}
      # Print model details
      print(f"Bundle: {bundle.internalName}, Type: {model.type}, Status: {bundle.status}, Overrides: {model_overrides}")
      # Print artifact details
      print(f"Artifact: {model.artifact.fileName}, Download URI: {model.artifact.downloadUri.uri}")
      # Print metadata details
      print(f"Metadata: {model.metadata.fileName}, Download URI: {model.metadata.downloadUri.uri}")
