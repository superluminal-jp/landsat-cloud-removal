import asyncio
import aiohttp
import aiofiles
import boto3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import settings

logger = logging.getLogger(__name__)

async def download_file(url: str, file_path: Path) -> bool:
    """
    Asynchronously download a file from a given URL.

    Args:
        url (str): The URL of the file to download.
        file_path (Path): The local path where the file will be saved.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    if file_path.exists():
        logger.info(f"File already exists: {file_path}")
        return True
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                async with aiofiles.open(file_path, mode='wb') as f:
                    await f.write(await response.read())
        logger.info(f"Downloaded file from {url}")
        return True
    except aiohttp.ClientError as e:
        logger.error(f"Failed to download file from {url}: {e}")
        return False

async def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract a zip file to a specified directory.

    Args:
        zip_path (Path): Path to the zip file.
        extract_to (Path): Directory to extract the contents to.

    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    if extract_to.exists():
        logger.info(f"Extraction folder already exists: {extract_to}")
        return True
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Extracted zip file to {extract_to}")
        return True
    except zipfile.BadZipFile as e:
        logger.error(f"Failed to extract zip file {zip_path}: {e}")
        return False

async def download_from_s3(s3_uri: str, local_path: Path) -> None:
    """
    Download a file from Amazon S3 to a local path.

    Args:
        s3_uri (str): The S3 URI of the file to download.
        local_path (Path): The local path where the file will be saved.
    """
    if local_path.exists():
        logger.info(f"File already exists: {local_path}")
        return
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        bucket, key = s3_uri.replace("s3://", "").split("/", 1)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local_path), ExtraArgs={'RequestPayer': 'requester'})
        logger.info(f"Downloaded from S3: {s3_uri}")
    except Exception as e:
        logger.error(f"Failed to download from S3 {s3_uri}: {e}")

async def fetch_landsat_data(bbox: List[float], year: int) -> Optional[Dict[str, Any]]:
    """
    Fetch Landsat data for a given bounding box and year.

    Args:
        bbox (List[float]): Bounding box coordinates [min_lon, min_lat, max_lon, max_lat].
        year (int): The year for which to fetch Landsat data.

    Returns:
        Optional[Dict[str, Any]]: The Landsat data response, or None if the request failed.
    """
    params = {
        **settings.LANDSAT_SEARCH_PARAMS,
        "bbox": bbox,
        "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(settings.LANDSAT_SEARCH_URL, json=params) as response:
                response.raise_for_status()
                data = await response.json()
        logger.info(f"Fetched Landsat data for bbox {bbox} in {year}")
        return data
    except aiohttp.ClientError as e:
        logger.error(f"Failed to fetch Landsat data: {e}")
        return None

async def save_metadata(response: Dict[str, Any], local_storage_path: Path) -> None:
    """
    Save metadata from Landsat response to local JSON files.

    Args:
        response (Dict[str, Any]): The Landsat API response containing metadata.
        local_storage_path (Path): The local path where metadata files will be saved.
    """
    for feature in response.get('features', []):
        metadata = {
            **feature['properties'],
            'id': feature['id'],
            'description': feature.get('description', ''),
            'bbox': feature['bbox'],
            'geometry': feature['geometry'],
            's3_uris': [feature['assets'][asset]['alternate']['s3']['href'] for asset in settings.ASSETS if asset in feature['assets']]
        }
        save_folder = local_storage_path / '/'.join(metadata['s3_uris'][0].split('/')[3:-1])
        save_folder.mkdir(parents=True, exist_ok=True)
        save_to = save_folder / f"{metadata['s3_uris'][0].split('/')[-2]}_metadata.json"
        if not save_to.exists():
            async with aiofiles.open(save_to, mode='w') as f:
                await f.write(json.dumps(metadata, indent=2))
    logger.info(f"Saved metadata as JSON")

async def download_all_s3_data(response: Dict[str, Any]) -> None:
    """
    Download all S3 data referenced in the Landsat response.

    Args:
        response (Dict[str, Any]): The Landsat API response containing S3 URIs.
    """
    s3_uris = [
        uri for feature in response.get('features', [])
        for uri in [feature['assets'][asset]['alternate']['s3']['href']
                    for asset in settings.ASSETS if asset in feature['assets']]
    ]
    tasks = [download_from_s3(uri, settings.LOCAL_STORAGE_PATH / uri.split('/', 3)[-1])
             for uri in s3_uris if 'oli-tirs' in uri]
    await asyncio.gather(*tasks)

async def main() -> None:
    """
    Main function to demonstrate the usage of download functions.
    """
    bbox = [135.0, 34.0, 136.0, 35.0]  # Example bounding box
    response = await fetch_landsat_data(bbox, settings.YEAR)
    if response:
        await save_metadata(response, settings.LOCAL_STORAGE_PATH)
        await download_all_s3_data(response)

if __name__ == "__main__":
    asyncio.run(main())