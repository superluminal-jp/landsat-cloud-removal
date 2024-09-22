import asyncio
import aiofiles
import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageOps, ImageChops, ImageFilter
import geopandas as gpd
from config import settings
from scipy import ndimage
import cv2

logger = logging.getLogger(__name__)

async def convert_shp_to_parquet(shp_path: Path, parquet_path: Path) -> bool:
    """
    Convert a shapefile to parquet format.

    Args:
        shp_path (Path): Path to the input shapefile.
        parquet_path (Path): Path where the parquet file will be saved.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    if parquet_path.exists():
        logger.info(f"Parquet file already exists: {parquet_path}")
        return True
    try:
        gdf = gpd.read_file(shp_path, encoding='shift-jis')
        gdf.to_parquet(parquet_path)
        logger.info(f"Converted shapefile {shp_path} to parquet {parquet_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to convert shapefile to parquet: {e}")
        return False

async def get_target_prefecture_bbox(pref: str, geometry_folder_path: Path) -> Optional[List[float]]:
    """
    Get the bounding box for a target prefecture.

    Args:
        pref (str): Name of the prefecture.
        geometry_folder_path (Path): Path to the folder containing geometry data.

    Returns:
        Optional[List[float]]: Bounding box coordinates [min_lon, min_lat, max_lon, max_lat] or None if not found.
    """
    try:
        fn = geometry_folder_path / 'N03-2023/N03-23_230101.parquet'
        geometries = gpd.read_parquet(fn)
        one_pref = geometries[geometries['N03_001'] == pref]
        return list(one_pref.total_bounds)
    except Exception as e:
        logger.error(f"Failed to get target prefecture bbox: {e}")
        return None

async def create_masks(local_storage_path: Path) -> None:
    """
    Create masks from QA pixel data for all Landsat images in the specified directory if they don't already exist.

    Args:
        local_storage_path (Path): Path to the directory containing Landsat images.
    """
    qa_paths = list(local_storage_path.glob('**/oli-tirs/**/*_QA_PIXEL.TIF'))
    
    async def process_qa_file(qa_path: Path) -> None:
        try:
            bit_indices = {
                'fill': 0, 'dilated_cloud': 1, 'cirrus': 2, 'cloud': 3,
                'cloud_shadow': 4, 'snow': 5, 'clear': 6, 'water': 7,
                'cloud_confidence_low': 8, 'cloud_confidence_medium': 9,
                'cloud_shadow_confidence_low': 10, 'cloud_shadow_confidence_high': 11,
                'snow_ice_confidence_low': 12, 'snow_ice_confidence_high': 13,
                'cirrus_confidence_low': 14, 'cirrus_confidence_high': 15
            }

            # Check if all mask files already exist
            all_masks_exist = all(qa_path.with_name(f"{qa_path.stem}_{bit_name}_mask.npy").exists() 
                                  for bit_name in bit_indices.keys())
            if all_masks_exist:
                logger.info(f"All masks already exist for {qa_path}. Skipping.")
                return

            data = np.array(Image.open(qa_path), dtype=np.uint16)
            for bit_name, bit_index in bit_indices.items():
                output_fn = qa_path.with_name(f"{qa_path.stem}_{bit_name}_mask.npy")
                if not output_fn.exists():
                    mask = ((data >> bit_index) & 1).astype(bool)
                    np.save(output_fn, mask)
                    logger.debug(f"Created mask: {output_fn}")
            
            logger.info(f"Created masks for {qa_path}")
        except Exception as e:
            logger.error(f"Failed to create mask for {qa_path}: {e}")

    await asyncio.gather(*[process_qa_file(path) for path in qa_paths])

async def process_images(local_storage_path: Path) -> None:
    """
    Process (rotate and crop) all Landsat images in the specified directory if they haven't been processed already.

    Args:
        local_storage_path (Path): Path to the directory containing Landsat images.
    """
    fill_mask_paths = list(local_storage_path.glob('**/oli-tirs/**/*_QA_PIXEL_fill_mask.npy'))

    def load_and_preprocess_mask(mask_path: Path) -> np.ndarray:
        """Load and preprocess the mask."""
        return np.load(mask_path).astype(np.uint8) * 255

    def detect_edges(image: np.ndarray) -> np.ndarray:
        """Detect edges in the image using Canny edge detection."""
        return cv2.Canny(image, 50, 150, apertureSize=3)

    def find_lines(edges: np.ndarray) -> Optional[np.ndarray]:
        """Find lines in the edge-detected image."""
        return cv2.HoughLines(edges, 1, np.pi/180, 200)

    def calculate_rotation_angle(lines: np.ndarray) -> float:
        """Calculate the rotation angle based on detected lines."""
        angles = [line[0][1] * 180/np.pi for line in lines]
        if min(angles) < 45:
            return np.mean([angle for angle in angles if angle < 45])
        return np.mean([angle for angle in angles if 45 <= angle < 90]) - 90

    def rotate_image(image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate the image by the given angle."""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        return 255 - rotated, matrix

    def crop_image(image: np.ndarray) -> np.ndarray:
        """Crop the image to remove empty borders."""
        points = cv2.findNonZero(image)
        x, y, w, h = cv2.boundingRect(points)
        for ii in range(10):
            cropped = image[y+2**ii:y+h-2**ii, x+2**ii:x+w-2**ii]
            if cropped.max() == cropped.min():
                break
        for ii in range(100):
            if cropped.shape[0] < 2**ii or cropped.shape[1] < 2**ii:
                return cropped[:2**(ii-1), :2**(ii-1)]
        return cropped

    def process_band(band_path: Path, matrix: np.ndarray, width: int, height: int, cropped_shape: Tuple[int, int]) -> np.ndarray:
        """Process a single band of the image."""
        mono = cv2.imread(str(band_path), cv2.IMREAD_UNCHANGED)
        mono = np.expand_dims(mono, axis=-1)
        mono_rotate = cv2.warpAffine(mono, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        center_y, center_x = mono_rotate.shape[0] // 2, mono_rotate.shape[1] // 2
        half_height, half_width = cropped_shape[0] // 2, cropped_shape[1] // 2
        return mono_rotate[center_y - half_height:center_y + half_height, 
                        center_x - half_width:center_x + half_width]

    def process_mask(mask_path: Path, matrix: np.ndarray, width: int, height: int, cropped_shape: Tuple[int, int]) -> np.ndarray:
        """Process a mask (cloud, cloud shadow, or cirrus)."""
        mask = load_and_preprocess_mask(mask_path)
        mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        center_y, center_x = mask.shape[0] // 2, mask.shape[1] // 2
        half_height, half_width = cropped_shape[0] // 2, cropped_shape[1] // 2
        return (mask[center_y - half_height:center_y + half_height, 
                    center_x - half_width:center_x + half_width] / 255).astype(bool)

    async def process_image(fill_mask_path: Path) -> None:
        """Process the Landsat image and its associated masks if not already processed."""
        try:
            output_path = fill_mask_path.with_name(f"{fill_mask_path.stem.replace('_QA_PIXEL_fill_mask', '_color')}.png")
            mask_output_path = fill_mask_path.with_name(f"{fill_mask_path.stem.replace('_QA_PIXEL_fill_mask', '_masks')}.png")
            
            if output_path.exists() and mask_output_path.exists():
                logger.info(f"Processed image and mask already exist for {fill_mask_path}. Skipping.")
                return

            gray = load_and_preprocess_mask(fill_mask_path)
            edges = detect_edges(gray)
            lines = find_lines(edges)

            if lines is None:
                logger.warning(f"No lines detected in the image: {fill_mask_path}")
                return

            angle = calculate_rotation_angle(lines)
            logger.debug(f"Detected rotation angle: {angle}")

            rotated_image, matrix = rotate_image(gray, angle)
            cropped_image = crop_image(rotated_image)
            height, width = gray.shape[:2]

            # Process color channels
            bands = ['B2', 'B3', 'B4']
            color_channels = [process_band(fill_mask_path.parent / f"{fill_mask_path.stem.replace('_QA_PIXEL_fill_mask', f'_SR_{band}')}.TIF", 
                                        matrix, width, height, cropped_image.shape) for band in bands]
            color = cv2.merge(color_channels)

            cv2.imwrite(str(output_path), color)
            logger.info(f"Saved processed color image: {output_path}")

            # Process masks
            mask_types = ['cloud', 'cloud_shadow', 'cirrus']
            masks = []

            for mask_type in mask_types:
                mask_path = fill_mask_path.with_name(f"{fill_mask_path.stem.replace('fill_mask', f'{mask_type}_mask')}.npy")
                if mask_path.exists():
                    masks.append(process_mask(mask_path, matrix, width, height, cropped_image.shape))
                else:
                    logger.warning(f"{mask_type} mask not found: {mask_path}")

            if masks:
                combined_mask = np.logical_or.reduce(masks) * 1
                cv2.imwrite(str(mask_output_path), combined_mask * 255)
                logger.info(f"Saved processed mask image: {mask_output_path}")
            else:
                logger.warning(f"No valid masks found for {fill_mask_path}")

            logger.info(f"Processed image and masks saved for {fill_mask_path}")
        except Exception as e:
            logger.error(f"Failed to process image {fill_mask_path}: {e}")
            logger.exception("Detailed error information:")

    await asyncio.gather(*[process_image(path) for path in fill_mask_paths])

async def select_train_dataset(local_storage_path: Path) -> None:
    """
    Select training dataset based on cloud coverage.

    Args:
        local_storage_path (Path): Path to the directory containing Landsat metadata.
    """
    try:
        logger.info(f"Starting to select training dataset from {local_storage_path}")
        metadata_paths = list(local_storage_path.glob('**/oli-tirs/**/*.json'))
        logger.info(f"Found {len(metadata_paths)} metadata files")
        cloud_coverages = {}

        async def read_metadata(path: Path) -> None:
            async with aiofiles.open(path, 'r') as f:
                metadata = json.loads(await f.read())
                cloud_coverages[path] = metadata.get('eo:cloud_cover', 100)

        logger.info("Reading metadata files")
        await asyncio.gather(*[read_metadata(path) for path in metadata_paths])
        logger.info("Finished reading metadata files")

        metadata_cloud_coverage_equal_0 = [str(path) for path, coverage in cloud_coverages.items() if coverage == 0]
        metadata_cloud_coverage_less_than_10 = [str(path) for path, coverage in cloud_coverages.items() if 0 < coverage <= 10]
        metadata_cloud_coverage_less_than_20 = [str(path) for path, coverage in cloud_coverages.items() if 0 < coverage <= 20]
        
        logger.info(f"metadata_cloud_coverage_equal_0: {len(metadata_cloud_coverage_equal_0)}")
        logger.info(f"metadata_cloud_coverage_less_than_10: {len(metadata_cloud_coverage_less_than_10)}")
        logger.info(f"metadata_cloud_coverage_less_than_20: {len(metadata_cloud_coverage_less_than_20)}")

        async def save_list_to_file(file_name: str, data_list: List[str]) -> None:
            async with aiofiles.open(file_name, 'w') as file:
                await file.write('\n'.join(data_list))
            logger.info(f"Saved {len(data_list)} items to {file_name}")
        
        logger.info("Saving results to files")
        await asyncio.gather(
            save_list_to_file('./metadata_cloud_coverage_equal_0.csv', metadata_cloud_coverage_equal_0),
            save_list_to_file('./metadata_cloud_coverage_less_than_10.csv', metadata_cloud_coverage_less_than_10),
            save_list_to_file('./metadata_cloud_coverage_less_than_20.csv', metadata_cloud_coverage_less_than_20)
        )
        
        logger.info("Selected training dataset based on cloud coverage")
    except Exception as e:
        logger.error(f"Failed to select training dataset: {e}", exc_info=True)

async def process_all_images() -> None:
    """
    Process all images and create tiles for training.
    """
    try:
        logger.info("Starting to process all images")
        metadata_files = [
            './metadata_cloud_coverage_equal_0.csv',
            './metadata_cloud_coverage_less_than_10.csv',
            './metadata_cloud_coverage_less_than_20.csv'
        ]
        
        async def read_metadata_file(filename: str) -> List[str]:
            async with aiofiles.open(filename, 'r') as f:
                return [line.strip().replace('./', '') for line in await f.readlines()]
        
        meta_data = []
        for filename in metadata_files:
            logger.info(f"Reading metadata from {filename}")
            meta_data.extend(await read_metadata_file(filename))
        
        list_ = list(set(meta_data))
        logger.info(f"Total unique metadata entries: {len(list_)}")
        
        image_pairs = [(item, item) for item in list_]
        logger.info(f"Created {len(image_pairs)} image pairs")
        
        logger.info("Starting to process image pairs")
        await asyncio.gather(*[process_image_pair(pair) for pair in image_pairs])
        logger.info("Finished processing image pairs")
        
        # Create tiles
        png_files = list((settings.TRAIN_VALID_PATH / 'png').glob('*.png'))
        logger.info(f"Found {len(png_files)} PNG files for tile creation")
        
        async def create_tiles_logged(file: Path) -> None:
            logger.debug(f"Creating tiles for file: {file}")
            await create_tiles(file)
            logger.debug(f"Finished creating tiles for file: {file}")
        
        logger.info("Starting tile creation")
        await asyncio.gather(*[create_tiles_logged(file) for file in png_files])
        logger.info("Finished creating tiles")
        
        logger.info("Processed all images and created tiles successfully")
    except Exception as e:
        logger.error(f"Failed to process all images: {e}", exc_info=True)

async def read_and_mask_image(metadata_file: str, mask_suffix: str = '_masks.png') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read an image and its mask based on the metadata file.
    
    Args:
        metadata_file (str): Path to the metadata file.
        mask_suffix (str): Suffix for the mask file.
    
    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Tuple containing (masked_image, original_image) or (None, None) if processing failed.
    """
    try:
        logger.info(f"Processing image for metadata file: {metadata_file}")
        
        # Construct image filename
        img_fn = metadata_file.replace('_metadata.json', '_color.png')
        if not Path(img_fn).exists():
            logger.warning(f"Image file not found: {img_fn}")
            return None, None
        
        logger.debug(f"Reading image file: {img_fn}")
        img = Image.open(img_fn)
        img_array = np.array(img)
        logger.debug(f"Image dimensions: {img_array.shape}")
        
        # Construct mask filename
        mask_fn = metadata_file.replace('_metadata.json', mask_suffix)
        if not Path(mask_fn).exists():
            logger.warning(f"Mask file not found: {mask_fn}")
            return None, None
        
        logger.debug(f"Reading mask file: {mask_fn}")
        mask = Image.open(mask_fn)
        mask_array = np.array(mask)
        logger.debug(f"Mask shape: {mask_array.shape}")
        
        # Ensure mask has the same dimensions as the image
        if mask_array.shape[:2] != img_array.shape[:2]:
            logger.warning(f"Mask shape {mask_array.shape} does not match image shape {img_array.shape}")
            mask_array = cv2.resize(mask_array, (img_array.shape[1], img_array.shape[0]))
            logger.info(f"Resized mask to {mask_array.shape}")
        
        # Ensure mask is 3D
        if len(mask_array.shape) == 2:
            mask_array = np.expand_dims(mask_array, axis=-1)
        
        # Apply mask to image
        masked_image = img_array * (mask_array == 0)  # Invert mask if necessary
        logger.info(f"Successfully masked image for {metadata_file}")
        logger.debug(f"Masked image shape: {masked_image.shape}")
        
        return masked_image, img_array
    
    except Exception as e:
        logger.error(f"Failed to read and mask image {metadata_file}: {e}", exc_info=True)
        return None

async def save_processed_images(x_masked: np.ndarray, y_original: np.ndarray) -> None:
    """
    Save processed image pairs if they don't already exist.
    Args:
    x_masked (np.ndarray): Input image array with mask applied.
    y_original (np.ndarray): Target image array without mask.
    """
    try:
        output_dir = settings.TRAIN_VALID_PATH / 'png'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Preparing to save processed images to {output_dir}")
        logger.debug(f"Input masked image shape: {x_masked.shape}, Target original image shape: {y_original.shape}")
        
        idx = 0  # len(list(output_dir.glob('X*.png')))
        logger.debug(f"Current index for saving: {idx}")
        
        x_path = output_dir / f'X{idx}.png'
        y_path = output_dir / f'Y{idx}.png'
        
        if x_path.exists() or y_path.exists():
            logger.info(f"Skipping save: X{idx}.png or Y{idx}.png already exists")
            return

        Image.fromarray(x_masked).save(x_path)
        Image.fromarray(y_original).save(y_path)
        
        logger.info(f"Successfully saved processed images: X{idx}.png (masked) and Y{idx}.png (original)")
    except Exception as e:
        logger.error(f"Failed to save processed images: {e}", exc_info=True)

async def process_image_pair(pair: Tuple[str, str]) -> None:
    logger.debug(f"Processing image pair: {pair}")
    x_masked, x_original = await read_and_mask_image(pair[0])
    if x_masked is not None and x_original is not None:
        await save_processed_images(x_masked, x_original)
        logger.debug(f"Saved processed images for pair: {pair}")
    else:
        logger.warning(f"Failed to process image pair: {pair}")

async def create_tiles(image_path: Path) -> None:
    """
    Create tiles from a given image if they don't already exist.
    Args:
    image_path (Path): Path to the input image.
    """
    try:
        logger.info(f"Preparing to create tiles for image: {image_path}")
        
        output_dir = settings.TRAIN_VALID_PATH / 'tiles'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if tiles already exist
        existing_tiles = list(output_dir.glob(f"tile_{image_path.stem}_*.png"))
        if existing_tiles:
            logger.info(f"Skipping tile creation: {len(existing_tiles)} tiles already exist for {image_path.stem}")
            return
        
        image = np.array(Image.open(image_path))
        logger.debug(f"Loaded image shape: {image.shape}")
        
        tiles_created = 0
        for y in range(0, image.shape[0], settings.TILE_SIZE):
            for x in range(0, image.shape[1], settings.TILE_SIZE):
                tile = image[y:y+settings.TILE_SIZE, x:x+settings.TILE_SIZE]
                if tile.shape[0] == settings.TILE_SIZE and tile.shape[1] == settings.TILE_SIZE:
                    tile_filename = f"tile_{image_path.stem}_{x//settings.TILE_SIZE}_{y//settings.TILE_SIZE}.png"
                    tile_path = output_dir / tile_filename
                    if not tile_path.exists():
                        Image.fromarray(tile).save(tile_path)
                        tiles_created += 1
                        logger.debug(f"Saved tile: {tile_filename}")
        
        logger.info(f"Created {tiles_created} new tiles for {image_path}")
    except Exception as e:
        logger.error(f"Failed to create tiles for {image_path}: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(process_all_images())