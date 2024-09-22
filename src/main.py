import asyncio
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
import tensorflow as tf

from config import settings
from download import (
    download_file,
    extract_zip,
    fetch_landsat_data,
    save_metadata,
    download_all_s3_data
)
from processing import (
    convert_shp_to_parquet,
    get_target_prefecture_bbox,
    create_masks,
    process_images,
    select_train_dataset,
    process_all_images
)
from conv_transformer import (
    create_datasets,
    train_model,
    MaskedMSELoss,
    TransformerBlock,
    ConvTransformer,
)
from compare_validation_images import compare_validation_images

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def download_geometry_data():
    url = "https://nlftp.mlit.go.jp/ksj/gml/data/N03/N03-2023/N03-20230101_GML.zip"
    file_path = settings.GEOMETRY_FOLDER_PATH / "N03-20230101_GML.zip"
    extract_to = settings.GEOMETRY_FOLDER_PATH / "N03-2023"
    shp_path = extract_to / 'N03-23_230101.shp'
    parquet_path = extract_to / 'N03-23_230101.parquet'

    if await download_file(url, file_path):
        if await extract_zip(file_path, extract_to):
            return await convert_shp_to_parquet(shp_path, parquet_path)
    return False

async def main(args):
    try:
        # Step 1: Download and extract geometry data
        logger.info("Step 1: Download and extract geometry data")
        if not await download_geometry_data():
            logger.error("Failed to process geometry data.")
            return

        # Step 2: Fetch Landsat data
        logger.info("Step 2: Fetch Landsat data")
        bbox = await get_target_prefecture_bbox(settings.PREFECTURE, settings.GEOMETRY_FOLDER_PATH)
        response = await fetch_landsat_data(bbox, settings.YEAR)
        if not response:
            logger.error("Failed to fetch Landsat data.")
            return

        # Step 3: Save metadata and download Landsat images
        logger.info("Step 3: Save metadata and download Landsat images")
        await save_metadata(response, settings.LOCAL_STORAGE_PATH)
        await download_all_s3_data(response)

        # Step 4: Create masks from QA pixel data
        logger.info("Step 4: Create masks from QA pixel data")
        await create_masks(settings.LOCAL_STORAGE_PATH)

        # Step 5: Process images (rotate and crop)
        logger.info("Step 5: Process images (rotate and crop)")
        await process_images(settings.LOCAL_STORAGE_PATH)

        # Step 6: Select training dataset based on cloud coverage
        logger.info("Step 6: Select training dataset based on cloud coverage")
        await select_train_dataset(settings.LOCAL_STORAGE_PATH)

        # Step 7: Process all images and create tiles
        logger.info("Step 7: Process all images and create tiles")
        await process_all_images()

        # Step 8: Prepare datasets and train model
        logger.info("Step 8: Prepare datasets and train model")
        train_ds, valid_ds = create_datasets(settings.TRAIN_VALID_PATH, settings.N_SAMPLE, settings.BATCH_SIZE)
        
        if args.train:
            train_model(
                train_ds, 
                valid_ds, 
                settings.MODEL_PATH, 
                settings.WEIGHT_PATH,
                settings.BASE_PATH,
                settings.EPOCHS,
                settings.PATIENCE
                )
        
        logger.info("Pipeline completed successfully!")

        # Load the latest trained model
        custom_objects = {
            'TransformerBlock': TransformerBlock,
            'MaskedMSELoss': MaskedMSELoss,
            'ConvTransformer': ConvTransformer
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(settings.MODEL_PATH)
            
        # Evaluate the model on validation dataset
        logger.info("Evaluating model on validation dataset")
        await compare_validation_images(model, valid_ds, num_samples=5)

    except Exception as e:
        logger.exception(f"An error occurred during pipeline execution: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landsat Cloud Removal Pipeline")
    parser.add_argument("--train", action="store_true", help="Train the model after data processing")
    args = parser.parse_args()

    asyncio.run(main(args))