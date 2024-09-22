import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, List, Optional
import logging
from pathlib import Path
import os
import datetime

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConvTransformer(Model):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (256, 256, 3),
        patch_size: Tuple[int, int] = (16, 16),
        depth: int = 1,
        num_heads: int = 64,
        projection_dim: int = 256,
        conv_depth: int = 3,
    ):
        super(ConvTransformer, self).__init__()
        
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.depth = depth
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.conv_depth = conv_depth

        self.input_layer = layers.Input(shape=input_shape, name="input_img")
        
        self.conv_layers = [
            layers.Conv2D(filters=input_shape[-1], kernel_size=(3, 3), padding='same', activation='gelu')
            for _ in range(conv_depth)
        ]
        
        num_patches_h = input_shape[0] // patch_size[0]
        num_patches_w = input_shape[1] // patch_size[1]
        self.patch_layer = layers.Reshape((num_patches_h * num_patches_w, patch_size[0] * patch_size[1] * input_shape[2]))
        
        self.transformer_layers = [
            TransformerBlock(num_heads, projection_dim)
            for _ in range(depth)
        ]
        
        self.reconstruction_layer = layers.Reshape(input_shape)
        
        self.deconv_layers = [
            layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=(3, 3), padding='same', activation='gelu')
            for _ in range(conv_depth)
        ]

        self.final_conv = layers.Conv2D(filters=input_shape[-1], kernel_size=(1, 1), activation='linear')

    def call(self, inputs, training=False):
        x = inputs
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        patches = self.patch_layer(x)
        
        for transformer_layer in self.transformer_layers:
            patches = transformer_layer(patches)
        
        x = self.reconstruction_layer(patches)
        
        for deconv_layer in self.deconv_layers:
            x = deconv_layer(x)
        
        x = self.final_conv(x)
        
        return x

    def build_graph(self):
        x = layers.Input(shape=self.input_shape)
        return Model(inputs=[x], outputs=self.call(x))

class TransformerBlock(layers.Layer):
    def __init__(self, num_heads: int, projection_dim: int, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.norm1(inputs)
        x = self.attn(x, x)
        return self.norm2(inputs + x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "projection_dim": self.projection_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MaskedMSELoss(tf.keras.losses.Loss):
    def __init__(self, base_mask_weight: float = 1.0, **kwargs):
        super(MaskedMSELoss, self).__init__(**kwargs)
        self.base_mask_weight = base_mask_weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # マスクがないため、単純なMSE損失を計算
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        return mse

    def get_config(self):
        config = super().get_config()
        config.update({"base_mask_weight": self.base_mask_weight})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def create_datasets(train_valid_path: Path, n_sample: int, batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create training and validation datasets."""
    logger.info(f"Creating datasets from {train_valid_path} with {n_sample} samples and batch size {batch_size}")

    def load_image(path: str) -> tf.Tensor:
        logger.debug(f"Loading image: {path}")
        img = tf.image.decode_png(tf.io.read_file(path), channels=3, dtype=tf.uint16)
        img = tf.cast(img, tf.float32) / 65535.0
        return img

    image_paths = sorted([str(fn) for fn in train_valid_path.glob('tiles/tile_X*.png')])[:n_sample]
    target_paths = sorted([str(fn) for fn in train_valid_path.glob('tiles/tile_Y*.png')])[:n_sample]

    logger.info(f"Found {len(image_paths)} input images and {len(target_paths)} target images")

    if len(image_paths) != len(target_paths):
        logger.warning(f"Mismatch in number of input and target images: {len(image_paths)} vs {len(target_paths)}")

    train_size = int(0.8 * len(image_paths))
    logger.info(f"Using {train_size} samples for training and {len(image_paths) - train_size} for validation")

    train_ds = tf.data.Dataset.from_tensor_slices((image_paths[:train_size], target_paths[:train_size]))
    valid_ds = tf.data.Dataset.from_tensor_slices((image_paths[train_size:], target_paths[train_size:]))

    logger.debug("Mapping load_image function to datasets")
    train_ds = train_ds.map(lambda x, y: (load_image(x), load_image(y)), num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(lambda x, y: (load_image(x), load_image(y)), num_parallel_calls=tf.data.AUTOTUNE)

    logger.debug(f"Applying shuffle (buffer_size=1000), batch (size={batch_size}), and prefetch to training dataset")
    train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    logger.debug(f"Applying batch (size={batch_size}) and prefetch to validation dataset")
    valid_ds = valid_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    logger.info("Dataset creation completed")
    return train_ds, valid_ds

def logistic_transformation(image: tf.Tensor, alpha: float = 10.0) -> tf.Tensor:
    """Apply logistic transformation to the input image."""
    non_zero_mask = tf.math.greater(image, 0)
    transformed_image = 1 / (1 + tf.exp(-alpha * (image - 0.5)))
    transformed_image = (transformed_image - tf.reduce_min(transformed_image)) / (tf.reduce_max(transformed_image) - tf.reduce_min(transformed_image))
    return tf.where(non_zero_mask, transformed_image, tf.zeros_like(image))

def train_model(
    train_ds: tf.data.Dataset,
    valid_ds: tf.data.Dataset,
    model_path: Path,
    weight_path: Path,
    base_path: Path,
    epochs: int,
    patience: int
) -> None:
    """Train the ConvTransformer model."""
    
    # Ensure base_path exists
    base_path.mkdir(parents=True, exist_ok=True)

    model = ConvTransformer(
        input_shape=(settings.TILE_SIZE, settings.TILE_SIZE, 3),
        depth=settings.DEPTH
    )
    model = model.build_graph()  # Build the model graph
    model.compile(optimizer="nadam", loss=MaskedMSELoss())

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=patience, 
        verbose=1, 
        restore_best_weights=True
    )
    
    # Use a more detailed naming convention for checkpoints
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_filepath = base_path / f'weights_epoch_{{epoch:02d}}_{timestamp}.weights.h5'
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_filepath),
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )

    tensorboard_log_dir = base_path / "logs" / timestamp
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir, 
        histogram_freq=1
    )

    try:
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=valid_ds,
            callbacks=[early_stopping, model_checkpoint, tensorboard_callback]
        )
        
        # Save the full model and weights
        model.save(model_path)
        model.save_weights(weight_path)
        
        # Save the best model weights
        best_epoch = early_stopping.best_epoch
        best_weights_path = base_path / f'best_weights_{timestamp}.weights.h5'
        model.save_weights(best_weights_path)
        
        logger.info(f"Model training completed successfully. Best epoch: {best_epoch}")
        logger.info(f"Best weights saved to: {best_weights_path}")
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}", exc_info=True)
        raise

def main():
    try:
        # Create datasets
        train_ds, valid_ds = create_datasets(settings.TRAIN_VALID_PATH, settings.N_SAMPLE, settings.BATCH_SIZE)
        
        def check_dataset_shapes(dataset):
            for x, y in dataset.take(1):
                print(f"Input shape: {x.shape}")
                print(f"Target shape: {y.shape}")
        
        check_dataset_shapes(train_ds)
        check_dataset_shapes(valid_ds)  

        # Train model
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
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()