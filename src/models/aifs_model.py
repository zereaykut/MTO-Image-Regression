import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils import get_logger
from .conv_transformer import PositionEmbedding2D, TransformerBlock, WarmUpCosineDecay

class AIFSForecasterModel:
    """
    Implements the Phase 1 (Pre-training) and Phase 2 (Fine-tuning) 
    architecture inspired by AIFS weather forecasting models.
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.logger = get_logger()
        
        # We define the core body separately so we can reuse it
        self.base_encoder = None 
        
        # The two different model heads
        self.pretrain_model = self._build_pretrain_model()
        self.finetune_model = None # Built later using build_finetune_model()

    def _build_pretrain_model(self):
        """
        Builds the Phase 1 Model: Predicts the NEXT hour's 40x40 weather grid.
        Architecture: CNN Encoder -> Transformer Processor -> CNN Decoder
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # --- 1. LOCAL ENCODER (CNN) ---
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 40x40 -> 20x20
        
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 20x20 -> 10x10
        
        H, W, C = x.shape[1], x.shape[2], x.shape[3]
        
        # --- 2. GLOBAL PROCESSOR (Transformer) ---
        x_flat = layers.Reshape((H * W, C))(x)
        x_pos = PositionEmbedding2D(height=H, width=W, embed_dim=C)(x_flat)
        
        x_trans = TransformerBlock(embed_dim=C, num_heads=4, ff_dim=128)(x_pos)
        x_trans = TransformerBlock(embed_dim=C, num_heads=4, ff_dim=128)(x_trans)
        
        # Reshape the sequence back into a 2D spatial grid
        x_spatial = layers.Reshape((H, W, C))(x_trans)
        
        # -> SAVE THIS CORE BODY FOR PHASE 2 <-
        self.base_encoder = models.Model(inputs=inputs, outputs=x_spatial, name="Core_Weather_Processor")
        
        # --- 3. WEATHER DECODER (Phase 1 Output) ---
        dec = layers.UpSampling2D(size=(2, 2))(x_spatial) # 10x10 -> 20x20
        dec = layers.Conv2D(32, 3, padding="same", activation="relu")(dec)
        
        dec = layers.UpSampling2D(size=(2, 2))(dec)       # 20x20 -> 40x40
        
        outputs = layers.Conv2D(self.input_shape[-1], 3, padding="same", activation="linear", name="Next_Frame_Grid")(dec)
        
        return models.Model(inputs=inputs, outputs=outputs, name="Phase1_Pretrainer")

    def build_finetune_model(self, freeze_encoder=False):
        """
        Builds the Phase 2 Model: Swaps the weather decoder for a Megawatt Regression Head.
        """
        self.logger.info("Constructing Phase 2 Fine-Tuning Model...")
        
        if freeze_encoder:
            self.base_encoder.trainable = False
            
        inputs = self.base_encoder.input
        x = self.base_encoder.output
        
        # --- 4. ENERGY DECODER (Phase 2 Output) ---
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
        
        # Using ReLU to prevent predicting negative Megawatts
        outputs = layers.Dense(1, activation="relu", name="Megawatt_Prediction")(x)
        
        self.finetune_model = models.Model(inputs=inputs, outputs=outputs, name="Phase2_Finetuner")
        return self.finetune_model

    def compile_pretrain(self, learning_rate=0.001):
        self.pretrain_model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=losses.MeanSquaredError(),
            metrics=["mae"]
        )

    def compile_finetune(self, learning_rate=0.0005):
        # We use a lower learning rate for fine-tuning so we don't destroy the physics knowledge
        lr_schedule = WarmUpCosineDecay(learning_rate, warmup_steps=500, decay_steps=10000)
        
        self.finetune_model.compile(
            optimizer=optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
            loss=losses.MeanSquaredError(),
            metrics=["mae", "mse"]
        )