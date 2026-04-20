import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow.keras.backend as K
from src.utils import get_logger
from .conv_transformer import PositionEmbedding2D, TransformerBlock, WarmUpCosineDecay

def asymmetric_mse(under_penalty=1.5, over_penalty=1.0):
    """
    Penalizes under-predictions more than over-predictions based on operational grid economics.
    """
    def loss(y_true, y_pred):
        error = y_true - y_pred
        is_under = tf.cast(error > 0, tf.float32)
        weight = (is_under * under_penalty) + ((1.0 - is_under) * over_penalty)
        return K.mean(weight * tf.square(error))
    return loss

class AIFSForecasterModel:
    """
    Implements Phase 1 (Pre-training) and Phase 2 (Spatio-Temporal Fine-tuning).
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.logger = get_logger()
        self.base_encoder = None 
        self.pretrain_model = self._build_pretrain_model()
        self.finetune_model = None 

    def _build_pretrain_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        # --- 1. LOCAL ENCODER (CNN) ---
        x1 = layers.Conv2D(32, 3, padding="same", activation="gelu")(inputs)
        x1 = layers.BatchNormalization()(x1)
        x1_down = layers.Conv2D(32, 3, strides=2, padding="same", activation="gelu")(x1) 
        
        x2 = layers.Conv2D(64, 3, padding="same", activation="gelu")(x1_down)
        x2 = layers.BatchNormalization()(x2)
        x2_down = layers.Conv2D(64, 3, strides=2, padding="same", activation="gelu")(x2) 
        
        H, W, C = x2_down.shape[1], x2_down.shape[2], x2_down.shape[3]
        
        # --- 2. GLOBAL PROCESSOR (Transformer) ---
        x_flat = layers.Reshape((H * W, C))(x2_down)
        x_pos = PositionEmbedding2D(height=H, width=W, embed_dim=C)(x_flat)
        
        x_trans = TransformerBlock(embed_dim=C, num_heads=4, ff_dim=256)(x_pos)
        x_trans = TransformerBlock(embed_dim=C, num_heads=4, ff_dim=256)(x_trans)
        
        x_spatial = layers.Reshape((H, W, C))(x_trans)
        
        # Core encoder saved for Phase 2
        self.base_encoder = models.Model(inputs=inputs, outputs=x_spatial, name="Base_Encoder")
        
        # --- 3. LOCAL DECODER (CNN) for Pre-training ---
        x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="gelu")(x_spatial)
        x = layers.Concatenate()([x, x2])
        x = layers.Conv2D(64, 3, padding="same", activation="gelu")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="gelu")(x)
        x = layers.Concatenate()([x, x1])
        x = layers.Conv2D(32, 3, padding="same", activation="gelu")(x)
        x = layers.BatchNormalization()(x)
        
        pretrain_outputs = layers.Conv2D(self.input_shape[-1], 1, activation="linear", name="Weather_Reconstruction")(x)
        
        return models.Model(inputs=inputs, outputs=pretrain_outputs, name="Phase1_Pretrainer")

    def build_finetune_model_multimodal(self, sequence_length=3, meta_dim=5, freeze_encoder=False):
        self.logger.info("Constructing Phase 2 Spatio-Temporal Multi-Modal Model...")
        
        if freeze_encoder:
            self.base_encoder.trainable = False
            
        # Inputs: Weather sequences + scalar metadata
        seq_inputs = layers.Input(shape=(sequence_length,) + self.input_shape, name="Weather_Sequence")
        meta_inputs = layers.Input(shape=(meta_dim,), name="Metadata_Input")
        
        # Branch 1: Spatial Weather features mapped over time (LSTM)
        feature_extractor = models.Sequential([
            self.base_encoder,
            layers.GlobalAveragePooling2D(),
        ], name="Feature_Extractor")
        
        encoded_frames = layers.TimeDistributed(feature_extractor)(seq_inputs)
        temporal_features = layers.LSTM(64, return_sequences=False)(encoded_frames)
        
        # Branch 2: Metadata (Time cyclics + Historical Gen)
        m = layers.Dense(32, activation="gelu")(meta_inputs)
        m = layers.BatchNormalization()(m)
        
        # Fusion Head
        merged = layers.Concatenate()([temporal_features, m])
        z = layers.Dense(128, activation="gelu")(merged)
        z = layers.Dropout(0.3)(z)
        z = layers.Dense(64, activation="gelu")(z)
        
        outputs = layers.Dense(1, activation="relu", name="Megawatt_Prediction")(z)
        
        self.finetune_model = models.Model(inputs=[seq_inputs, meta_inputs], outputs=outputs, name="Phase2_Multimodal")
        return self.finetune_model

    def compile_pretrain(self, learning_rate=0.001):
        self.pretrain_model.compile(
            optimizer=optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
            loss="mse",
            metrics=["mae"]
        )

    def compile_finetune(self, learning_rate=0.0005, under_penalty=1.5, over_penalty=1.0):
        lr_schedule = WarmUpCosineDecay(learning_rate, warmup_steps=500, decay_steps=10000)
        self.finetune_model.compile(
            optimizer=optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
            loss=asymmetric_mse(under_penalty, over_penalty),
            metrics=["mae"]
        )