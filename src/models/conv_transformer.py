import os
import math
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils import get_logger

class StaticPatchWeighting(layers.Layer):
    """
    Applies a fixed, learnable weight to each patch position.
    Good for prioritizing specific spatial regions.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        seq_len = input_shape[1]
        self.patch_weights = self.add_weight(
            shape=(1, seq_len, 1), 
            initializer="ones",    
            trainable=True,
            name="static_patch_weights"
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.patch_weights


class DynamicPatchWeighting(layers.Layer):
    """
    Calculates weights dynamically based on the patch's content.
    Acts as a lightweight spatial attention mechanism.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention_dense = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        weights = self.attention_dense(inputs)
        return inputs * weights


class PositionEmbedding2D(layers.Layer):
    """
    2D Positional Embedding.
    Generates separate embeddings for Height (Y) and Width (X) to retain 
    spatial grid awareness before flattening into a 1D sequence.
    """
    def __init__(self, height, width, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.height_emb = layers.Embedding(input_dim=height, output_dim=embed_dim)
        self.width_emb = layers.Embedding(input_dim=width, output_dim=embed_dim)
        self.height = height
        self.width = width
        self.embed_dim = embed_dim

    def call(self, inputs):
        h_positions = tf.range(start=0, limit=self.height, delta=1)
        w_positions = tf.range(start=0, limit=self.width, delta=1)
        
        h_emb = self.height_emb(h_positions) # Shape: (H, embed_dim)
        w_emb = self.width_emb(w_positions)  # Shape: (W, embed_dim)
        
        # Broadcast to create a 2D grid: (H, W, embed_dim)
        h_emb = tf.expand_dims(h_emb, 1) # (H, 1, embed_dim)
        w_emb = tf.expand_dims(w_emb, 0) # (1, W, embed_dim)
        pos_emb_2d = h_emb + w_emb
        
        # Flatten the spatial dimensions to match the patch sequence: (H*W, embed_dim)
        pos_emb_2d = tf.reshape(pos_emb_2d, (-1, self.embed_dim))
        
        return inputs + pos_emb_2d

class PositionEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim):
        super().__init__()
        self.pos_emb = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        return inputs + self.pos_emb(positions)


class TransformerBlock(layers.Layer):
    """
    A standard Transformer encoder block with Multi-Head Attention
    and a Feed-Forward Network (MLP), including Residual connections and LayerNorm.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = models.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        x1 = self.layernorm1(inputs)
        attn_output = self.att(x1, x1)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output
        
        x2 = self.layernorm2(out1)
        ffn_output = self.ffn(x2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output


class WarmUpCosineDecay(optimizers.schedules.LearningRateSchedule):
    """
    Learning Rate Warmup.
    Linearly increases learning rate from 0 to the target over `warmup_steps`, 
    then decays following a cosine curve.
    """
    def __init__(self, initial_learning_rate, warmup_steps, decay_steps, alpha=0.01):
        super().__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.alpha = tf.cast(alpha, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Linear Warmup phase
        warmup_lr = self.initial_learning_rate * (step / self.warmup_steps)
        
        # Cosine Decay phase
        decay_step = tf.minimum(step - self.warmup_steps, self.decay_steps)
        cosine_decay = 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * decay_step / self.decay_steps))
        decayed_lr = (1.0 - self.alpha) * cosine_decay + self.alpha
        decayed_lr = self.initial_learning_rate * decayed_lr
        
        return tf.where(step < self.warmup_steps, warmup_lr, decayed_lr)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha
        }


class ConvTransformerModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.logger = get_logger()
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        # --- 1. CNN Stem ---
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x) 
        
        # --- 2. Transformer Preparation with 2D Position Embedding ---
        H, W, C = x.shape[1], x.shape[2], x.shape[3]
        
        x = layers.Reshape((-1, C))(x) 

        # x = StaticPatchWeighting()(x)
        # x = DynamicPatchWeighting()(x)
        
        # x = PositionEmbedding(sequence_length=2048, output_dim=64)(x)
        x = PositionEmbedding2D(height=H, width=W, embed_dim=C)(x)

        # --- 3. Transformer Encoder ---
        x = TransformerBlock(embed_dim=C, num_heads=4, ff_dim=128)(x)
        x = TransformerBlock(embed_dim=C, num_heads=4, ff_dim=128)(x)

        # --- 4. Head ---
        # x = layers.Flatten()(x) # This allows the final Dense layer to interpret explicit spatial locations
        x = layers.GlobalAveragePooling1D()(x) # prevent massive parameter explosion and overfitting
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(1, activation="linear")(x)

        return models.Model(inputs=inputs, outputs=outputs)

    def compile(self, learning_rate=0.001, weight_decay=1e-4):

        # self.logger.info(f"Compiling with CosineDecay Scheduler...")
        # lr_schedule = optimizers.schedules.CosineDecay(
        #     initial_learning_rate=learning_rate,
        #     decay_steps=10000, 
        #     alpha=0.01
        # )
        
        self.logger.info(f"Compiling with WarmUp + CosineDecay Scheduler...")
        lr_schedule = WarmUpCosineDecay(
            initial_learning_rate=learning_rate,
            warmup_steps=1000,   # Adjust based on your batch size / dataset size
            decay_steps=10000, 
            alpha=0.01
        )
        
        optimizer_fn = optimizers.AdamW(
            learning_rate=lr_schedule, 
            weight_decay=weight_decay
        )
        
        self.model.compile(
            optimizer=optimizer_fn, 
            loss=losses.MeanSquaredError(), 
            metrics=["mae", "mse"]
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, callbacks=None, checkpoint_dir="../data/models"):
        """Trains the model with shuffle=False and default callbacks."""
        self.logger.info("Starting ConvTransformer training...")
        
        # Default callbacks if none are provided
        if callbacks is None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                
            checkpoint_path = os.path.join(checkpoint_dir, "best_conv_transformer_model.weights.h5")
            
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", 
                    patience=10, 
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    filepath=checkpoint_path, 
                    monitor="val_loss", 
                    save_best_only=True, 
                    save_weights_only=True,
                    verbose=1
                )
            ]
            self.logger.info(f"Using default callbacks. Checkpoints will be saved to: {checkpoint_path}")

        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=False, 
            verbose=1
        )