import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from src.utils import get_logger

class PositionEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim):
        super().__init__()
        # Learnable embedding for each position in the sequence
        self.pos_emb = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        # Broadcast positions to match batch size
        return inputs + self.pos_emb(positions)

class TransformerBlock(layers.Layer):
    """
    A standard Transformer encoder block with Multi-Head Attention
    and a Feed-Forward Network (MLP), including Residual connections and LayerNorm.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        # FIX 1: key_dim should be size per head (embed_dim // num_heads) to match standard ViT architectures
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = models.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        # FIX 2: Implement Pre-LayerNorm. 
        # Normalization happens before Attention and FFN, which makes gradients highly stable.
        x1 = self.layernorm1(inputs)
        attn_output = self.att(x1, x1)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output
        
        x2 = self.layernorm2(out1)
        ffn_output = self.ffn(x2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output

class ConvTransformerModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.logger = get_logger()
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        # --- 1. CNN Stem ---
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        
        # --- 2. Transformer Preparation with Position Embedding ---
        shape = x.shape
        x = layers.Reshape((-1, shape[-1]))(x) 
        x = PositionEmbedding(sequence_length=2048, output_dim=64)(x)

        # --- 3. Transformer Encoder ---
        x = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128)(x)
        x = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128)(x)

        # --- 4. Head ---
        # FIX 3: Replace GlobalAveragePooling1D with Flatten.
        # This allows the final Dense layer to interpret explicit spatial locations, just like your CNN model does.
        x = layers.Flatten()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)

        return models.Model(inputs=inputs, outputs=outputs)

    def compile(self, learning_rate=0.001, weight_decay=1e-4):
        self.logger.info(f"Compiling with CosineDecay Scheduler...")
        
        lr_schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
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
            metrics=['mae', 'mse']
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, callbacks=None):
        self.logger.info("Starting ConvTransformer training...")
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=False, 
            verbose=1
        )