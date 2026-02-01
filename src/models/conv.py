import tensorflow as tf
from tensorflow.keras import layers, models

class ConvModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        """Defines the CNN architecture."""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),    
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, 3, activation='relu'),
            layers.Dropout(0.1),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='linear'),
        ])
        return model

    def compile(self, learning_rate=0.001):
        """Compiles the model."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, callbacks=None):
        """Trains the model."""
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
    def predict(self, X):
        return self.model.predict(X)
        
    def summary(self):
        self.model.summary()