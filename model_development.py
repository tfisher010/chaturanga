from tensorflow import keras
import numpy as np
import pickle
from utils import save_model


def train_model():
    """train and save a model using currently saved game data"""

    with open("artifacts/prepped_game_data.pkl", "rb") as f:
        Xy_pairs = pickle.load(f)

    X = np.array([x[0] for x in Xy_pairs])
    y = np.array([x[1] for x in Xy_pairs])

    # keras.utils.set_random_seed()

    model = keras.models.Sequential(
        [
            keras.layers.Dense(100, input_shape=(8, 8, 12)),
            keras.layers.Conv2D(
                filters=20,
                kernel_size=2,
                strides=(1, 1),
                activation="relu",
            ),
            keras.layers.Dense(
                10,
                activation="relu",
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x=X, y=y, epochs=1)

    return model
