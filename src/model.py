# ==========================================================
# SMARTBRIDGE AI LEARNING ASSISTANT
# TASK 3 : BiLSTM MODEL
# ==========================================================

import tensorflow as tf

# ==========================================================
# CONFIGURATION
# ==========================================================

VOCAB_SIZE = 30000
MAX_SEQUENCE_LENGTH = 80
EMBEDDING_DIM = 128
LSTM_UNITS = 128
NUM_CLASSES = 4

# ==========================================================
# FOCAL LOSS
# ==========================================================

def focal_loss(gamma=2.0, alpha=0.25):

    def loss(y_true, y_pred):

        y_true = tf.one_hot(
            tf.cast(y_true, tf.int32),
            depth=NUM_CLASSES
        )

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7
        )

        cross_entropy = -y_true * tf.math.log(y_pred)

        weight = alpha * tf.pow(1 - y_pred, gamma)

        focal = weight * cross_entropy

        return tf.reduce_mean(
            tf.reduce_sum(focal, axis=1)
        )

    return loss


# ==========================================================
# BUILD MODEL
# ==========================================================

def build_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_SEQUENCE_LENGTH
        ),

        tf.keras.layers.Bidirectional(

            tf.keras.layers.LSTM(
                LSTM_UNITS,
                dropout=0.2,
                recurrent_dropout=0.2
            )

        ),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )

    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss=focal_loss(),

        metrics=["accuracy"]

    )

    return model


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    model = build_model()

    model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))

    print("=" * 60)
    print("BiLSTM MODEL SUMMARY")
    print("=" * 60)

    model.summary()