import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH

OUTPUT_UNITS = 38  # Will be dynamically set by vocabulary size
NUM_UNITS = [256]  # List to allow multiple LSTM layers if needed
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"


def build_model(vocabulary_size, num_units, loss, learning_rate):
    """
    Builds and compiles the LSTM model for melody generation.
    
    :param vocabulary_size (int): Size of the vocabulary (unique notes/rests).
    :param num_units (list of int): List of units for hidden layers.
    :param loss (str): Loss function type.
    :param learning_rate (float): Learning rate.
    :return: Compiled Keras model.
    """
    
    # Input shape: (sequence_length, vocabulary_size)
    input_layer = keras.layers.Input(shape=(SEQUENCE_LENGTH, vocabulary_size))
    
    # LSTM layers
    x = input_layer
    for units in num_units:
        x = keras.layers.LSTM(units, return_sequences=False)(x)
        x = keras.layers.Dropout(0.3)(x)

    # Output layer (softmax)
    output_layer = keras.layers.Dense(vocabulary_size, activation="softmax")(x)

    # Build model
    model = keras.Model(input_layer, output_layer)

    # Compile the model
    model.compile(loss=loss, 
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    model.summary()
    return model


def train_model(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    """
    Trains the LSTM model and saves it to a file.
    
    :param output_units (int): Number of output units.
    :param num_units (list of int): Number of hidden units in LSTM layers.
    :param loss (str): Loss function to use.
    :param learning_rate (float): Learning rate for optimizer.
    """
    # Generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # Dynamically determine the vocabulary size from the inputs shape
    vocabulary_size = inputs.shape[-1]

    # Build the model
    model = build_model(vocabulary_size, num_units, loss, learning_rate)

    # Train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train_model()
