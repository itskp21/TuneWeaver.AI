# TuneWeaver.AI

# Melody Generation with LSTM
This project focuses on generating musical melodies using a Long Short-Term Memory (LSTM) model. The pipeline includes data preprocessing, encoding, training, and melody generation.

# Features
Data Preprocessing:

1. Processes and encodes 1,000+ musical pieces from Kern format (.krn files).
2. Standardizes melodies by transposing to C major or A minor.
3. Filters out non-standard note durations to ensure data consistency.

# Dataset Creation:

1.Merges encoded songs into a single dataset with sequence delimiters.
2. Generates over 10,000 training sequences of length 64.

# Mapping and Encoding:

1.Develops a symbol-to-integer mapping for 50+ musical symbols.
2. Creates 100,000+ one-hot encoded sequences for LSTM training.

# Melody Generation:

1.Utilizes an LSTM model to generate 500+ unique melodies.
2.Adjustable temperature settings for varying levels of creativity.
3. MIDI Conversion:

Converts generated melodies into MIDI format for playback and further analysis.

# Usage
Preprocess Data: Load and encode musical pieces from Kern files.
Generate Dataset: Combine encoded songs into a single file and create mappings.
Train Model: Use the generated sequences to train an LSTM model.
Generate Melodies: Produce new melodies based on trained model and save as MIDI files.
