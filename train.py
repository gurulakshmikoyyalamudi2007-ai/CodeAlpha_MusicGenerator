import glob
import numpy as np

from music21 import converter, note, chord

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

# Store notes
notes = []

# Read MIDI files
for file in glob.glob("dataset/*.mid"):

    try:
        midi = converter.parse(file)

    except Exception as e:

        print("Error reading:", file)
        print(e)

        continue

    print("Parsing:", file)

    notes_to_parse = midi.flatten().notes

    for element in notes_to_parse:

        if isinstance(element, note.Note):
            notes.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

print("Total Notes:", len(notes))

# Stop if dataset too small
if len(notes) < 50:
    print("Not enough notes in dataset.")
    exit()

# Create pitch names
pitchnames = sorted(set(notes))

# Map notes to integers
note_to_int = dict(
    (note_name, number)
    for number, note_name in enumerate(pitchnames)
)

sequence_length = 20

network_input = []
network_output = []

# Create sequences
for i in range(0, len(notes) - sequence_length):

    sequence_in = notes[i:i + sequence_length]

    sequence_out = notes[i + sequence_length]

    network_input.append(
        [note_to_int[char] for char in sequence_in]
    )

    network_output.append(
        note_to_int[sequence_out]
    )

n_patterns = len(network_input)

print("Total Patterns:", n_patterns)

# Reshape input
network_input = np.reshape(
    network_input,
    (n_patterns, sequence_length, 1)
)

# Normalize
network_input = network_input / float(len(pitchnames))

# One-hot encode output
network_output = to_categorical(
    network_output,
    num_classes=len(pitchnames)
)

# Build model
model = Sequential()

model.add(
    LSTM(
        256,
        input_shape=(
            network_input.shape[1],
            network_input.shape[2]
        ),
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(LSTM(256))

model.add(Dense(128))

model.add(Dropout(0.3))

model.add(Dense(
    len(pitchnames),
    activation='softmax'
))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam'
)

# Train model
model.fit(
    network_input,
    network_output,
    epochs=20,
    batch_size=64
)

# Save model
model.save("model.h5")

print("Training Complete")