import numpy as np

from music21 import instrument, note, stream

from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model.h5")

# Example notes
pitchnames = sorted(set([
    'C4', 'D4', 'E4', 'F4',
    'G4', 'A4', 'B4',
    'C5', 'D5', 'E5',
    'F5', 'G5', 'A5',
    'B5', 'C3', 'D3',
    'E3', 'F3', 'G3',
    'A3', 'B3', 'C6'
]))
# Mapping
note_to_int = dict(
    (note_name, number)
    for number, note_name in enumerate(pitchnames)
)

int_to_note = dict(
    (number, note_name)
    for number, note_name in enumerate(pitchnames)
)

# Random starting pattern
pattern = np.random.randint(
    0,
    len(pitchnames),
    20
)

prediction_output = []

# Generate notes
for note_index in range(100):

    prediction_input = np.reshape(
        pattern,
        (1, len(pattern), 1)
    )

    prediction_input = prediction_input / float(len(pitchnames))

    prediction = model.predict(
        prediction_input,
        verbose=0
    )

    index = np.argmax(prediction)

    result = int_to_note[index]

    prediction_output.append(result)

    pattern = np.append(pattern, index)

    pattern = pattern[1:]

# Create MIDI notes
offset = 0

output_notes = []

for pattern in prediction_output:

    new_note = note.Note(pattern)

    new_note.offset = offset

    new_note.storedInstrument = instrument.Piano()

    output_notes.append(new_note)

    offset += 0.5

# Create MIDI stream
midi_stream = stream.Stream(output_notes)

# Save MIDI file
midi_stream.write(
    'midi',
    fp='generated_music.mid'
)

print("Music Generated Successfully")