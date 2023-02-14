import music21
import numpy as np
from io import BytesIO

# Constants
# Todo: a better way to store them
import sys
sys.path.append("..")
from common.constants import N_PITCH, N_MEASURE

KEY_IDX = {
    "C-": 11, "C": 0, "C#": 1, "D-": 1, "D": 2, "D#": 3, "E-": 3, "E": 4, "E#": 5,
    "F-": 4, "F": 5, "F#": 6, "G-": 6, "G": 7, "G#": 8, "A-": 8, "A": 9, "A#": 10,
    "B-": 10, "B": 11, "B#": 0
}


def note_coverage(pm):
    """return note duration in total

    Args:
        pm (PrettyMIDI): PrettyMIDI

    Returns:
        float: the sum of note duration time
    """
    duration = 0

    # Get notes from all instruments and sorted by starting time
    notes = sum([i.notes for i in pm.instruments], [])
    notes = sorted(notes, key=lambda x: x.start)

    if not len(notes):
        return duration

    duration = notes[0].end - notes[0].start
    for i, note in enumerate(notes[1:]):
        if note.start > notes[i].end:
            duration += note.end - note.start
        else:
            duration += max(note.end - notes[i].end, 0)

    return duration


def get_chroma(pr):
    """
    Get chromagram from piano roll
    """
    t, _ = pr.shape
    chroma = np.zeros((t, N_PITCH))
    for note in range(N_PITCH):
        chroma[:, note] = np.sum(pr[:, note::N_PITCH], axis=1)
    return chroma


def get_avg_velocity(pr):
    """Compute Average velocity at each tick

    Args:
        pr (np.array): piano roll

    Returns:
        np.array: average velocity
    """

    v_sum = np.sum(pr, axis=1)
    n_note = np.sum(pr.astype(bool), axis=1)
    n_note[np.argwhere(n_note == 0)] = 1
    velocity = (v_sum / n_note).astype(int)
    return velocity


def get_harmony(pm, is_one_hot=False, thresh=0.1):
    """Get tonal feature, i.e. the probability of each tone.

    Args:
        pm (pretty_midi.PrettyMIDI): prettyMIDI slice
        is_one_hot (bool, optional): _description_. Defaults to False.
        thresh (float, optional): _description_. Defaults to 0.1.

    Returns:
        np.array: harmonic vector in (,24)
    """
    # convert pretty_midi to string
    with BytesIO() as f:
        pm.write(f)
        f.seek(0)
        midi_string = f.read()

    harmony = np.zeros(24,)
    score = music21.midi.translate.midiStringToStream(midi_string)
    key = score.analyze('key')

    idx = KEY_IDX[key.tonic.name] + N_PITCH * (key.mode == "minor")
    if is_one_hot:

        # One hot encoding of the most likely key
        harmony[idx] = 1

    else:
        # return the probability of each key
        harmony[idx] = key.correlationCoefficient

        for tmp_key in key.alternateInterpretations:
            idx = KEY_IDX[tmp_key.tonic.name] + \
                N_PITCH * (tmp_key.mode == "minor")
            harmony[idx] = tmp_key.correlationCoefficient

        # Thresholding
        harmony[harmony < thresh] = 0

    return harmony


class FeatureExtract:

    def __init__(self, pm, pr=None, n_measure=N_MEASURE, t_scale=0.5):
        """_summary_

        Args
            pm (pretty_midi.PrettyMIDI): sliced pretty midi
            pr (np.array, optional): sliced piano roll
            n_measure: number of measure per sample/segment
            t_scale: second per measure
        """

        self.pm = pm
        self.pr = pr

        if pr is None:
            self.get_melody(n_measure, t_scale)
        else:
            self.get_melody_pr()

    def low_level_feat(self):
        self.rhythm = self.melody_to_rhythm()
        self.note = self.count_note()
        self.harmony = get_harmony(self.pm)
        # self.velocity = get_avg_velocity(pr)
        # self.chroma = get_chroma(pr)

    def get_melody(self, n_measure, t_scale):
        """_summary_

        Args:
            n_measure (_type_): _description_
            t_scale (_type_): _description_
        """
        notes = sum([i.notes for i in self.pm.instruments], [])
        self.melody = [[] for _ in range(n_measure)]

        for note in notes:
            start_idx = np.floor(note.start / t_scale).astype(int)
            end_idx = min(np.ceil(note.end / t_scale).astype(int), n_measure)

            # `end_idx` should <= n_measure naturally, however, since note.end could be slightly larger than `t_frame` (4.096) because of tick scales, float, etc.. Manually cut-off end_idx > n_measure.

            for i in range(start_idx, end_idx):
                self.melody[i] += [note.pitch]

    def get_melody_pr(self):
        """
        Get melody and velocity for each tick from piano roll
        """
        note_idx = np.nonzero(self.pr)
        t, _ = self.pr.shape

        self.melody = [[] for _ in range(t)]

        for t_idx, pitch in zip(*note_idx):
            self.melody[t_idx].append(pitch)

    def melody_to_rhythm(self):
        """
        Extract rhythm from melody using three states 0,1,2 to represent rest, activate, hold.
        """
        rhythm = []
        prev_note = []

        for curr_note in self.melody:
            # rest
            if not len(curr_note):
                rhythm.append(0)
            # hold
            elif set(curr_note).issubset(prev_note):
                rhythm.append(2)
            else:
                rhythm.append(1)
            prev_note = curr_note
        return rhythm

    def count_note(self):
        return np.array([len(note) for note in self.melody])

    def get_avg_velocity(self, piano_roll):
        """
        Average velocity at each tick
        """
        v_sum = np.sum(piano_roll, axis=1)
        n_note = np.sum(piano_roll.astype(bool), axis=1)
        n_note[np.argwhere(n_note == 0)] = 1
        self.velocity = v_sum / n_note
        self.velocity.astype(int)


"""
For valence/arousal.
Don't need these functions for now.
"""
# import wave
# import math

# def get_duration(audio_fname):
#     """_summary_
#     """
#     seconds = 0
#     with wave.open(audio_fname, 'r') as f:
#         frames = f.getnframes()
#         rate = f.getframerate()
#         seconds = frames / rate
#     return seconds


# def get_measures(midi_fname):
#     """count number of bars from midi

#     Args:
#         (int):
#     """
#     midi = music21.midi.MidiFile()

#     try:
#         midi.open(midi_filename)
#         midi.read()
#         midi.close()
#     except:
#         print("Skipping file: Midi file has bad formatting")
#         return 0

#     midi_stream = music21.midi.translate.midiFileToStream(midi)
#     measures = math.ceil(midi_stream.duration.quarterLength/sample_freq)

#     return measures
