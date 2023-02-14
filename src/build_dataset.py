"""Preprocess dataset
Truncate the midi/audio to segment with duration of 4.096 s, which is determined by mt3 models,  extract features and save the output to the following files:

1. midi token (mt3) -> `midi_token.npz`
2. audio encoding (mt3) -> `audio_encoding.npz`
3. Harmonic/Tonal feature, Rhythmic feature, Note Count, arousal, valence -> `feature.npz`
4. excluded entries -> `excluded_entry.csv`

Returns:
    packed_output (tuple): (midi token, audio encoding, note, rhythm, harmony, valence, arousal)
    excluded_entry (pandas.DataFrame): Entries excluded from the preprocessed data

Usage:
    python build_dataset.py -d [Path to metadata .csv files] [-l]
"""
import os
import librosa
import pretty_midi
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys  # Todo: fix this
sys.path.append("..")

from common.utils import slice_midi
from common.tokenization import tokenize_pm
from common.audio_encoding import Encoder
from common.feature import FeatureExtract, note_coverage
from common.constants import GM_VAE_DIRS, N_MEASURE, SR, T_FRAME

# Constants

# Paths
model_type = "ismir2021"  # Auto transcription model, single track with velocity
model_dir = '/dartfs-hpc/rc/home/q/f004kkq/mt3'
checkpoint_path = os.path.join(model_dir, f'checkpoints/{model_type}/')

# Meta
index_dir = os.path.join(GM_VAE_DIRS["data"], "index")

# Load audio encoder
audio_encoder = Encoder(checkpoint_path, model_dir)
print("Audio Encoder Loaded")
# frame_size = audio_encoder.inputs_length * audio_encoder.spectrogram_config.hop_width
frame_size = int(SR * T_FRAME)


def main(metadata_fname, t_frame=T_FRAME, high_level=True, max_token_len=1024):
    """

    Args:
        metadata_fname(str): _description_
        high_level(bool, optional): Load valence / arousal annotations.
        max_token_len(int, optional): Max midi token sequence length.
    """
    # MIDI feature time scale
    t_scale = T_FRAME / N_MEASURE

    data_dir = GM_VAE_DIRS["data"]

    data = pd.read_csv(metadata_fname)
    prev = ""

    tokens = []
    audio_encoding = []
    note, rhythm, harmony = [], [], []
    valence, arousal = [], []
    excluded_entry_idx = []

    for i, row in tqdm(data.iterrows(), total=len(data)):

        # Avoid repeatedly loading midi if meta data sorted by midi/audio filename
        if row['midi_filename'] != prev:
            prev = row['midi_filename']  # update prev

            midi_fname = os.path.join(data_dir, prev)
            pm = pretty_midi.PrettyMIDI(midi_fname)

            audio_fname = os.path.join(data_dir, row['audio_filename'])
            audio, _ = librosa.load(audio_fname, sr=SR)

        t_start, t_end = row['start'], row['end']

        # Encoding audio slice (input)
        start_idx, end_idx = int(t_start * SR), int(t_end * SR)
        # Manually clip the audio_slice to `frame_size - 1`, otherwise audio_encoder generates two encoding sequences: 1st sequence enocding frame_size - 1 time points and 2nd encoding 1 time point.
        end_idx = min(end_idx, start_idx + frame_size - 1)
        audio_slice = audio[start_idx: end_idx]
        tmp_encoding = audio_encoder(audio_slice)
        audio_encoding.append(tmp_encoding)

        # Tokenize MIDI slice (target)
        pm_slice = slice_midi(pm, t_start, t_end)
        token = get_token(pm_slice)

        if not len(token) or len(token) > max_token_len:
            excluded_entry_idx.append(i)
            continue

        tokens.append(token)

        # Note Count/Rhythm/Harmonic Feature
        feat = FeatureExtract(pm_slice, n_measure=N_MEASURE, t_scale=t_scale)
        feat.low_level_feat()

        rhythm.append(feat.rhythm)
        note.append(feat.note)
        harmony.append(feat.harmony)

        # Valence/Arousal
        if high_level:
            valence.append(row['valence'])
            arousal.append(row['arousal'])

    packed_output = (tokens, audio_encoding,
                     note, rhythm, harmony,
                     valence, arousal)

    excluded_entry = data.iloc[excluded_entry_idx]

    # Save output
    msg = "{} saved to {}."
    # prefix = os.path.basename(metadata_fname).split(".")[0]
    output_dir = os.path.join(data_dir, "data", "prefix")
    os.makedirs(output_dir, exist_ok=True)

    # Save output
    fname = os.path.join(output_dir, "feature")
    np.savez_compressed(fname, **{"note": note,
                                  "rhythm": rhythm,
                                  "harmony": harmony,
                                  "valence": valence,
                                  "arousal": arousal})
    print(msg.format("feature", fname))

    fname = os.path.join(output_dir, "midi_token")
    tokens = np.asanyarray(tokens, dtype=object)
    np.savez_compressed(fname, **{"midi_token": tokens})
    print(msg.format("midi_token", fname))

    fname = os.path.join(output_dir, "audio_encoding")
    np.savez_compressed(fname, **{"audio_encoding": np.vstack(audio_encoding)})
    print(msg.format("audio_encoding", fname))

    fname = os.path.join(output_dir, "excluded_entry.csv")
    excluded_entry.to_csv(fname, index=False)

    return packed_output, excluded_entry


def get_token(pm, thresh=0.75):
    notes = sum([i.notes for i in pm.instruments], [])

    # If there is less than one note
    if not len(notes):
        return []

    # If the input is too sparse
    is_sparse = ((note_coverage(pm) / notes[-1].end) < thresh)
    if is_sparse:
        return []

    token = tokenize_pm(pm)
    return token


if __name__ == "__main__":

    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="index_dir", type=str,
                        default=index_dir, help="Path to metadata")
    parser.add_argument("-t", dest="t_frame", type=float,
                        default=T_FRAME, help="Resample frame size in seconds.")
    parser.add_argument("-l", dest="low_level_only",
                        default=False, action="store_true", help="Load arousal/valence annotations.")

    args = parser.parse_args()

    high_level = (not args.low_level_only)

    for fname in ["train.csv", "valid.csv", "test.csv"]:
        fname = os.path.join(args.index_dir, fname)
        print(f"Build dataset for {fname}.")

        t0 = time.time()
        _, _ = main(fname, args.t_frame, high_level)

        t1 = (time.time() - t0) / 60
        print(f"Finished in {t1:4f} min.")
        print("---------------------------")
