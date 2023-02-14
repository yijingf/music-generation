""" midi (note sequence) tokenization.
"""
import note_seq
import numpy as np

import sys  # Todo: fix this
sys.path.append("..")

from mt3_utils import vocabularies
from mt3_utils import note_sequences
from mt3_utils import run_length_encoding
from common.constants import NUM_VELOCITY_BINS

# Constants
codec = vocabularies.build_codec(
    vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=NUM_VELOCITY_BINS))
encoding_spec = note_sequences.NoteEncodingSpec

unk_id = 2
eos_id = 1
pad_id = 0

num_special_tokens = 3
base_vocab_size = 1517

def rle_shift(events):
    """Run length encoding: compress time shift tokens.

    Args:
        events (list): _description_

    Returns:
        list: _description_
    """
    shift_steps, total_shift_steps = 0, 0
    output = []

    for event in events:
        if codec.is_shift_event_index(event):
            shift_steps += 1
            total_shift_steps += 1

        else:
            if shift_steps > 0:
                shift_steps = total_shift_steps
                while shift_steps > 0:
                    output_steps = min(codec.max_shift_steps, shift_steps)
                    output.append(output_steps)
                    shift_steps -= output_steps
            output.append(event)

    return output


def tokenize(ns):
    """Tokenize note sequence.

    Args:
        ns (NoteSequence): _description_

    Returns:
        list: tokens
    """
    event_times, event_values = (
        note_sequences.note_sequence_to_onsets_and_offsets(ns))

    frame_times = np.arange(0, ns.total_time, step=1 / codec.steps_per_second)

    events, _, _, _, _ = run_length_encoding.encode_and_index_events(
        state=None,
        event_times=event_times,
        event_values=event_values,
        encode_event_fn=note_sequences.note_event_data_to_events,
        codec=codec,
        frame_times=frame_times)

    tokens = rle_shift(events)
    return tokens


def tokenize_pm(pm):
    """Tokenize pretty midi. 
    Todo: remove this function once all pm coverted to ns.

    Args:
        pm (pretty_midi.PrettyMIDI): _description_

    Returns:
        list: tokens
    """
    ns = note_seq.midi_to_note_sequence(pm)
    return tokenize(ns)


def trim_eos(tokens):
    """
    For only one sequence!
    """
    tokens = np.array(tokens, np.int32)
    if vocabularies.DECODED_EOS_ID in tokens:
        tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
    return tokens


def decode_token(tokens, start_time=0):

    decoding_state = encoding_spec.init_decoding_state_fn()

    invalid_ids, dropped_events = run_length_encoding.decode_events(
        state=decoding_state,
        tokens=tokens,
        start_time=start_time, max_time=None,
        codec=codec, decode_event_fn=encoding_spec.decode_event_fn)

    # ns = note_sequences.flush_note_decoding_state(decoding_state)
    ns = encoding_spec.flush_decoding_state_fn(decoding_state)

    return ns, invalid_ids, dropped_events


def postprocess(ids):

    # Replace ids > base_vocab_size with unk_id (unknown id).
    ids = np.where(np.less(ids, base_vocab_size), ids, unk_id)
    
    # Replace everything after the first eos_id with pad_id.
    equal = (ids == eos_id)
    # shift equal to exclude the first eos_id
    equal = np.pad(equal[:, :-1], ((0, 0),(1,0)), 'constant', constant_values=False)
    after_eos = np.cumsum(equal, axis=-1).astype(bool)

    ids = np.where(after_eos, pad_id, ids)
    
    eos_and_after = np.cumsum(ids == eos_id, axis=-1).astype(bool)

    ids = np.where(eos_and_after,
                   vocabularies.DECODED_EOS_ID,
                   np.where(
                       np.logical_and(
                           np.greater_equal(ids, num_special_tokens),
                           np.less(ids, base_vocab_size)),
                       ids - num_special_tokens,
                       vocabularies.DECODED_INVALID_ID))
    
    return ids

# # Adapted from mt3
# import functools
# from mt3_utils import event_codec
# Validate Data Type
# from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TypeVar
# S = TypeVar('S')
# T = TypeVar('T')
# CombineExamplesFunctionType = Callable[[Sequence[Mapping[str, Any]]],
#                                        Mapping[str, Any]]
# def event_predictions_to_ns(
#         predictions: Sequence[Mapping[str, Any]],
#         codec: event_codec.Codec,
#         encoding_spec: note_sequences.NoteEncodingSpecType) -> Mapping[str, Any]:
#     """Convert a sequence of predictions to a combined NoteSequence."""
#     ns, total_invalid_events, total_dropped_events = decode_and_combine_predictions(
#         predictions=predictions,
#         init_state_fn=encoding_spec.init_decoding_state_fn,
#         begin_segment_fn=encoding_spec.begin_decoding_segment_fn,
#         decode_tokens_fn=functools.partial(
#             run_length_encoding.decode_events,
#             codec=codec,
#             decode_event_fn=encoding_spec.decode_event_fn),
#         flush_state_fn=encoding_spec.flush_decoding_state_fn)

#     # Also concatenate raw inputs from all predictions.
#     # sorted_predictions = sorted(
#     # predictions, key=lambda pred: pred['start_time'])
#     # raw_inputs = np.concatenate(
#     # [pred['raw_inputs'] for pred in sorted_predictions], axis=0)
#     # start_times = [pred['start_time'] for pred in sorted_predictions]

#     return ns


# def decode_and_combine_predictions(
#         predictions: Sequence[Mapping[str, Any]],
#         init_state_fn: Callable[[], S],
#         begin_segment_fn: Callable[[S], None],
#         decode_tokens_fn: Callable[[S, Sequence[int], int, Optional[int]],
#                                    Tuple[int, int]],
#         flush_state_fn: Callable[[S], T]) -> Tuple[T, int, int]:
#     """Decode and combine a sequence of predictions to a full result.
#     Args:
#     predictions: List of predictions, each of which is a dictionary containing
#         estimated tokens ('est_tokens') and start time ('start_time') fields.
#     init_state_fn: Function that takes no arguments and returns an initial
#         decoding state.
#     begin_segment_fn: Function that updates the decoding state at the beginning
#         of a segment.
#     decode_tokens_fn: Function that takes a decoding state, estimated tokens
#         (for a single segment), start time, and max time, and processes the
#         tokens, updating the decoding state in place. Also returns the number of
#         invalid and dropped events for the segment.
#     flush_state_fn: Function that flushes the final decoding state into the result.

#     Returns:
#     result: The full combined decoding.
#     total_invalid_events: Total number of invalid event tokens across all predictions.
#     total_dropped_events: Total number of dropped event tokens across all predictions.
#     """
#     sorted_predictions = sorted(
#         predictions, key=lambda pred: pred['start_time'])

#     state = init_state_fn()
#     total_invalid_events = 0
#     total_dropped_events = 0

#     for pred_idx, pred in enumerate(sorted_predictions):

#         begin_segment_fn(state)

#         # Depending on the audio token hop length, each symbolic token could be
#         # associated with multiple audio frames. Since we split up the audio frames
#         # into segments for prediction, this could lead to overlap. To prevent
#         # overlap issues, ensure that the current segment does not make any
#         # predictions for the time period covered by the subsequent segment.

#         max_decode_time = None
#         if pred_idx < len(sorted_predictions) - 1:
#             max_decode_time = sorted_predictions[pred_idx + 1]['start_time']

#         invalid_events, dropped_events = decode_tokens_fn(
#             state, pred['est_tokens'], pred['start_time'], max_decode_time)

#         total_invalid_events += invalid_events
#         total_dropped_events += dropped_events

#     return flush_state_fn(state), total_invalid_events, total_dropped_events


# def postprocess(tokens, start_time=0, steps_per_second=100):
#     tokens = trim_eos(tokens)
#     # Round down to nearest symbolic token step.
#     start_time -= start_time % (1 / steps_per_second)
#     return {
#         'est_tokens': tokens,
#         'start_time': start_time,
#         # Internal MT3 code expects raw inputs, not used here.
#         'raw_inputs': []
#     }


if __name__ == "__main__":

    ns = note_seq.NoteSequence()
    ns.notes.add(start_time=1.0, end_time=3.0,
                 pitch=70, velocity=1)

    ns.notes.add(start_time=0.5, end_time=4.0,
                 pitch=62, velocity=127)
    ns.total_time = ns.notes[-1].end_time

    tokens = tokenize(ns)
    decoded_ns = decode_token(tokens)

    # Todo: unit test
    note_seq.note_sequence_to_midi_file(decoded_ns[0], 'output.mid')

    # predictions = [postprocess(events)]
    # decoded_ns = event_predictions_to_ns(predictions, codec, encoding_spec)
