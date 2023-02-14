"""Helper functions for loading data and evaluation.
"""
import torch
import pretty_midi
import numpy as np

# Todo: fix this
import sys
sys.path.append("..")
from common.constants import N_PITCH


def subsequent_mask(size):
    """Mask out subsequent positions.

    Args:
        size (int): Size of mask. 

    Returns:
        torch.Tensor: Mask.
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def collate_skip_none(batch):
    """Skip none item in a batch, helper function for data loader.

    Args:
        batch (list): Might be list? a list of items. 

    Returns:
        torch.tensor
    """
    batch = filter(lambda item: item is not None, batch)
    return torch.utils.data.dataloader.default_collate(list(batch))


def pretty_midi_sort(pm):
    """
    Sort notes/control changes by time in place
    """
    for i in range(len(pm.instruments)):
        pm.instruments[i].notes = sorted(
            pm.instruments[i].notes, key=lambda note: note.start)
        pm.instruments[i].control_changes = sorted(
            pm.instruments[i].control_changes, key=lambda event: event.time)
    return


def slice_midi(pm, t_start, t_end, sorted=True, meta=True):
    """Slice midi given the starting and ending time.

    Args:
        midi (PrettyMIDI): pretty_midi loaded by pretty midi. Assume that notes are sorted by start time.
        t_start (float): starting time in second.
        t_end (float): ending time in second.
        sorted (Optional, bool): whether notes have been sorted. 

    Returns:
        PrettyMIDI: Sliced pretty_midi.
    """
    if not sorted:
        pretty_midi_sort(pm)

    if meta:
        # Initial tempo
        prev_tempo = [tempo for t, tempo in zip(*pm.get_tempo_changes())
                      if t <= t_start]
        initial_tempo = prev_tempo[-1]

        # Initialize pm
        pm_slice = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo,
                                          resolution=pm.resolution)

        # Todo: fix transfer tick scales
        start_tick = pm.time_to_tick(t_start)
        end_tick = pm.time_to_tick(t_end)

        n_change = len(pm._tick_scales)

        p = 0  # points to the last tempo change before start_tick
        while p < n_change and pm._tick_scales[p][0] < start_tick:
            p += 1
        q = p  # points to the last tempo change before end_tick
        while q < n_change and pm._tick_scales[q][0] < end_tick:
            q += 1

        tick_scales = pm._tick_scales[max(0, p - 1):q]
        pm_slice._tick_scales = list(map(lambda x: (max(0, int(x[0] - start_tick)), x[1]),
                                         tick_scales))

        # Initialize meta data
        # Todo: Modulize this
        # Time Signature
        prev_ts_obj = None
        for ts_obj in pm.time_signature_changes:
            if ts_obj.time < t_start:
                prev_ts_obj = pretty_midi.TimeSignature(numerator=ts_obj.numerator,
                                                        denominator=ts_obj.denominator,
                                                        time=0)
                continue
            if ts_obj.time > t_end:
                break
            new_ts_obj = pretty_midi.TimeSignature(numerator=ts_obj.numerator,
                                                   denominator=ts_obj.denominator,
                                                   time=ts_obj.time - t_start)
            pm_slice.time_signature_changes.append(new_ts_obj)
        if prev_ts_obj is not None:
            pm_slice.time_signature_changes.append(prev_ts_obj)

        prev_key_obj = None
        for key_obj in pm.key_signature_changes:
            if key_obj.time < t_start:
                prev_key_obj = pretty_midi.KeySignature(key_number=key_obj.key_number,
                                                        time=0)
                continue
            if key_obj.time > t_end:
                break
            new_key_obj = pretty_midi.KeySignature(key_number=key_obj.key_number,
                                                   time=key_obj.time - t_start)
            pm_slice.key_signature_changes.append(new_key_obj)

        if prev_key_obj is not None:
            pm_slice.key_signature_changes.append(prev_key_obj)

    # Looping through all instruments
    for orig_inst in pm.instruments:
        inst = pretty_midi.Instrument(program=orig_inst.program,
                                      is_drum=orig_inst.is_drum,
                                      name=orig_inst.name)
        for note in orig_inst.notes:
            if note.start < t_start or note.end < t_start:
                continue
            if note.start > t_end:
                break

            new_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch,
                                        start=max(0, note.start - t_start),
                                        end=min(t_end - t_start, note.end - t_start))
            inst.notes.append(new_note)

        for ctrl in orig_inst.control_changes:
            if ctrl.time >= t_start and ctrl.time < t_end:
                new_ctrl = pretty_midi.ControlChange(number=ctrl.number,
                                                     value=ctrl.value,
                                                     time=ctrl.time - t_start)
                inst.control_changes.append(new_ctrl)

        pm_slice.instruments.append(inst)
    return pm_slice


def padding(seq, max_len, pad_idx=0):
    """Pad a magenta performance encoding sequence to `max_len`. 

    Args:
        seq (list): Magenta performance encoding sequence. 
        max_len (int): Maximum length of the sequence.
        pad_idx (int): Padding index of magenta performance encoding. Defaults to 0.

    Returns:
        list: The padded sequence.
    """

    len_pad = max_len - len(seq)
    if len_pad >= 0:
        return np.concatenate((seq, [pad_idx for _ in range(len_pad)]))
    return np.concatenate(seq[:max_len])

# For Evaluation


def fill_zero(p, q, coeff=1e-16):
    """Fill zero entry with small number to avoid nan when calculating KL-divergence.

    Args:
        p (numpy.ndarray): Distribution p.
        q (numpy.ndarray): Distribution q.
        coeff (float, optional): Small number to replace zero. Defaults to 1e-16.

    Returns:
        numpy.ndarray, numpy.ndarray : Filled p and q.
    """

    idx = np.argwhere(q == 0)
    n_zeroitem = len(idx)
    q_rand = np.random.rand(n_zeroitem, 1) * coeff
    q[idx] = q_rand
    p[idx] = q_rand
    return p, q


def pad_dist(p, q):
    """Pad shorter distribution with zero to ensure p and q have same length.

    Args:
        p (numpy.ndarray): Distribution p.
        q (numpy.ndarray): Distribution q.

    Returns:
        numpy.ndarray, numpy.ndarray: Padded distribtution 1, padded distribution 2.
    """
    l_p = len(p) // N_PITCH
    l_q = len(q) // N_PITCH

    l = max(l_p, l_q)
    p = np.concatenate((p, np.zeros((l - l_p) * N_PITCH)))
    q = np.concatenate((q, np.zeros((l - l_q) * N_PITCH)))
    return p, q

# Speed and other performance


class PerformanceMeter(object):
    def __init__(self):
        self.batch_time = AverageMeter()
        self.per_sample_time = AverageMeter()
        self.data_time = AverageMeter()
        self.per_sample_data_time = AverageMeter()
        self.loss_meter = AverageMeter()
        self.per_sample_dnn_time = AverageMeter()

    def log_brief(self):
        msg = f"Loss: {self.loss_meter.avg:.4f}\t Time per Batch: {self.batch_time.avg:.2f}"
        print(msg, flush=True)

        return

    def reset_all(self):
        self.batch_time.reset()
        self.per_sample_time.reset()
        self.data_time.reset()
        self.per_sample_data_time.reset()
        self.loss_meter.reset()
        self.per_sample_dnn_time.reset()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.init_val()

    def init_val(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val *= 0
        self.avg *= 0
        self.sum *= 0
        self.count = 0

    def update(self, val, n=1):
        if val is not None:
            self.val = np.array(val)
            self.sum += self.val * n
            self.count += n
            self.avg = self.sum / self.count


# Convert frame_shift in ms to hop_length size
def frameshift2hoplength(frame_shift, sr):
    """Convert frame_shift in ms to hop_length size.

    Args:
        frame_shift (float): frame shift in ms.
        sr (int): Sampling frequency

    Returns:
        int: hop length size
    """
    hop_length = int(frame_shift / 1000 * sr)
    return np.power(2, int(np.log2(hop_length)))
