import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(curr_dir))


if 'dartfs-hpc' in curr_dir:
    # on discovery
    DATA_DIR = '/isi/music/yijing'
else:
    DATA_DIR = os.path.join(root_dir, "data")

# Midi feature
N_PITCH = 12
N_MEASURE = 16  # number of measures per sample/segment

# codec.event_type_range('tie')[-1], drum/program were excluded, original mt3 num_class=1513
N_VOCAB = 1257


# Feature dimension
D_RHYTHM = 3  # 3 status, active, rest, hold
D_HARMONY = N_PITCH * 2  # Tonic feature Major/Minor
D_NOTE = 12  # 98% of the entries has Note Count <= 11

# Audio feature
# sampling rate for mt3
SR = 16000
# duration of sample/segment in seconds, 1/SR * 128 * 512, determined by mt3
T_FRAME = 4.096
HOP_WIDTH = 128
NUM_MEL_BINS = 512

# fixed constants; add these to SpectrogramConfig before changing
FFT_SIZE = 2048
MEL_LO_HZ = 20.0

# mt3 tokenization
NUM_VELOCITY_BINS = 127


MT_DIRS = {"data": os.path.join(DATA_DIR, "maestro-v3.0.0"),
           "res": os.path.join(root_dir, "res", "mt"),
           "models": os.path.join(root_dir, "models", "mt"),
           "meta": os.path.join(DATA_DIR, "maestro-v3.0.0", "meta",
                                "maestro-v3.0.0.csv")}

GM_VAE_DIRS = {"data": os.path.join(DATA_DIR, "vgmidi"),
               "res": os.path.join(root_dir, "res", "gm_vae"),
               "models": os.path.join(root_dir, "models", "gm_vae")}
