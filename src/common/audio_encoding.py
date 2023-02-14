"""
Get encoding from mt3/t5 auto transcription model.
Require mt3.
"""

import functools
import os
import numpy as np
import tensorflow.compat.v2 as tf

import gin
import jax
import seqio
import t5
import t5x

from t5x import decoding
from mt3 import network
from mt3 import vocabularies
from mt3 import spectrograms
from mt3 import models
from mt3 import preprocessors

import sys
sys.path.append("..")
from common.constants import NUM_VELOCITY_BINS

# Steal from t5x, for encoding
import abc
import jax.numpy as jnp
from typing import Mapping, Optional, Tuple

import nest_asyncio
nest_asyncio.apply()

PyTreeDef = type(jax.tree_structure(None))


class Encoder(object):
    """
    Get encoding from t5x https://github.com/google-research/t5x/blob/main/t5x/models.py, only using encoder. Adapted from mt3 inference model: https://github.com/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb.
    """

    def __init__(self, checkpoint_path, model_dir, model_type='ismir2021'):

        # Model Constants.
        self.inputs_length = 512

        gin_files = [os.path.join(model_dir, 'mt3/gin/model.gin'),
                     os.path.join(model_dir, f'mt3/gin/{model_type}.gin')]

        self.batch_size = 8
        self.outputs_length = 1024
        self.sequence_length = {'inputs': self.inputs_length,
                                'targets': self.outputs_length}

        self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)

        # Build Codecs and Vocabularies.
        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(
                num_velocity_bins=NUM_VELOCITY_BINS))
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
        self.output_features = {
            'inputs': seqio.ContinuousFeature(dtype=tf.float32, rank=2),
            'targets': seqio.Feature(vocabulary=self.vocabulary),
        }

        # Create a T5X model.
        self._parse_gin(gin_files)
        self.model = self._load_model()

        # Restore from checkpoint.
        self.restore_from_checkpoint(checkpoint_path)

    @property
    def input_shapes(self):
        return {
            'encoder_input_tokens': (self.batch_size, self.inputs_length),
            'decoder_input_tokens': (self.batch_size, self.outputs_length)
        }

    def _parse_gin(self, gin_files):
        """Parse gin files used to train the model."""
        gin_bindings = [
            'from __gin__ import dynamic_registration',
            'from mt3 import vocabularies',
            'VOCAB_CONFIG=@vocabularies.VocabularyConfig()',
            'vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS'
        ]

        with gin.unlock_config():
            gin.parse_config_files_and_bindings(
                gin_files, gin_bindings, finalize_config=False)

    def _load_model(self):
        """Load up a T5X `Model` after parsing training gin config."""
        model_config = gin.get_configurable(network.T5Config)()
        module = network.Transformer(config=model_config)

        return models.ContinuousInputsEncoderDecoderModel(
            module=module,
            input_vocabulary=self.output_features['inputs'].vocabulary,
            output_vocabulary=self.output_features['targets'].vocabulary,
            optimizer_def=t5x.adafactor.Adafactor(
                decay_rate=0.8, step_offset=0),
            input_depth=spectrograms.input_depth(self.spectrogram_config))

    def restore_from_checkpoint(self, checkpoint_path):
        """Restore training state from checkpoint, resets self._predict_fn()."""
        train_state_initializer = t5x.utils.TrainStateInitializer(
            optimizer_def=self.model.optimizer_def,
            init_fn=self.model.get_initial_variables,
            input_shapes=self.input_shapes,
            partitioner=self.partitioner)

        restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
            path=checkpoint_path, mode='specific', dtype='float32')

        train_state_axes = train_state_initializer.train_state_axes
        self._encode_fn = self._get_encode_fn(train_state_axes)
        self._train_state = train_state_initializer.from_checkpoint_or_scratch(
            [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0))

    @abc.abstractmethod
    def encode(self,
               params: PyTreeDef,
               batch: Mapping[str, jnp.ndarray],
               rng: Optional[jax.random.KeyArray] = None,
               return_all_decodes: bool = False,
               num_decodes: int = 1,
               prompt_with_targets: bool = False
               ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:

        inputs = batch['encoder_input_tokens']

        encoded_inputs = decoding.flat_batch_beam_expand(
            self.model.module.apply({'params': params},
                                    inputs,
                                    enable_dropout=False,
                                    method=self.model.module.encode),
            num_decodes)
        return encoded_inputs

    @functools.lru_cache()
    def _get_encode_fn(self, train_state_axes):
        """Generate a partitioned prediction function for decoding."""

        def partial_encode_fn(params, batch):
            return self.encode(params, batch)

        return self.partitioner.partition(
            partial_encode_fn,
            in_axis_resources=(
                train_state_axes.params,
                t5x.partitioning.PartitionSpec('data',), None),
            out_axis_resources=t5x.partitioning.PartitionSpec('data',)
        )

    def get_encoding(self, batch, seed=0):
        """Predict tokens from preprocessed dataset batch."""
        encoding = self._encode_fn(self._train_state.params, batch)
        return encoding

    def audio_to_dataset(self, audio):
        """Create a TF Dataset of spectrograms from input audio."""
        frames, frame_times = self._audio_to_frames(audio)
        return tf.data.Dataset.from_tensors({
            'inputs': frames,
            'input_times': frame_times,
        })

    def _audio_to_frames(self, audio):
        """Compute spectrogram frames from audio."""
        frame_size = self.spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]
        audio = np.pad(audio, padding, mode='constant')
        frames = spectrograms.split_audio(audio, self.spectrogram_config)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / \
            self.spectrogram_config.frames_per_second
        return frames, times

    def preprocess(self, ds):
        pp_chain = [
            functools.partial(
                t5.data.preprocessors.split_tokens_to_inputs_length,
                sequence_length=self.sequence_length,
                output_features=self.output_features,
                feature_key='inputs',
                additional_feature_keys=['input_times']),
            # Cache occurs here during training.
            preprocessors.add_dummy_targets,
            functools.partial(
                preprocessors.compute_spectrograms,
                spectrogram_config=self.spectrogram_config)
        ]
        for pp in pp_chain:
            ds = pp(ds)
        return ds

    def __call__(self, audio):
        """Infer note sequence from audio samples.

        Args:
        audio: 1-d numpy array of audio samples (16kHz) for a single example.

        Returns:
        A note_sequence of the transcribed audio.
        """
        ds = self.audio_to_dataset(audio)
        ds = self.preprocess(ds)

        model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
            ds, task_feature_lengths=self.sequence_length)
        model_ds = model_ds.batch(self.batch_size)

        inferences = (tokens for batch in model_ds.as_numpy_iterator()
                      for tokens in self.get_encoding(batch))

        encodings = []
        for item in inferences:
            encodings.append(item)

        return np.asarray(encodings)


if __name__ == "__main__":
    import librosa

    model_type = "ismir2021"  # Auto transcription model, single track with velocity

    model_dir = '/dartfs-hpc/rc/home/q/f004kkq/mt3'
    checkpoint_path = os.path.join(model_dir, f'checkpoints/{model_type}/')

    # Load audio encoder
    audio_encoder = Encoder(checkpoint_path, model_dir)
    print("Audio Encoder Loaded")

    # Load audio
    SAMPLE_RATE = 16000
    fname = '/dartfs-hpc/rc/home/q/f004kkq/ECoG/Stimulus/K448-90s.wav'
    audio, _ = librosa.load(fname, sr=SAMPLE_RATE)  # return audio, sr
    print('Load Audio Complete')

    encodings = audio_encoder(audio)
    # [n_sample, 512, 512]
    print('Encoding shape', encodings.shape)

    np.save('mt3_encoding', encodings)
