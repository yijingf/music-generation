# Dataset

## Maestro
The music auto transcription model, i.e. `fadernet/src/music_transformer` is trained Read more about Maestro [here](https://magenta.tensorflow.org/datasets/maestro).

## VGMIDI Dataset
[VGMIDI Dataset]((https://github.com/lucasnfe/vgmidi)) was annotated using Music Annotation Tool (https://github.com/lucasnfe/adl-music-annotation). Read more details about data annotation [here](https://github.com/lucasnfe/adl-music-annotation/blob/master/py/midi2annotation/midi2annotation.py).


## Rearrange
1. Move all MIDI files from `./labelled/midi` and `./unlabelled/midi` to `./midi`.
2. Original audio files were moved from `./labelled/audio` to `OriginalVGMIDI`. Re-Synthesize audio from all MIDI files, since the timing of the original audio files don't exactly match the corresponding MIDI files. 
3. The rest of the files were moved to `OriginalVGMIDI`.


### Annotations
`./vgmidi/labelled/annotations/vgmidi_raw_1/2.json`: annotations in original dataset. Each piece was annotated by around 30 human subjects according to a valence-arousal model of emotion.
* `pieces`:
    * `measures`: number of bars (assumption: all pieces are 4/4, I think, but actually not).
    * `duration`: audio duration in second.
* `annotations`:
    * `valence`/`arousal`: a list with length of `measures`.
    * `musicianship`: 1-5, the higher the better. We select annotations from subjects with musicianship >= 3.
    * other demographic data of the evaluators like age, gender.

### Preprocess
See `../src/preprocess/README.md` for more details.
