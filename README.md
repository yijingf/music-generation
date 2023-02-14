## Introduction
Working towards a transformer-GM-VAE based model for latent acoustic feature extraction and constrained music generation. The extracted acoustic latent features will be used in another neuroscience study later, as well as evaluating the music generation quality.

## Preprocess
* Tokenize symbolic music using Magenta's midi tokenization methods. 
* Encode the audio using the mt3 model for audio transcription. 

Usage:
```
python3 ./src/build_dataset.py [-d Path to metadata file]
```

## Transformer GM-VAE
Todo


## Data
See `Data.md` for more details.