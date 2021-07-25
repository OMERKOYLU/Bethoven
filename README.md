# Bethoven Ai

An ai which produced for compose classical music artifacts

mp3 folder contains two mp3 files which composed by **Bethoven Ai**

*classical-piano* and *classical-piano-type0* folders has classical music compositions, composed by famous artists for training **Bethoven-Ai**. *train.py* and *create_and_train_Ai.ipynb* creates and trains ai by using keras and our model like figure *model.png*

at *midi* folder you can see 27 compositions which composed at different stages of train, *model* folder has models (they are same) and *weights* folder which has weights of ai at train stages *musicianAi_weights.h5* is the newest but its best is *musicianAi_weights-backup-3.h5*, I recommend it to use.

our train happened as 10 epochs per stage and we caught best weights at 3h stage

## Requirements

pretty_midi, tensorflow

## Usage

`python3 compose.py <tempo> <note_count>`

if you install requirements and give this command at commandline you will get a composition with specified tempo and note count

