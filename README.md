# CMVCG Model
This is  the implementation of CMVCG model for Live Video Comment Generation based on [PyTorch](https://pytorch.org/) and [Huggingface's Transformers](https://github.com/huggingface/transformers).

This model is a Transformer based model to generate live video comments non-autoregressively by their context comments and visual frames.

## Requirment
Install Transformers:
```
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
```

## Dataset
You can download our Bili-22 dataset at [here](https://drive.google.com/drive/folders/1BlW8O6VM8tVSSP4iF-opGYjEAAhSRXjs?usp=sharing) 

Another Livebot dataset be found at [here](https://drive.google.com/drive/folders/1QEZzKEv0G52WE_z8_7f4QpIq1mcs7ea1). This dataset is based on [Livebot](https://arxiv.org/abs/1809.04938) and the raw data can be found at [livebot](https://github.com/lancopku/livebot).

## Config
All the parameters can be set at: `\MyCMVCG\config`.

## Train
Praining step:
```
python pretrain.py 
```
Generate fine-tuning step:
```
python fine_tuning_CMVCG.py
```


## Test:
Generate comments:
```
python test_generate_CMVCG.py 
```
