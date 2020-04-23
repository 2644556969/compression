This directory was originally adapted from https://github.com/NervanaSystems/he-transformer

* `train.py` is used to train the models. Currently, we have a simple CNN model with two parameters: `batch_norm` and `train_poly_act`
  * if `batch_norm` is enabled (i.e. `batch_norm=True`), a batch norm layer will follow each convolution layer
  * if `train_poly_act` is enabled, the polynomial activations will be of the form `ax^2 + bx`, where `a=0, b=1`, intitially, but `a,b` are trained. If `train_poly_act` is not enabled, the polynomial activation is fixed at `0.125x^2 + 0.5x + 0.25`
* `test.py` is used to perform inference.
HE-transformer expects serialized models to have inputs as placeholders, and model weights stored as constants.
* `test.py` also performs this serialization. Finally, `test.py` also optimizes the model for inference, for example by folding batch norm weights into convolution weights to reduce the mutliplicative depth of the model.

By default, train.py trains these three models are trained on logit data 

cnn (depth 8)


small_cnn (depth 5) 



small_cnnact (depth 7) 



Each model is trained with --batch_norm=True and --train_poly_act=True. Further, they use polynomial activation functions with gradient clipping. 


# 1 Train a model
```python
python train.py --model=cnn --batch_norm=True --train_poly_act=True [--max_steps=10000]
```

## To resume training a model
```python
python train.py --model=cnn --resume=True
```

# 2. Run trained model (specify --model=<modelname>)
## Skip inference, just export serialized graph:
```python
NGRAPH_TF_BACKEND=NOP NGRAPH_ENABLE_SERIALIZE=1 python test.py --model=cnn --batch_norm=True --train_poly_act=True --batch_size=1
```

## Run inference on CPU backend:
```python
python test.py --model=cnn --batch_norm=True --train_poly_act=True --batch_norm=True --train_poly_act=True --batch_size=10000
```

## Run inference on HE backend
```python
NGRAPH_HE_SEAL_CONFIG=ckks_config_13_depth12.json NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --model=cnn --batch_size=1000
```

