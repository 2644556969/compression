These examples depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge). Make sure the python environment with ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`. Also ensure the `pyhe_client` wheel has been installed (see `python` folder for instructions).

This folder contains the base Cryptonets model and Compressnets for testing shallower network architectures for MNIST.


# Train the networks
To train the networks:
```bash
python Compressnets/train.py --epochs=20 --batch_size=128 --save_file=debug --logit_scale=0.5 --generate_architectures_depth = 4
```

--epochs : number of epochs to train 
--batch_size: batch size 
--save_file: filename to save the trained model to in "models/*.pb"
--logit_scale: linearly scaling the logits for model compression 
--genereate_architectures_depth: max multiplicative depth of architectures to generate (default -1 will not generate)

Training logits can be modified by changing y_train_logits / y_test_logits in code. 


# Test the network

## CPU backend
To test a network using the CPU backend, call
```bash
python test.py --batch_size=10000 \
               --backend=CPU \
               --model_file=models/<filename>.pb
```

## HE_SEAL backend
#### Encrypted
To test a network using the HE_SEAL backend using encrypted data, call
```bash
python test.py --batch_size=4096 \
               --backend=HE_SEAL \
               --model_file=models/<filename>.pb \
               --encrypt_server_data=true \
               --encryption_parameters=../configs/test_18_8.json
```
According to HE-transformer:
This setting stores the secret key and public key on the same object, and should only be used for debugging, and estimating the runtime and memory overhead.


For detailed description of choosing encryption parameters, see  [here](https://github.com/IntelAI/he-transformer/tree/master/examples/MNIST) and [here](https://github.com/microsoft/SEAL/tree/master/dotnet/examples). 

The test configs files in ../configs are labeled as ` test_a_b.json` where `a` is the post-decimal bit precision and `b` is the pre-decimal bit precision (ex. `test_18_8.json`).
