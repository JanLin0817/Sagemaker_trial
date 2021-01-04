# README

## Train

### Training

* wrap your train.py with `sagemaker.pytorch.PyTorch` and aws will train your `trian.py` on sepcify ec2 type device
* training entry point: `code/train.py`
    * reference [pytorch-offical](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
     
### Deploy

* `code/inference.py`
    * deploy 的 `entry_ponit`

### example

* pytorch_sagemaker.ipynb
* example inculde train/deploy
    * deploy from attached job(just trained): Success
    * deploy from re-attached job: Fail


## Tunnuing

* wrap `sagemaker.pytorch.PyTorch` into `sagemaker.tuner.HyperparameterTuner` and aws will tune your `train.py` on sepcify ec2 type device

### exmple

* pytorch_sagemaker_tunner.ipynb
* example include Train/Tune/Analyze/Deploy
    * deploy from best tunning job: Fail
    * deploy from specify tunning jib: Fail
    
