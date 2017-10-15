# Abstract
This repository contains Tensorflow samples of making unaggregated gradients.

We have implemeted 4 implementations

 - calculating with batch size 1 (named "batch1", implemented in "total_grad.py")
 - using jacobian (named "jacob", implemented in "loop_grad.py")
 - copying infererence network batch size times and differentiating totally
 (named "copy", implemented in "copy_grad.py")
 - overwrite default convolution gradients (named "overwrite", implemented in "overwrite_grad.py")

### jacobian
This implementation refers to [issue](https://github.com/tensorflow/tensorflow/issues/675#issuecomment-319891923).

### copy
This implementation refers to [issue](https://github.com/tensorflow/tensorflow/issues/4897#issuecomment-290997283).

### overwrite
This decompose conv2d gradients into unaggregated gradients and aggregation of them.
Using depthwise_conv2d to get unaggregated gradients, this implementation is slow.

# Performance
### Environment
We use a network which has 64x64 input, 7 convolution, 3 fc layer in all measurements.

|Title|Contents|
|:--|--:|
|GPU|	GTX 1070|
|CPU|	Core(TM) i5-4440|
|Cuda|	Cuda 8.0 cuDNN 5.1.5|
|Tensorflow| 1.2.1|

### Result
"Time per data" is the average time where each implementation calculates a gradient
for each data.

|Implementation|Time per data(msec)|
|:--|--:|
|batch1|5.657|
|jacob|169.55|
|copy |4.964|
|overwrite|19.65|
|(aggregated case )batch64|1.138|
