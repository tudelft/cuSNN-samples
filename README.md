# cuSNN-samples
Samples for [**cuSNN**](https://github.com/fedepare/cuSNN#cusnn-samples) developers which demonstrate the main features 
provided by the library.

At the moment, all samples available are to the motion-selective hierarchical SNN proposed in 
["*Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global 
Motion Perception*" (Paredes-Vallés, F., Scheper, K.Y., and de Croon, G.C.H.E., 2018)](https://arxiv.org/abs/1807.10936).
More varied samples will be included in the near future.

### Samples list

* **train-SSConv**: Train a sigle-synaptic Conv2d (SS-Conv) layer.
* **test-SSConv**: Test an SS-Conv layer.
* **train-MSConv**: Train a multi-synaptic Conv2d (MS-Conv) layer, preceded by a pre-trained SS-Conv layer.
* **test-MSConv**: Test a three-layer network with a SS-Conv, a Merge, and an MS-Conv layer (in this order).
* **train-Dense**: Train a Dense layer, preceded by pre-trained SS-Conv and MS-Conv layers.
* **test-MSConv**: Test a five-layer network with a SS-Conv, a Merge, an MS-Conv, a Pooling, and a Dense layer 
(in this order).
* **record-spikes**: Record the spiking activity of a five-layer network with a SS-Conv, a Merge, an MS-Conv, 
a Pooling, and a Dense layer (in this order). Requires the [**cnpy library**](https://github.com/rogersce/cnpy "cnpy library (C++ arrays to Numpy)").