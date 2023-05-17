# A Topological Deep Learning Framework for Neural Spike Decoding

Edward C. Mitchell, Brittany Story, David Boothe, Piotr J. Franaszcuk, Vasileios Maroulas

> We present simplicial convolutional neural networks (SCRNNs), which combine simplicial convolutions with backend recurrently connected layers.
> We showcase the network by decoding two types of brain cells: grid cells and head direction (HD) cells.
> The neural activity is first defined on a simplicial complex via a pre-processing procedure and then fed to the SCRNN for decoding.
> We also include the code for comparisons to a feed forward fully connected neural network (FFNN) and a recurrent neural network (RNN).

* Paper: [arXiv:2212.05037][arxiv]

[arxiv]: https://arxiv.org/abs/2212.05037

## How to run

1. Decoding of each dataset is in properly named folder.

2. Open terminal at network_scripts folder

3. Run desired NN architecture 
	* main.py: run either SCRNN or SCNN
	* main_mods.py (grid cell only): run either an SCRNN or SCNN where each grid cell module is treated independently in the pre-processing procedure
	* main_FFNN.py: run a FFNN
	* main_RNN.py: run a RNN
