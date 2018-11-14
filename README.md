# SpotNet &mdash; Learned Iterations for Cell Detection in Image-based Immunoassays
### Pol del Aguila Pla [\[2\]][2], Vidit Saxena [\[3\]][3], and Joakim Jaldén [\[4\]][4]

This GitHub repository contains the code and explanations that complement the paper of the same name (which can be downloaded from [\[1\]][1]), submitted to the 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019) [\[5\]][5]. 

[\[1\]][1]: Pre-print of the article in arXiv.org, arXiv:1810.06132 \[eess.IV\]  
[\[2\]][2]: Pol del Aguila Pla's research website  
[\[3\]][3]: Vidit Saxena's research profile at KTH  
[\[4\]][4]: Joakim Jaldén's research profile at KTH  
[\[5\]][5]: ISBI 2019 website    
[\[6\]][6]: Sebastian Eschweiler, "Getting started with Jupyter notebook for Python", [_CodingTheSmartWay.com_](https://codingthesmartway.com), Dec. 2017

[1]: https://arxiv.org/abs/1810.06132
[2]: https://poldap.github.io  
[3]: https://kth.se/profile/vidits   
[4]: https://kth.se/profile/jalden 
[5]: https://biomedicalimaging.org/2019/
[6]: https://codingthesmartway.com/getting-started-with-jupyter-notebook-for-python/

## Prerequisites

### Hardware 

In order to run our code successfully and in a moderate time, you will need access to a powerful computer, preferably equipped with a GPU. For reference, we report the timings for the most demanding computations, and all our experiments were run on a computer equipped with an [NVIDIA Tesla P100](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-p100/pdf/nvidia-tesla-p100-datasheet.pdf) GPU, an [Intel Xeon E5-1650 v3](https://ark.intel.com/products/82765/Intel-Xeon-Processor-E5-1650-v3-15M-Cache-3-50-GHz-) CPU, and 62 GB of RAM. If you do not have this option, we recommend skipping the training of the models and running only the code that relates to data generation ([data_simulation.ipynb](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/data_simulation.ipynb)) and evaluation ([evaluation.ipynb](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/evaluation.ipynb)). Doing this will use the pre-trained models that we provide, and still verify our results. Of course, you can investigate how these models were created and trained exploring the code and explanations in [spot_net.ipynb](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/spot_net.ipynb) and [conv_net.ipynb](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/conv_net.ipynb).

### Software

There are two recommended ways of exploring the code and explanations we provide here:  
1. visualizing them through [`nbviewer.jupyter.org`](https://nbviewer.jupyter.org/github/poldap/SpotNet/tree/master/), or
2. running them on your computer / computational server to fully or partially replicate our results.    

For option 1, the software requirements are minimal. You can use [`nbviewer.jupyter.org`](https://nbviewer.jupyter.org/github/poldap/SpotNet/tree/master/) from any modern browser, without any further installation requirements. Nonetheless, this will only allow you to visualize the notebooks, and reproducing the results will not be possible.

For option 2, i.e., to run our code in your own computer or computational server, you will need to ensure the following software is available:
1. [Python 3](https://www.python.org/) (for reference, we used `python 3.6.5`), along with the scientific computing packages [`numpy`](http://www.numpy.org/), [`scipy`](https://www.scipy.org/), [`scikit-image`](https://scikit-image.org/), and [`matplotlib`](https://matplotlib.org/).
2. [Jupyter](https://jupyter.org/) (for reference, we used `jupyter 4.4.0`),
3. and [TensorFlow](https://www.tensorflow.org/) (for reference, we used `tensorflow 1.8.0` compiled for use in the GPU with [`cudnn 7.1`](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/spot_net.ipynb) and [`nccl 2.1.15`](https://developer.nvidia.com/nccl)).   

## Executing the code

To execute our code, you will need to be familiar with Jupyter notebooks, which are the support we use to contain most of our Python code. For that, you can refer to any of the many existing tutorials, e.g., [\[6\]][6]. After installation of the required packages, navigating to the folder where this repository has been cloned and executing `jupyter notebook` launches the Jupyter notebook server in that folder. Then, a simple click to any of the `*.ipynb` files listed in the Jupyter notebook website loads it on a new tab, where controls for running our code are available. In the following, we go through the different notebooks included in the order they should be run to reproduce our results.

### Data generation

The generation of synthetic FluoroSpot images is provided by [`data_simulation.ipynb`](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/data_simulation.ipynb). To understand the theory and the implementation behind the generation of synthetic FluoroSpot images, follow the explanations for each cell and run them sequentially. Alternatively, to simply generate the images with the default parameters, one can use the menu and select `Kernel -> Restart and Run All`. In either case, the synthetic FluoroSpot images that form our training and test databases will be stored in the `sim_data` directory when all cells have finished running. In particular, [`data_simulation.ipynb`](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/data_simulation.ipynb) will create 4 `*.npy` files in the `sim_data` directory, of which one (`result_1250_cells_10_images.npy`) will contain our training database, while the remaining three contain our test database. The accumulated size of these four files is expected to be around **10 GB**.

### SpotNet training (optional)

Once the data have been generated, they will be stored in the `sim_data` directory. You can then run the [`spot_net.ipynb`](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/spot_net.ipynb) notebook step by step to see how the proposed network is defined. To simply train the network, select `Kernel -> Restart and Run All`. Running this notebook took approximately **23 hours in our reference computer** (see above for technical specifications). Regardless of how you do it, the training metrics (i.e., the train and validation losses at regular time intervals) and some trained models will be stored in the `results` directory when all cells have finished running.

### ConvNet training (optional)
  
Once the data have been generated, they will be stored in the `sim_data` directory. You can then run the [`conv_net.ipynb`](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/conv_net.ipynb) notebook step by step to see how the ConvNet structure is defined. To simply train the network, select running `Kernel -> Restart and Run All`. As in the previous case, this may take a long time, and the training metrics and some trained models will be available in the `results` directory when all cells have finished running.

### Evaluation of the trained models

Once the data have been generated, and optionally, the models have been trained, they will be available in the `sim_data` and `results` directories, respectively. You can the run the [`evaluation.ipynb`](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/evaluation.ipynb) notebook step by step to see how the different models fare on the test database of 150 images, both in terms of fitting loss and detection accuracy. If you have trained the models yourself, you can set `own_data = True` to evaluate them. If you do not do that, the pre-trained models of SpotNet and ConvNet we provide will be loaded and evaluated instead. 

  
