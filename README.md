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

## Computation environment 

In order to run our code successfully and in a moderate time, you will need access to a powerful computer, preferably equipped with a GPU. For reference, all our experiments have been run on a computer equipped with an [NVIDIA Tesla P100](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-p100/pdf/nvidia-tesla-p100-datasheet.pdf) GPU, an [Intel Xeon E5-1650 v3](https://ark.intel.com/products/82765/Intel-Xeon-Processor-E5-1650-v3-15M-Cache-3-50-GHz-) CPU, and 62 GB of RAM. In case you do not have access to a GPU, we recommend skipping the compute-intensive model training part and running only the code that relates to the data generation ([data_simulation.ipynb](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/data_simulation.ipynb)) and evaluation of the algorithms proposed in the paper ([evaluation.ipynb](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/evaluation.ipynb)). Doing this will verify the results for the training dataset generated by you and using the pre-trained models from our simulations for the paper. You can also investigate how these models were created and trained exploring the code and explanations in [spot_net.ipynb](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/spot_net.ipynb) and [conv_net.ipynb](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/conv_net.ipynb). (Note that the code for data generation uses a fixed seed value that you can modify to generate randomized training images.)

## Reading through the code and explanations

If you want to explore the code and related explanations without running them, you can visualize them through [`nbviewer.jupyter.org`](https://nbviewer.jupyter.org/github/poldap/SpotNet/tree/master/). This option entails few software requirements. You can use [`nbviewer.jupyter.org`](https://nbviewer.jupyter.org/github/poldap/SpotNet/tree/master/) from any modern browser, without any further installation requirements. Nonetheless, this will only allow you to visualize the notebooks, and reproducing the results will not be possible.  

## Executing the code  

If you would like to execute the code on your own computation environment and fully or partially replicate our results, the software requirements are:  
1. [Python 3](https://www.python.org/) (for reference, we used `python 3.6.5`), along with the scientific computing packages [`numpy`](http://www.numpy.org/), [`scipy`](https://www.scipy.org/), [`scikit-image`](https://scikit-image.org/), and [`matplotlib`](https://matplotlib.org/).
2. [Jupyter](https://jupyter.org/) (for reference, we used `jupyter 4.4.0`), and
3. [TensorFlow](https://www.tensorflow.org/) (for reference, we used `tensorflow 1.8.0` compiled for use in the GPU with [`cudnn 7.1`](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/spot_net.ipynb) and [`nccl 2.1.15`](https://developer.nvidia.com/nccl)).   

To execute our code, you will need to be familiar with Jupyter notebooks running Python code. For that, you can refer to any of the several freely available tutorials, e.g., [\[6\]][6]. After installation of the required packages, navigate to the folder where this repository has been cloned and execute `jupyter notebook` to launch the Jupyter notebook server in that folder. Then, a simple click to any of the `*.ipynb` files listed in the Jupyter notebook website loads it on a new tab, where controls for running our code are available. In the following, we go through the different notebooks included in the order they should be run to reproduce our results.

### Data generation

The generation of synthetic FluoroSpot images is provided by [`data_simulation.ipynb`](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/data_simulation.ipynb). To understand the theory and the implementation behind the generation of synthetic FluoroSpot images, follow the explanations in each cell preceding the code and run them sequentially. You can freely change the code parameters, however, we provide no guarantees against invalid or ill-defined parameter values. Alternatively, to simply generate the images with the default parameters, one can use the menu and select `Kernel -> Restart and Run All`. The synthetic FluoroSpot images that form our training and test databases will be generated and stored in the `sim_data` directory when all cells have finished running. In particular, [`data_simulation.ipynb`](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/data_simulation.ipynb) will create 4 `*.npy` files in the `sim_data` directory, of which one (`result_1250_cells_10_images.npy`) will contain our training database, while the remaining three contain our test database. The accumulated size of these four files is expected to be approximately **10 GB**.  

### SpotNet training (optional)  

Once the data has been generated, it will be stored in the `sim_data` directory. You can then run the [`spot_net.ipynb`](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/spot_net.ipynb) notebook step by step to see how the proposed network is defined. To simply train the network, select `Kernel -> Restart and Run All`. Running this notebook took approximately **23 hours in our reference computer** (see above for information regarding our computation environment). When all cells have finished running, the training metrics (i.e., the train and validation losses at regular time intervals) and some trained models will be stored in the `results` directory.

### ConvNet training (optional)
  
Once the data has been generated, it will be stored in the `sim_data` directory. You can then run the [`conv_net.ipynb`](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/conv_net.ipynb) notebook step by step to see how the ConvNet structure is defined. To simply train the network, select running `Kernel -> Restart and Run All`. As in the previous case, this may take a long time, and the training metrics and some trained models will finally be available in the `results` directory.  

### Evaluation of the trained models  

Once the data has been generated, and optionally, the models have been trained, they will be available in the `sim_data` and `results` directories respectively. You can the run the [`evaluation.ipynb`](https://nbviewer.jupyter.org/github/poldap/SpotNet/blob/master/evaluation.ipynb) notebook step by step to see how the different models fare on the test database of the generated images, both in terms of the fitting loss and the detection accuracy. If you have trained the models yourself, you can set `own_data = True` to evaluate them. If you do not do that, the pre-trained models of SpotNet and ConvNet we provide will be loaded and evaluated instead.  
