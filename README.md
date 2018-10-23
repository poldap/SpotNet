# SpotNet &mdash; Learned Iterations for Cell Detection in Image-based Immunoassays
### Pol del Aguila Pla [\[1\]][1], Vidit Saxena [\[2\]][2], and Joakim Jaldén [\[3\]][3]

This GitHub repository contains the code and explanations that complement the paper of the same name, submitted to the 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019) [\[4\]][4]. The code provided here is **still not ready** for functioning when being run directly, and it will be improved during the coming weeks.

[\[1\]][1]: Pol del Aguila Pla's research website  
[\[2\]][2]: Vidit Saxena's research profile at KTH  
[\[3\]][3]: Joakim Jaldén's research profile at KTH  
[\[4\]][4]: ISBI 2019 website

[1]: https://poldap.github.io  
[2]: https://kth.se/profile/vidits   
[3]: https://kth.se/profile/jalden 
[4]: https://biomedicalimaging.org/2019/

## Prerequisites
1. Python 3  
2. Jupyter  
3. Tensorflow  
  
Access to a GPU is expected to greatly speed up the data generation and evaluations.  

## Executing the code

The code is organized as Jupyter notebooks running IPython. If you are unfamilir with Jupyter notebooks, you can refer to any of the several existing tutorials, for example this one: [Getting Started With Jupyter Notebook for Python](https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46).

## Data Generation

The generation of synthetic Fluorospot images is provided by `data_simulation.ipynb`. After you have started the Jupyter notebook server from your cloned Spotnet directory, you should be able to open and run `data_simulation.ipynb` in your browser.  
  
If you need to understand the theory behind image generation, follow the explanations for each cell and run them sequentially. When all the cells have been run, the images will be generated and stored in a newly created `sim_data` directory.  
  
Alternatively, You can generate images directly with the default parameters by running `Kernel -> Restart and Run All`

**Note**: The generation of images with default parameters is expected to take several minutes, and may even take up to a few hours in non-GPU systems.

## SpotNet Evaluation

Once the data has been generated, it will be stored in the `sim_data` directory. Run the `spot_net.ipynb` notebook by running `Kernel -> Restart and Run All`.  

## ConvNet Evaluation
  
Once the data has been generated, it will be stored in the `sim_data` directory. Run the `conv_net.ipynb` notebook by running `Kernel -> Restart and Run All`.
  
