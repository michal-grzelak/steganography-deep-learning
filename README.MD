# How to run the program

1. Ensure that [Anaconda](https://www.anaconda.com/distribution/) is installed on the computer (it provides Python and Conda)
2. Download [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) (needed for TensorFlow GPU and any other machine learning library performing calculations on the GPU)
3. Download the repository listed above
4. Go to the *train_data* directory and unpack the archive `train_data.zip.001` (it will unpack all archives)
5. Open a command console in the root of the repository
6. Type `conda env create --file environment.yaml` (this will create the python virtual environment and download all required packages)
7. Type `conda activate tensorflow` (it will activate the environment)
8. Type `python main.py` (this will run the program)

Step 7 has to be made every time console window is reloaded.