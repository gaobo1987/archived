## Build and install Faiss library
Faiss is a library for efficient similarity search and clustering of dense vectors.

#### Prerequisites: 
* This tutorial is for x86_64 machines on Linux and MacOS;
* This tutorial is verified with Ubuntu 20.04, but other Ubuntu versions should also work, or with minimum change;
* If you have an NVidia GPU machine available, make sure that CUDA 10+ is installed.

#### Installation
Install the CPU or GPU versions of Faiss via pip:
```shell script
pip install faiss-cpu --no-cache
pip install faiss-gpu --no-cache
```

If you prefer to build from source, then install, read on:

#### Step 1: Download the Faiss source code
* Go to a directory of your choice
* `wget https://github.com/facebookresearch/faiss/archive/master.zip`
* `unzip master.zip && mv faiss-master faiss && rm master.zip`

#### Step 2 (optional, but recommended): Activate virtual environment
If you want to install Faiss in your Python virtual environment, activate it, e.g. by
`source path/to/venv/bin/activate`, and do installation within the environment.

#### Step 3 (optional, but recommended): Manage multiple GCC and G++ compilers
CUDA 10 and earlier versions work with the GCC version no later than 8. 
Thus make sure you have the right compilers. One way to do this is to pro-actively manage 
multiple versions of GCC and G++. 

1.Install multiple C and C++ compiler versions
```shell script
sudo apt install build-essential
sudo apt -y install gcc-7 g++-7 gcc-8 g++-8 gcc-9 g++-9
```
2.Use the update-alternatives tool to create list of multiple GCC and G++ compiler alternatives:
```shell script
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
```
3.Check the available C and C++ compilers list and select desired version by entering relevant selection number:
```shell script
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

4.Each time after switch check your currently selected compiler version:
```shell script
gcc --version
g++ --version
```

#### Step 4: Install dependencies
* Faiss depends on BLAS, which is a linear algebra library.
    - `$ sudo apt-get install libblas-dev liblapack-dev`
* Faiss depends on swig, which is a C/C++ interface generator.
    - `$ sudo apt-get install swig`
* Faiss depepends on numpy as well.
    - `(venv) $ pip install numpy`

#### Step 5: Installing Faiss
Make sure you have the right dependencies (see step 3 and 4) before moving on.

Follow the instructions [here](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) 
for the installation. The instructions provided by the Faiss team is good enough,
however, it still requires some try-and-errors to make it work, 
below are the general steps to take, along with some tips that hopefully can assist other triers.
0. Go to the ./faiss folder
    * `cd faiss`
1. Generate system-dependent configuration for the makefile:
    * specify cuda path if applicable: `$ ./configure --with-cuda=/usr/local/cuda` 
    * otherwise: `$ ./configure --without-cuda`
2. Build the C++ library, it'd only take a few minutes when building on CPU,
    but will take about an hour to build on GPU:
    * `$ make`
3. Install headers and libraries (optional), with sudo:
    * `$ sudo make install`
4. Build the python interface:
    * `$ make py`
5. Install the python library (egg), it's better to install Faiss in your virtual environment:
    * `(venv) $ make -C python install` 
    
