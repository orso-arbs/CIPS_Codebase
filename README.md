
# README
# Visit-Cellpose-LineageTree Pipeline

## Description: 
The Visit-Cellpose-Lineage Pipeline provides a quick way of analysing cellular structures in large data using VisIt, Cellpose.

Pipeline:
- CIPS_Pipe_2 - orchestrates the data pipeline.
- Visit_Projector - Uses VisIt on the Euler cluster to read a database and plot a
pseudo-colour on an isosurface, from a particular view, then saves a .png file for each state in
the database. Also saves properties such as average radius and time.
- CP_segment - Uses the Cellpose 3 algorithm with a specified segmentation model to
automatically estimate and segment the image.
- CP_extract - Combines the data from the previous two steps and deduce initial cell prop-
erties such as centroids, areas, diameters, and contour lenths.
- Analysis_Altantzis2011 - Orchestrates the following functions for the analysis for the
specific case of the Altantzis (2011) numerical simulation data (DNS).
- dim3_A11 - Dimensionalises the extracted properties to physical units.
- Spherical_Reconstruction_2 - Reconstructs the properties 3D with a perfect sphere.
- CST_Selection - Selects the cells inside the cubed sphere symmetric tile and reconstructs
the properties of the entire spherically expanding flame (SEF).
- Auxilliaries - Various functions deal with colour table generation, and plotting. These are
not further described here but in their docstrings

Note: The code has been checked for functionality only in Windows 11 with Visit 3.4.2 and Cellpose 3.1.1.1

----

## Table of Contents
- [Installation](#installation)
- [Usage](#Usage)


## Installation
1. Clone the GitHub repository:
2. Install python dependencies:
   1. Assuming you have Conda installed, use your Terminal to run:
    ``conda env create --name envname --file=MastersThesis_Env2_py39.yml``
   - Note: Cellpose was found to work with numpy in python 3.9.
   - Note: This installs torch for CUDA GPU acceleration. If you don't run with an NVIDIA GPU, this won't help your speed. 
3. Install VisIt 3.4.2 December 2024 both on your mashine and on your Euler /cluster/home/username:
   1. [Visit releases](https://visit-dav.github.io/visit-website/releases-as-tables/#series-34).
   2. [Installation guide and Starting VisIt](https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.3.3/getting_started/Installing_VisIt.html).
4. Access Euler
   1. Make sure you have accessed Euler at least once to unlock it. Here is the [Euler wiki](https://scicomp.ethz.ch/wiki/Tutorials#Cluster_tutorials).
   2. (Optional) To not have to enter the Euler password every time visit is lanched, you can setup a passwordless login with an SSH key. [Here](https://www.ssh.com/academy/ssh/putty/windows/puttygen) are tutorial videos.
      1. PuTTYgen to generate key.
      2. PuTTY Pagent for passwordless login .


## Usage

1. Installation
2. Activate the python environment
   1. Open a terminal and run:
      ``conda activate cellpose_env_2``
      or using your IDE (like VScode) select cellpose_env_2
3. Connect to the ETH network
   1. Either use an ETH WIFI
   2. [Or connect via VPN](https://unlimited.ethz.ch/spaces/itkb/pages/21125994/VPN)
4. Setup the Pipeline in CIPS_Pipe_2
   1. in CIPS_Pipe_2 set the input_dir as the directory where all the output data is stored
   2. in Visit_Projector_1.py, set VisIt and cluster parameters
5. Run CIPS_Pipe_2
