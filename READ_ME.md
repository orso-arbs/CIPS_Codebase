
# README
# Visit-Cellpose-LineageTree Pipeline

## Description: 
The Visit-Cellpose-Lineage Pipeline provides a quick way of analysing cellular structures in large data using VisIt, Cellpose and [TBD Lineage tree].

Pipeline:


Note: The code has been checked for functionality only in Windows 11 with Visit 3.4.2 and Cellpose 3.1.1.1

----

## Table of Contents
- [Installation](#installation)
- [Usage](#Usage)


## Installation
1. Clone the repository:
2. Install python dependencies:
   1. Assuming you have Conda installed, use your Terminal to run:
    ``conda env create --name envname --file=cellpose_env_2.yml`` - [TBD]
   - Note: Cellpose was found to work with numpy in python 3.9.
   - Note: This installs torch for CUDA GPU acceleration. If you don't run with an NVIDIA GPU, this won't help your speed. 
1. Install VisIt 3.4.2 December 2024 both on your mashine and on your Euler /cluster/home/username:
   1. [Visit releases](https://visit-dav.github.io/visit-website/releases-as-tables/#series-34)
   2. [Installation guide and Starting VisIt](https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.3.3/getting_started/Installing_VisIt.html)


## Usage


1. Installation
2. Activate the python environment
   1. Open a terminal and run:
      ``conda activate cellpose_env_2``
      or using your IDE (like VScode) select cellpose_env_2
3. Connect to the ETH network
   1. Either use an ETH WIFI
   2. [Or connect via VPN](https://unlimited.ethz.ch/spaces/itkb/pages/21125994/VPN)
4. Run Pipeline
