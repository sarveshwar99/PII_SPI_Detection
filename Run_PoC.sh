#!/usr/bin/env bash
#Copyright 11/2019 Jack K. Rasmus-Vorrath

read -p "Is Python 3 Anaconda Distribution installed? (y/n)" CONDA

if [ "$CONDA" == "n" ]; then

	read -p "Choose OS: Linux/MacOSX/Windows (L,M,W)" input_var
	echo "$input_var was selected";

	if [ "$input_var" == "L" ]; then
		wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -s -o ~/miniconda.sh;

	elif [ "$input_var" == "M" ]; then
		curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -s -o ~/miniconda.sh;

	elif [ "$input_var" == "W" ]; then
		curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -s -o ~/miniconda.sh;

	else
		echo "Please select valid OS, using L, M, or W"
		exit 1;
	fi

	chmod +x ~/miniconda.sh
	~/miniconda.sh -b -p $HOME/Miniconda3
	rm ~/miniconda.sh

	export PATH=~/Miniconda3/Scripts:"$PATH";
fi

##############################################################

read -p "Create Conda environment? (y/n)" CREATE

if [ "$CREATE" == "n" ]; then
	echo "Must create environment to install dependencies"
	echo "Exiting program..."
	exit 1;
else
	echo "Using default environment name: PoC_Env"
	echo "# BASH: conda env create
name: PoC_Env
channels:
- !!python/unicode
  'defaults'
dependencies:
- python=3.6
- pip==9.0.1
- numpy==1.13.3
- pandas==0.20.3
- pip:
  - tensorflow==1.9.0
  - keras==2.2.0" > PoC_Env.yml
  
	echo "Registering Conda-Forge channel..."
	conda config --append channels conda-forge
	
	echo "Installing with Conda..."	
	conda create --name PoC_Env\
	python=3.6 pip==9.0.1\
	numpy==1.13.3 pandas==0.20.3 tensorflow==1.9.0 keras==2.2.0
	
	echo "Activating environment..."
	if [ "$CONDA" == "n" ]; then
		source ~/Miniconda3/etc/profile.d/conda.sh;
	else
		source ~/anaconda3/etc/profile.d/conda.sh;
	fi
	
	conda activate PoC_Env
	
	echo "Installing with Pip..."
	pip install argparse_prompt
	pip install scikit-learn==0.19.1
	
	python ./CNN_BLSTM.py
	
	conda deactivate
	echo "Execution Complete";
fi