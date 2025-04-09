#!/bin/bash

# For the Macbook Air
VENV_PATH="/Users/jasonmulderrig/research/projects/irreg-net-prve"

# Set up Python virtual environment and associated Python packages

if [ ! -d ${VENV_PATH} ]
then
  mkdir -p ${VENV_PATH}
  python3 -m venv ${VENV_PATH}
  cd ${VENV_PATH}
else
  cd ${VENV_PATH}
  if [ ! -f pyvenv.cfg ]
  then
    python3 -m venv ${VENV_PATH}
  else
    rm -rf bin include lib share && rm lib64 && rm pyvenv.cfg
    python3 -m venv ${VENV_PATH}
  fi
fi

source bin/activate

pip3 install wheel && pip3 install --upgrade setuptools && pip3 install --upgrade pip
pip3 install hydra-core numpy scipy networkx[default] matplotlib ipython ipykernel ipympl

deactivate
