# Pytorch-DPPO
Pytorch implementation of Distributed Proximal Policy Optimization: https://arxiv.org/abs/1707.02286
Using PPO with clip loss (from https://arxiv.org/pdf/1707.06347.pdf).

work in progress

#req
python3.6 pytorch cmake

#install
use cmake-gui and visual studio to build on windows
./build.sh to build on linux


#run
./run_with_log.sh  on linux
python .\main.py cppSimulator on windows

## Acknowledgments
The structure of this code is based on https://github.com/alexis-jacq/Pytorch-DPPO.

Hyperparameters and loss computation has been taken from https://github.com/openai/baselines
