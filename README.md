# SEPS: A Separability Measure for Robust Unlearning in LLMs
This repository is the official implementation for the paper: **SEPS: A Separability Measure for Robust Unlearning in LLMs**


<p align="center">
  <a href="https://arxiv.org/abs/2505.14832"> 📜 Paper</a>
</p>



## Installation

We follow the [TOFU Benchmark](https://github.com/locuslab/tofu/?tab=readme-ov-file#installation) to install the required dependencies, please run the following commands:

```shell
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Fictitious unlearning scenario

**(1) LLM-as-Judge Score with GPT4 (except for ME+GD)**

```shell
bash scripts/tofu/eval_gpt.sh
```

**(2) LLM-as-Judge Score with Meta-Llama-3-8B-Instruct (except for ME+GD)**

```shell
bash scripts/tofu/eval_llama.sh
```

## Acknowledgments

This repository builds upon selected components of the codebase from [A Closer Look at Machine Unlearning for Large Language Models](https://github.com/sail-sg/closer-look-LLM-unlearning) and extends it with our own experiments. We appreciate their outstanding work!