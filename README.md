# STABLE: Gated Continual Learning for Large Language Models  
*An extension of SEAL focused on gated continual self editing and bounded forgetting in large language models.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2510.16089-b31b1b.svg)](https://arxiv.org/abs/2510.16089)

👥 **Authors:** William Hoy and Nurcin Celik


**SEAL (Self Adapting Language Models)** — introduced by Zweiger et al. (2025) — frames model editing as a reinforcement learning problem in which a language model autonomously proposes and evaluates its own parameter updates to improve factual consistency. SEAL’s architecture includes utilities for LoRA based fine tuning, data generation, and evaluation servers that enable adaptive model updates in a few shot setting.

**STABLE** builds on SEAL’s open source foundation but extends it in a new direction: **gated continual learning**.  
Rather than focusing on few shot RL adaptation, STABLE introduces *bounded forgetting mechanisms* that regulate sequential LoRA merges through gating metrics—**Exact Match (EM) drop**, **Bits increase**, and **KL divergence** thresholds.  
This design allows continual self editing of large language models while preserving prior knowledge and mitigating catastrophic forgetting.



### 📂 Repository Structure

The continual editing pipeline is launched via:
general-knowledge/scripts1/continual_self_edits.sh


## 🔧 Setup

### 1. Clone the repository

```bash
git clone https://github.com/Bhoy1/STABLE.git
cd STABLE
```

### 2. Set up a virtual environment

Using **conda**:

```bash
conda create -n seal_env python=3.12
conda activate seal_env
```

Using **venv**:

```bash
python3.12 -m venv seal_env
source seal_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Create a `.env` file in the project root and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. SLURM users

Before running any shell scripts, make sure to update the SLURM directives at the top of each `.sh` file to match your system configuration. All experiments can be run with 2 A100/H100 GPUs. Other setups may require refactoring and/or changing model sizes.

## License & Attribution

This repository builds upon the open-source [**SEAL**](https://github.com/Continual-Intelligence/SEAL)  
framework by **Adam Zweiger et al.** (MIT License), but **excludes modification of SEAL’s few shot learning  
and reinforcement based self adaptation components.**

All new gating and continual learning extensions are part of **STABLE** and are  
© 2025 **William Hoy** and **Nurcin Celik**, released under the same [MIT License](LICENSE).


## 📄 Citation

If you found this code useful, please cite:

```
@misc{hoy2025stable,
      title        = {{STABLE: Gated Continual Learning for Large Language Models}},
      author       = {William Hoy and Nurcin Celik},
      year         = {2025},
      eprint       = {2510.16089},
      archivePrefix= {arXiv},
      primaryClass = {cs.LG},
      url          = {https://arxiv.org/abs/2510.16089}
}

@misc{zweiger2025selfadaptinglanguagemodels,
      title        = {{Self-Adapting Language Models}}, 
      author       = {Adam Zweiger and Jyothish Pari and Han Guo and Ekin Akyürek and Yoon Kim and Pulkit Agrawal},
      year         = {2025},
      eprint       = {2506.10943},
      archivePrefix= {arXiv},
      primaryClass = {cs.LG},
      url          = {https://arxiv.org/abs/2506.10943}
}
```
