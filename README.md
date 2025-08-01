<div align="center">
  <img src="assets/HoPE.png" alt="HoPE" width="100"/>
</div>

<h2 align="center" style="font-size: 30px;">HoPE: Hybrid of Position Embedding for Length Generalization in Vision-Language Models</h2>

<h5 align="center">
  
[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.20444) 
[![Hugging Face Collection](https://img.shields.io/badge/HuggingFace-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/papers/2505.20444)
[![GitHub Stars](https://img.shields.io/github/stars/hrlics/HoPE?style=for-the-badge&logo=github&logoColor=white&label=Stars&color=000000)](https://github.com/hrlics/HoPE)

</h5>


## 📢 News
- **\[07/13/2025\]** All training and evaluation scripts are released. Check it out!
- **\[06/29/2025\]** Our work is covered by [JIQIZHIXIN (机器之心)](https://mp.weixin.qq.com/s/KQHGw8_v0rEY8pS7jufRbQ)!
- **\[05/26/2025\]** Release our paper on [arXiv](https://arxiv.org/abs/2505.20444).


## 🔭 Overview

Extending RoPE to multimodal scenarios typically involves allocating different frequencies to encode different positional components (*t*, *x*, *y*). In this paper:

1️⃣ We first investigate **how different frequency allocation strategies impact the semantic modeling capabilities of VLMs**. Our analysis reveals that current multimodal RoPEs, which keep all frequencies, are unreliable in long-term semantic modeling. HoPE tackles this issue by **Hybrid Frequency Allocation (HFA)**, which integrates *zero frequencies* for reliable semantic modeling over extended contexts. 

2️⃣ Moreover, we point out that existing temporal index scaling of visual tokens lacks flexibility and robustness during inference, where videos exhibit varying speeds and information densities. To address this, HoPE introduces **Dynamic Temporal Scaling (DTS)**, which enables VLMs to learn multi-scale temporal relationships during training and adaptively select temporal scaling during inference.

<div align="center">
  <img src="assets/Figure1.png" alt="Figure1" width=70%/>
</div>


## 🛠️ Requirements
1. Clone this repository and install `transformers==4.45.2` from source
```
git clone https://github.com/hrlics/HoPE.git
cd HoPE
wget https://github.com/huggingface/transformers/archive/refs/tags/v4.45.2.tar.gz
tar -xzf v4.45.2.tar.gz
```

2. Install required packages
```
bash setup_env.sh
```

3. Replace the code in
```
HoPE/transformers-4.45.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
```
with `HoPE/modeling_hope.py`. The differences are marked with `# MODIFIED`.


## 🚀 Train

We utilize a subset of [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) as training data, which comprises 30k videos with durations under 2 minutes and 3k videos with durations between 2 to 3 minuts (~300k pairs).

Under `LLaMA-Factory/`, run the following script to start training:
```
train_hope.sh
```
Adjustments are made to `LLaMA-Factory/src/llamafactory/data/mm_plugin.py` to accomodate Qwen2-VL's training recipe.

## 🔍 Evaluation

#### Long Video Understanding

Evaluations on long video understanding are based on [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). The first step is to install relevant dependencies: 
```
cd lmms-eval
pip install -e .
```

Then, run the following script to start evaluations on MLVU, LongVideoBench, and Video-MME:
```
bash eval_LVU.sh
```

Adjustments are made to `lmms-eval/lmms_eval/models/qwen2_vl.py` to accomodate our evaluation configs.

#### Long Video Retrieval

Under `vision_niah/`, run the following script to produce haystack and needle embeddings for long video retrieval:
```
bash produce_haystack_and_needle_embedding.sh
```

Now, we can run evaluations:
```
bash eval.sh
```

## 📖 Citation
If you find our work helpful, please consider citing 📝 and giving us a star ⭐
```
@article{li2025hope,
  title={HoPE: Hybrid of Position Embedding for Length Generalization in Vision-Language Models},
  author={Li, Haoran and Qin, Yingjie and Ou, Baoyuan and Xu, Lai and Xu, Ruiwen},
  journal={arXiv preprint arXiv:2505.20444},
  year={2025}
}
```

## 🙏 Acknowledgements
We thank the authors of [Qwen2-VL](https://github.com/QwenLM/Qwen2.5-VL), [VideoRoPE](https://github.com/Wiselnn570/VideoRoPE), [transformers](https://github.com/huggingface/transformers), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and [vLLM](https://github.com/vllm-project/vllm) for their wonderful work.
