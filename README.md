<div align="center">
  <img src="assets/HoPE.png" alt="HoPE" width="100"/>
</div>

<h1 align="center" style="font-size: 30px;">HoPE: Hybrid of Position Embedding for Length Generalization in Vision-Language Models</h1>

<h5 align="center">
  
[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.20444) 
[![Hugging Face Collection](https://img.shields.io/badge/HuggingFace-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/papers/2505.20444)
[![GitHub Stars](https://img.shields.io/github/stars/hrlics/HoPE?style=for-the-badge&logo=github&logoColor=white&label=Stars&color=000000)](https://github.com/hrlics/HoPE)

</h5>


## 🚀 News
- **\[06/29/2025\]** Our work is covered by [Synced (机器之心)](https://mp.weixin.qq.com/s/KQHGw8_v0rEY8pS7jufRbQ)!
- **\[05/26/2025\]** Release our paper on [arXiv](https://arxiv.org/abs/2505.20444).

## 🛠️ Requirements
1. Clone this repository and install `transformers==4.45.2` from source
```
git clone https://github.com/hrlics/HoPE.git
cd HoPE/LLaMA-Factory
wget https://github.com/huggingface/transformers/archive/refs/tags/v4.45.2.tar.gz
tar -xzf v4.45.2.tar.gz
```

2. Install required packages
```
bash setup_env.sh
```



## :clipboard: Todo
- [x] Release [paper](https://arxiv.org/abs/2505.20444) on arXiv
- [ ] Release training code. Stay tuned! 


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
We thank the authors of [Qwen2-VL](https://github.com/QwenLM/Qwen2.5-VL), [VideoRoPE](https://github.com/Wiselnn570/VideoRoPE), [transformers](https://github.com/huggingface/transformers), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and [vLLM](https://github.com/vllm-project/vllm) for releasing their helpful implementation.
