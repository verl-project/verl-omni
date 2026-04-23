边界怎么划
应该留在 verl
这些更像“共享底座”：

single_controller
Ray worker / ResourcePoolManager / Role
dataset / tokenizer / processor / checkpoint / utils
通用 registry 机制
能被 text RL 和 multimodal RL 共同复用的 engine/worker 基建


应该留在 verl-omni
这些更像“多模态/扩散专属增量”：

vllm-omni rollout backend
FlowGRPO / diffusion trainer orchestration
diffusion / omni-specific loss, metric, sampling glue
QwenImage / BAGEL / Qwen-Omni 这类 custom pipeline
async reward / agent loop 的 multimodal 扩展


---

custom_pipeline 指的是针对某个具体生成模型的自定义 pipeline / adapter，用来让它能够接入 verl-omni 的 diffusion RL 训练或 rollout 流程。

所以你说的这句本质上是对的，只是我会稍微修正一下表述：

不是单纯“custom 的 diffusion pipeline”，而是：

面向某个模型族的 custom pipeline
这个 pipeline 被定制出来，目的是让它适配 diffusion RL / multimodal RL 的训练与采样需求
例如 FlowGRPO，也可以是以后别的 diffusion RL 算法
为什么说它是“为 RL 定制”的
因为普通的生成 pipeline，通常只关心：

给 prompt
跑 denoising
输出 image / video
但 RL 训练里的 pipeline 还经常需要额外支持：

rollout 阶段返回 log_probs
训练阶段按 step 构造 prepare_model_inputs
支持 true CFG / negative prompt 的特殊处理
支持 SDE window / noise level / scheduler patch
暴露 RL loss 所需的中间量
