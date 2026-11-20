# Qwen-Image 2.1 DiffusionNFT LoRA

本目录对应 [RFC #799](https://github.com/verl-project/verl-omni/issues/799) 的 PR 1：
在线 T2I OCR 与 1–4 张参考图编辑 LoRA。DiT 基座、Qwen3-VL encoder、VAE 和 reward 模型冻结。
全参策略状态、全量权重同步及全参 checkpoint 扩展属于 PR 2。

调用链沿用 V1 synchronous trainer、NFTDiffusersFSDPEngine、DiffusionNFTLoss 和已有 LoRA 同步机制：

```text
run_qwen_image_21_{ocr,edit}_lora_v1.sh
└── main_diffusion_v1
    ├── prepare_processor_files → 标准 hf_processor
    ├── QwenImage21DiffusionSingleTurnAgentLoop → 继承通用 run()
    │   ├── 解码、缩放参考图
    │   └── 原生模板分词 → 同一批参考图及 token IDs 送入 rollout
    ├── QwenImage21DiffusionNFTPipeline → vLLM-Omni + 已有 QwenImageLoRAMixin
    └── QwenImage21DiffusionNFT → 已有 NFT engine/loss
```

实测环境为 PyTorch 2.13.0+cu130、vLLM 0.30.0、Transformers 5.14.1。
训练使用基于 vLLM-Omni 提交 `f35e1b592eeb5377b14dc5764e39089aa5321df6`
构建的 wheel（依赖元数据调整为 Diffusers `0.41.0`），并未修改 LoRA kernel。当前仓库依赖仍固定 vLLM-Omni
`12e9280010afc41ce5c6a1f46900fc329cf08377`，该提交没有 Qwen-Image 2.1 pipeline。
按当前 pyproject.toml 安装后，其他模型可继续注册；运行本模型的 rollout 需要支持 2.1 的依赖版本。[上游依赖升级 PR #878](https://github.com/verl-project/verl-omni/pull/878) 使用不同的版本组合，仍需单独验证。
模型目录需要包含 processor、text_encoder、transformer、vae、scheduler。
原 checkpoint 的 processor 缺少 config.json，标准 hf_processor 会在 AutoConfig 阶段失败。
模型的 prepare_processor_files() 复用既有接口，将 text_encoder/config.json 复制到 processor；
该模型缓存目录必须可写，已有 config.json 不会被覆盖，权重文件不变。

两个 launcher 需要设置：

```bash
export QWEN21_MODEL_PATH=/models/Qwen-Image-2.1
export QWEN21_REWARD_MODEL_PATH=/models/Qwen3-VL-8B-Instruct
export QWEN21_TRAIN_DATA=/data/qwen21/train.parquet
export QWEN21_VAL_DATA=/data/qwen21/val.parquet
export QWEN21_OUTPUT_DIR=/checkpoints/qwen21-lora
```

OCR 数据使用 `prepare_ocr_data.py` 转换已有 Flow-GRPO OCR parquet。
编辑实验采用 [UniRL 的原始 image_edit 划分](https://github.com/Tencent-Hunyuan/UniRL/tree/d38b677ad642d3c7c86b1625850d86a6c406f6f1/datasets/image_edit)：
固定提交 `d38b677ad642d3c7c86b1625850d86a6c406f6f1`，19,000 条 train、1,000 条 test，均为单参考图编辑。
将两份原始 JSONL 放在同一输入目录，原图按其中 `media[].uri` 的相对路径存放，然后执行：

```bash
python examples/diffusionnft_trainer/qwen_image_21/prepare_edit_data.py \
    --input-dir /data/unirl --output-dir /data/qwen21-edit
export QWEN21_TRAIN_DATA=/data/qwen21-edit/train.parquet
export QWEN21_VAL_DATA=/data/qwen21-edit/test.parquet
```

转换按原始顺序完整读取 `train.jsonl` 和 `test.jsonl`，不抽样、去重或重新划分，也不另选长短指令。
原始 ID、划分、行号和完整记录保存在 `extra_info`；已有 `<image>` 标记不会重复添加。
原图不缩放、不重新编码，actor 和编辑 scorer 使用相同原图 bytes。
发布者划分可能复用同一原图；转换保留这些记录。
`metadata.edit_reward_score` 只保留用于溯源，不作为训练 reward。
编辑 scorer 在线接收参考图、原始指令和生成图，reward 为 edit_success 与 preservation 的几何平均。

默认配置为 rank=64、alpha=128、to_q/to_k/to_v/to_out.0、四卡 FSDP2、BF16、CPU offload、
40 步生成、全部训练时刻和 20 次更新。分辨率默认为 512，可用 QWEN21_RESOLUTION 调整；
这组默认值用于集成，不能据此声称完成 RFC 正式收敛实验。
附加 Hydra 参数放在脚本参数末尾，可覆盖默认值。

CPU 测试复用 tests/pipelines、tests/agent_loop、tests/trainer/diffusion 和 tests/utils 的现有组织。
LoRA 测试复用 test_qwen_image_lora_mapping_on_cpu.py：除 key 映射外，使用 tiny DiT 检查
NFT old adapter 导出、真实 LoRA manager 的 QKV/输出投影绑定、同 ID 权重替换及卸载后的 buffer 清理。
这些 CPU 检查不覆盖 FSDP 多卡通信或推理 kernel 的数值一致性。
GPU smoke 由 tests/gpu_smoke/run_gpu_smoke_diffusion_e2e.sh 调用；独立脚本为
 tests/special_e2e/run_diffusionnft_qwen_image_21.sh，NUM_REFERENCE_IMAGES=0..4。
它使用随机 tiny checkpoint、合成数据和已有 dummy reward，默认两步，不能用于质量、收敛或恢复验收。
真实 LoRA 加载/替换/卸载一致性、多卡更新、保存恢复及 reward 收敛须另外验收。
