# ViT-TAD 主实验复现分析报告

---

## Step A. 主实验信息盘点与证据地图

### A1) 主实验清单

| Experiment ID | 数据集/任务场景 | 论文证据位置 |
|---------------|----------------|-------------|
| E1 | THUMOS14 时序动作检测 | Table 8, Section 4.4 |
| E2 | ActivityNet-1.3 时序动作检测 | Table 9, Section 4.4 |
| E3 | FineAction 时序动作检测 | Table 10, Section 4.4 |

**被排除实验清单（不展开）：**
- Table 1: Inner-backbone信息传播消融实验
- Table 2: Global传播策略消融实验
- Table 3: Post-backbone信息传播消融实验
- Table 4-5: ViT-TAD与其他TAD流水线对比（组件分析）
- Table 6-7: 运行时效率分析
- Figure 5: 错误分析
- Figure 6: 可视化分析

### A2) 代码仓库关键入口文件清单

| 文件路径 | 用途 |
|---------|------|
| `tools/vittad_thumos.sh` | THUMOS14训练+测试主脚本 |
| `tools/vittad_anet.sh` | ActivityNet-1.3训练+测试主脚本 |
| `tools/dist_trainval.sh` | 分布式训练启动脚本 |
| `tools/trainval.py` | 训练主入口 |
| `tools/thumos/test_af.py` | THUMOS14测试脚本 |
| `configs/vittad.py` | THUMOS14配置文件 |
| `AFSD/anet_video_cls/configs/anet.yaml` | ActivityNet-1.3主配置 |
| `AFSD/anet_video_cls/configs/membank/anet.py` | ActivityNet-1.3模型配置 |
| `AFSD/anet_video_cls/membank/train_membank.py` | ActivityNet-1.3训练脚本 |
| `AFSD/anet_video_cls/test.py` | ActivityNet-1.3测试脚本 |
| `AFSD/anet_video_cls/eval.py` | ActivityNet-1.3评测脚本 |
| `preprocess/extract_thumos_val_videos_in_8fps.sh` | THUMOS14训练集预处理 |
| `preprocess/extract_thumos_test_videos_in_8fps.sh` | THUMOS14测试集预处理 |
| `preprocess/extract_anet_frames.py` | ActivityNet-1.3帧提取 |
| `requirements/vittad.yml` | Conda环境配置 |
| `tools/Eval_Thumos14/eval_detection.py` | THUMOS14评测代码 |
| `AFSD/evaluation/eval_detection.py` | ActivityNet-1.3评测代码 |

### A3) 主实验映射表

| Experiment ID | 论文证据 | 代码证据 | 映射状态 |
|---------------|---------|---------|---------|
| E1 (THUMOS14) | Table 8, Section 4.4 | `tools/vittad_thumos.sh`, `configs/vittad.py`, `tools/thumos/test_af.py` | ✅ 完整映射 |
| E2 (ActivityNet-1.3) | Table 9, Section 4.4 | `tools/vittad_anet.sh`, `AFSD/anet_video_cls/configs/anet.yaml`, `AFSD/anet_video_cls/configs/membank/anet.py` | ✅ 完整映射 |
| E3 (FineAction) | Table 10, Section 4.4 | 【未映射】 | ❌ 缺少训练/测试脚本、配置文件、数据处理代码 |

---

## 0. 主实验复现结论总览

| Experiment ID | 场景/数据集 | 任务 | 论文主指标与数值 | 代码入口 | 复现难度 | 可复现性判断 | 主要风险点 |
|---------------|------------|------|-----------------|---------|---------|-------------|-----------|
| E1 | THUMOS14 | 时序动作检测 | Avg mAP=69.5% (Table 8) | `bash tools/vittad_thumos.sh` | 中 | 部分可复现 | 预训练权重需手动下载；8×TITAN Xp GPU需求高 |
| E2 | ActivityNet-1.3 | 时序动作检测 | Avg mAP=37.40% (Table 9) | `bash tools/vittad_anet.sh` | 中 | 部分可复现 | 预训练权重需手动下载；视频数据量大；4×GPU需求 |
| E3 | FineAction | 时序动作检测 | Avg mAP=17.20% (Table 10) | 【未提供】 | 高 | 不可复现 | 代码仓库未提供FineAction相关代码/配置 |

---

## 1. 论文概述

### 1.1 标题
**Adapting Short-Term Transformers for Action Detection in Untrimmed Videos (ViT-TAD)**
[Paper: 标题]

### 1.2 方法一句话总结
ViT-TAD是一个端到端的时序动作检测框架，输入为未剪辑的长视频（RGB帧序列），输出为检测到的动作实例（起始时间、结束时间、类别、置信度）；核心机制是将预训练的短期ViT模型适配为统一的长视频Transformer，通过inner-backbone信息传播模块（跨snippet的全局注意力）和post-backbone信息传播模块（时序Transformer层）实现多snippet间的时序信息交互。
[Paper: Abstract, Section 3]

### 1.3 核心贡献
1. **首个基于plain ViT的端到端TAD框架**：ViT-TAD是第一个利用plain ViT backbone进行端到端时序动作检测的方法 [Paper: Section 1, Contributions]
2. **Inner-backbone信息传播模块**：设计跨snippet传播模块（Local/Global Propagation Block），在backbone内部实现多snippet时序特征交互 [Paper: Section 3.1]
3. **Post-backbone信息传播模块**：提出时序Transformer层进行clip级别建模，扩大时序感受野 [Paper: Section 3.2]
4. **端到端训练机制**：通过checkpoint训练策略在有限GPU显存下实现端到端训练 [Paper: Section 4.2, Table 7]
5. **【归纳】充分利用VideoMAE预训练**：方法能够直接利用VideoMAE的强大预训练权重，无需额外预训练

---

## 2. 主实验复现详解

---

### 【E1 主实验标题：THUMOS14时序动作检测】

#### A. 这个主实验在回答什么问题
- **实验目的**：验证ViT-TAD在THUMOS14数据集上的时序动作检测性能，与现有SOTA方法进行对比
- **核心结论**：ViT-TAD with ViT-B在THUMOS14上达到69.5% average mAP，超越除VideoMAE V2 (ViT-G)外的所有方法
- **论文证据位置**：Table 8 (Comparison with state-of-the-art methods on THUMOS14), Section 4.4
[Paper: Table 8, Section 4.4]

#### B. 实验任务与工作原理

**任务定义**：
- **输入**：未剪辑的长视频（RGB帧序列），采样率8 FPS，帧分辨率160×160
- **输出**：检测到的动作实例列表，每个实例包含(start_time, end_time, category, confidence)
- **预测目标**：定位视频中所有动作实例的时间边界并识别其类别（20类动作+背景）
- **约束条件**：temporal window为32秒（256帧@8FPS），覆盖99.7%的动作实例
[Paper: Section 4, Implementation Details]

**方法关键流程**：
1. **数据采样**：将视频按8 FPS采样，划分为32秒的temporal window（256帧）
2. **Snippet划分**：将256帧划分为16个snippet，每个snippet 16帧
3. **Backbone特征提取**：使用VideoMAE预训练的ViT-B，通过inner-backbone传播模块实现跨snippet交互
4. **Post-backbone处理**：3层时序Transformer进行clip级别建模
5. **TAD Head**：使用BasicTAD的FCOS Head进行动作检测
6. **损失函数**：Focal Loss (分类) + IoU Loss (回归)
7. **后处理**：NMS + 阈值过滤
[Paper: Section 3, Figure 2]

**最终设置**：
- Backbone: ViT-B (VideoMAE预训练)
- 帧分辨率: 160×160
- 帧率: 8 FPS
- Temporal window: 256帧 (32秒)
- Snippets: 16个，每个16帧
- Inner-backbone: 4个evenly placed global propagation blocks
- Post-backbone: 3层Transformer
- TAD Head: BasicTAD FCOS Head
[Paper: Section 4, Table 8; Repo: configs/vittad.py]

**实例说明**：
对于一个5分钟的THUMOS14测试视频，系统会：
1. 按8 FPS采样得到2400帧
2. 使用滑动窗口（overlap_ratio=0.25）切分为多个256帧的clip
3. 每个clip经过ViT-TAD处理得到动作预测
4. 合并所有clip的预测，通过NMS去重
5. 输出最终的动作检测结果（如"TennisSwing: 10.5s-12.3s, confidence=0.95"）

#### C. 数据

**数据集名称与来源**：
- **名称**：THUMOS14
- **来源**：官方下载或通过BasicTAD仓库获取
- **README说明**：`For THUMOS14, please check BasicTAD for downloading videos.`
[Repo: README.md]

**数据许可/访问限制**：
- 【未知】论文和README未明确说明数据许可

**数据结构示例**：
```
data/
├── thumos_video_val/          # 原始验证集视频（训练用）
│   ├── video_validation_0000051.mp4
│   └── ...
├── thumos_video_test/         # 原始测试集视频
│   ├── video_test_0000004.mp4
│   └── ...
└── thumos/
    ├── video_8fps/            # 预处理后的8FPS视频
    │   ├── validation/
    │   │   ├── video_validation_0000051.mp4
    │   │   └── ...
    │   └── test/
    │       ├── video_test_0000004.mp4
    │       └── ...
    └── annotations/
        ├── val.json           # 训练集标注
        ├── test.json          # 测试集标注
        └── detclasslist.txt   # 类别列表
```
[Repo: data/thumos/, configs/vittad.py]

**标注JSON格式**（`val.json`/`test.json`）：
```json
{
  "database": {
    "video_validation_0000281": {
      "annotations": [
        {"segment": [169.7, 175.3], "label": "GolfSwing"},
        {"segment": [221.6, 225.3], "label": "GolfSwing"}
      ],
      "duration": 227.76666666666668,
      "resolution": "320x180",
      "subset": "val"
    }
  }
}
```
[Repo: data/thumos/annotations/val.json]

**Dataset类返回内容**（`Thumos14Dataset.__getitem__`）：
- `imgs`: Tensor [T, H, W, C] 视频帧
- `gt_segments`: Tensor [N, 2] 动作时间段
- `gt_labels`: Tensor [N] 动作类别
- `gt_segments_ignore`: Tensor [M, 2] 忽略的时间段（Ambiguous）
[Repo: vedatad/datasets/thumos14.py]

**数据量**：
- 训练集（validation split）：200个视频
- 测试集（test split）：213个视频
- 类别数：20类动作 + 1背景
[Paper: Section 4, Datasets and Evaluation Metric]

**训练集构建**：
1. 从原始视频转换为8 FPS
2. 使用`TemporalRandomCrop`随机裁剪256帧
3. 帧resize到short-180，然后随机裁剪160×160
4. 数据增强：PhotoMetricDistortion, Rotate, SpatialRandomFlip
5. Normalize: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
[Repo: configs/vittad.py, data.train.pipeline]

**测试集构建**：
1. 从原始视频转换为8 FPS
2. 使用`OverlapCropAug`滑动窗口裁剪（overlap_ratio=0.25）
3. 帧resize到short-180，中心裁剪160×160
4. Normalize同训练集
[Repo: configs/vittad.py, data.val.pipeline]

**预处理与缓存**：
```bash
# 训练集预处理（生成8FPS视频）
bash preprocess/extract_thumos_val_videos_in_8fps.sh
# 输出: ./data/thumos/video_8fps/validation/*.mp4

# 测试集预处理
bash preprocess/extract_thumos_test_videos_in_8fps.sh
# 输出: ./data/thumos/video_8fps/test/*.mp4
```
[Repo: preprocess/extract_thumos_val_videos_in_8fps.sh, preprocess/extract_thumos_test_videos_in_8fps.sh]

#### D. 模型与依赖

**基础模型/Backbone**：
- **名称**：ViT-B (vit_base_patch16_224)
- **版本**：VideoMAE v2预训练权重
- **下载方式**：从VideoMAE v2官方仓库或百度网盘下载
- **放置路径**：`./pretrained/vit-b.pth`
- **配置键**：`model.backbone.finetune`
[Repo: configs/vittad.py line 93; README.md]

**关键模块**：
| 模块 | 配置 | 代码位置 |
|-----|------|---------|
| Backbone | ViT-B, patch_size=16, embed_dim=768, depth=12, num_heads=12 | `vedatad/models/backbones/videomae.py` |
| Global Propagation | 4 blocks, evenly placed | `configs/vittad.py` glob_attn参数 |
| Post-backbone Transformer | 3 layers, dim=768, num_heads=6 | `configs/vittad.py` neck[0] |
| TDM Neck | 5 layers, kernel_size=3, stride=2 | `configs/vittad.py` neck[1] |
| FCOS Head | num_classes=21, in_channels=768 | `configs/vittad.py` head |
[Repo: configs/vittad.py, vedatad/models/]

**训练策略**：
- **优化器**：SGD, lr=0.01, momentum=0.9, weight_decay=0.0001
- **学习率调度**：CosineRestartLrScheduler, periods=[100]*12, warmup_iters=500
- **Batch size**：2 per GPU × 8 GPUs = 16
- **训练轮数**：1200 epochs
- **混合精度**：【未知】代码中未明确
- **Checkpoint训练**：use_checkpoint=True（节省显存）
[Repo: configs/vittad.py lines 156, 182-194, 199]

**随机性控制**：
- **seed**：10
- **deterministic**：True
[Repo: configs/vittad.py lines 202, 206]

#### E. 评价指标与论文主表预期结果

**指标定义**：
- **mAP@tIoU**：在给定tIoU阈值下的mean Average Precision
- **tIoU阈值**：[0.3, 0.4, 0.5, 0.6, 0.7]
- **Avg**：上述5个阈值的平均mAP
- **计算方式**：使用ActivityNet官方评测代码
[Paper: Section 4; Repo: tools/Eval_Thumos14/eval_detection.py]

**论文主结果数值**（Table 8, ViT-TAD with ViT-B 160×160）：

| tIoU | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Avg |
|------|-----|-----|-----|-----|-----|-----|
| mAP (%) | 85.1 | 80.9 | 74.2 | 61.8 | 45.4 | 69.5 |

[Paper: Table 8]

**复现预期**：以论文主表数值为准，即Avg mAP = 69.5%

#### F. 环境与硬件需求

**软件环境**：
- Python: 3.8.18
- PyTorch: 1.10.1
- CUDA: 11.3
- torchvision: 0.11.2
- timm: 0.4.12
- mmcv-full: 1.7.1
- transformers: 4.38.2
- decord: 0.6.0
- av: 11.0.0
[Repo: requirements/vittad.yml]

**硬件要求**：
- **GPU**：8× TITAN Xp（论文明确说明）
- **显存**：约9GB per GPU（使用checkpoint训练策略）
- **磁盘**：THUMOS14原始视频约20GB，8FPS视频约5GB
[Paper: Section 4, Implementation Details; Repo: configs/vittad.py Table 7]

**训练时长**：【未知】论文和README未明确说明

#### G. 可直接照做的主实验复现步骤

**步骤1：获取代码与安装依赖**
```bash
# 克隆仓库
git clone https://github.com/MCG-NJU/ViT-TAD.git
cd ViT-TAD

# 创建conda环境
conda env create -f requirements/vittad.yml
conda activate vittad

# 安装项目
pip install -v -e .
```
- **目的**：搭建运行环境
- **预期产物**：可用的vittad conda环境
[Repo: README.md]

**步骤2：获取数据**
```bash
# 创建数据目录
mkdir -p data/thumos_video_val data/thumos_video_test

# 下载THUMOS14视频（从BasicTAD或官方渠道）
# 将validation视频放入 data/thumos_video_val/
# 将test视频放入 data/thumos_video_test/

# 验证数据
ls data/thumos_video_val/*.mp4 | wc -l  # 应为200
ls data/thumos_video_test/*.mp4 | wc -l  # 应为213
```
- **目的**：准备原始视频数据
- **关键配置**：视频路径在预处理脚本中硬编码
[Repo: README.md, preprocess/*.sh]

**步骤3：数据预处理**
```bash
# 创建输出目录
mkdir -p data/thumos/video_8fps/validation
mkdir -p data/thumos/video_8fps/test

# 转换训练集视频为8FPS
bash preprocess/extract_thumos_val_videos_in_8fps.sh

# 转换测试集视频为8FPS
bash preprocess/extract_thumos_test_videos_in_8fps.sh
```
- **目的**：将原始视频转换为8 FPS
- **预期产物**：
  - `data/thumos/video_8fps/validation/*.mp4`（200个文件）
  - `data/thumos/video_8fps/test/*.mp4`（213个文件）
[Repo: preprocess/extract_thumos_val_videos_in_8fps.sh, preprocess/extract_thumos_test_videos_in_8fps.sh]

**步骤4：获取预训练权重**
```bash
# 创建预训练权重目录
mkdir -p pretrained

# 方式1：从VideoMAE v2官方下载
# https://github.com/OpenGVLab/VideoMAEv2

# 方式2：从百度网盘下载（密码：8tw7）
# https://pan.baidu.com/s/1z0nf6nU9Iq1GM2Lyuppjyw

# 将权重文件放置到
# pretrained/vit-b.pth
```
- **目的**：获取VideoMAE预训练的ViT-B权重
- **预期产物**：`pretrained/vit-b.pth`
[Repo: README.md, configs/vittad.py line 93]

**步骤5：训练**
```bash
# 分布式训练（8 GPU）
bash tools/dist_trainval.sh configs/vittad.py "0,1,2,3,4,5,6,7"

# 或单GPU训练【推断】
# CUDA_VISIBLE_DEVICES=0 python tools/trainval.py configs/vittad.py
```
- **目的**：训练ViT-TAD模型
- **关键参数**：
  - 配置文件：`configs/vittad.py`
  - GPU：8卡
  - Batch size：2 per GPU
  - Epochs：1200
- **预期产物**：
  - `workdir/vittad/epoch_XXX_weights.pth`（每100 epoch保存）
  - `workdir/vittad/YYYYMMDD_HHMMSS.log`（训练日志）
[Repo: tools/vittad_thumos.sh, tools/dist_trainval.sh]

**步骤6：测试与评测**
```bash
# 测试（遍历epoch 200-1200）
for i in {2..12..1}; do
    epoch="${i}00_weights"
    echo "Testing epoch_$epoch"
    CUDA_VISIBLE_DEVICES=0 python tools/thumos/test_af.py \
        --framerate 8 \
        configs/vittad.py \
        workdir/vittad/epoch_${epoch}.pth
done
```
- **目的**：在测试集上评测模型
- **关键参数**：
  - `--framerate 8`：帧率
  - checkpoint路径
- **预期产物**：
  - `tools/output_epoch_XXX_weights.json`：检测结果JSON
  - 终端输出mAP@各tIoU阈值
[Repo: tools/vittad_thumos.sh, tools/thumos/test_af.py]

**步骤7：主表指标对齐**
测试脚本会自动调用评测代码并输出：
```
mAP at tIoU 0.3 is XX.X
mAP at tIoU 0.4 is XX.X
mAP at tIoU 0.5 is XX.X
mAP at tIoU 0.6 is XX.X
mAP at tIoU 0.7 is XX.X
sum average: XX.X
```
对比论文Table 8中ViT-TAD (ViT-B, 160×160)的结果：Avg = 69.5%
[Repo: tools/thumos/test_af.py lines 369-381]

#### H. 可复现性判断

**结论**：部分可复现

**依据清单**：
| 项目 | 状态 | 说明 |
|-----|------|------|
| 数据可得性 | ⚠️ | THUMOS14需从BasicTAD或官方渠道获取，非直接提供 |
| 预训练权重 | ⚠️ | 需手动从VideoMAE v2或百度网盘下载 |
| 训练脚本 | ✅ | 完整提供 |
| 测试脚本 | ✅ | 完整提供 |
| 配置文件 | ✅ | 完整提供 |
| 评测代码 | ✅ | 完整提供 |
| 硬件需求 | ⚠️ | 需要8×TITAN Xp，门槛较高 |
| 环境配置 | ✅ | 提供完整的conda yml文件 |

**补救路径**：
1. **数据获取**：参考BasicTAD仓库的数据下载说明
2. **预训练权重**：使用百度网盘链接（密码：8tw7）
3. **硬件限制**：【经验】可尝试减小batch size或使用梯度累积，但可能影响最终性能
4. **只跑评测**：如果作者提供训练好的checkpoint，可直接运行测试脚本

#### I. 主实验专属排错要点

1. **视频路径约定**：
   - 原始视频必须放在`data/thumos_video_val/`和`data/thumos_video_test/`
   - 预处理后视频在`data/thumos/video_8fps/validation/`和`data/thumos/video_8fps/test/`
   - 视频文件名格式：`video_validation_XXXXXXX.mp4`或`video_test_XXXXXXX.mp4`

2. **预训练权重路径**：
   - 必须放在`./pretrained/vit-b.pth`
   - 配置文件中硬编码了此路径

3. **标注文件**：
   - 仓库已提供`data/thumos/annotations/val.json`和`test.json`
   - 无需额外下载

4. **ffmpeg依赖**：
   - 预处理脚本依赖ffmpeg
   - 确保ffmpeg已安装且在PATH中

5. **分布式训练**：
   - 使用`torch.distributed.launch`
   - 端口随机选择（29500 + RANDOM % 100）
   - 如遇端口冲突，可修改`tools/dist_trainval.sh`

6. **测试脚本GPU设置**：
   - `tools/thumos/test_af.py`中硬编码了`CUDA_VISIBLE_DEVICES=3`
   - 需根据实际情况修改

---

### 【E2 主实验标题：ActivityNet-1.3时序动作检测】

#### A. 这个主实验在回答什么问题
- **实验目的**：验证ViT-TAD在大规模数据集ActivityNet-1.3上的时序动作检测性能
- **核心结论**：ViT-TAD with ViT-B达到37.40% average mAP，在端到端方法中达到SOTA
- **论文证据位置**：Table 9 (Comparison with state-of-the-art methods on ActivityNet-1.3), Section 4.4
[Paper: Table 9, Section 4.4]

#### B. 实验任务与工作原理

**任务定义**：
- **输入**：未剪辑的长视频，统一采样为768帧，帧分辨率160×160
- **输出**：检测到的动作实例列表，每个实例包含(start_time, end_time, category, confidence)
- **预测目标**：定位视频中所有动作实例的时间边界并识别其类别（200类动作+背景）
- **特殊处理**：使用二分类proposal + 外部分类器（CUHK/GT）获得最终类别
[Paper: Section 4, Section 4.4]

**方法关键流程**：
1. **数据采样**：将每个视频统一采样为768帧
2. **Snippet划分**：将768帧划分为48个snippet，每个snippet 16帧
3. **Backbone特征提取**：使用VideoMAE预训练的ViT-B
4. **TAD Head**：使用AFSD的两阶段anchor-free方法
5. **分类策略**：预测二分类proposal，使用外部分类器（CUHK或GT）获得类别
[Paper: Section 4, Implementation Details]

**最终设置**：
- Backbone: ViT-B (VideoMAE预训练)
- 帧分辨率: 160×160
- 总帧数: 768帧
- Snippets: 48个，每个16帧
- TAD Head: AFSD两阶段anchor-free
- 优化器: AdamW, lr=0.0002
[Paper: Section 4; Repo: AFSD/anet_video_cls/configs/anet.yaml, AFSD/anet_video_cls/configs/membank/anet.py]

**实例说明**：
对于一个ActivityNet-1.3视频：
1. 统一采样为768帧（无论原始长度）
2. 划分为48个snippet进行处理
3. 预测二分类proposal（是否为动作）
4. 使用CUHK分类器或GT标签获得动作类别
5. 输出最终检测结果

#### C. 数据

**数据集名称与来源**：
- **名称**：ActivityNet-1.3
- **来源**：官方下载或通过TALLFormer仓库获取
- **README说明**：`For ActivityNet-1.3, please check TALLFormer for downloading videos.`
[Repo: README.md]

**数据许可/访问限制**：
- 【未知】论文和README未明确说明

**数据结构示例**：
```
data/
├── anet/
│   ├── anet_train/              # 原始训练视频
│   │   ├── 2aHetC-N-P4.avi
│   │   └── ...
│   ├── anet_val/                # 原始验证视频
│   │   ├── NjTk2naIaac.avi
│   │   └── ...
│   └── afsd_anet_768frames/     # 预处理后的帧
│       ├── training/
│       │   ├── aHetC-N-P4/
│       │   │   ├── image_00001.jpg
│       │   │   └── ...
│       │   └── ...
│       └── validation/
│           └── ...
└── annots/
    └── anet/
        ├── video_info_train_val.json
        ├── activity_net_1_3_new.json
        ├── cuhk_val_simp_share.json
        ├── action_name.txt
        └── class_name.txt
```
[Repo: data/annots/anet/, AFSD/anet_video_cls/configs/anet.yaml]

**数据量**：
- 训练集：10,024个视频
- 验证集：4,926个视频
- 类别数：200类动作
[Paper: Section 4, Datasets and Evaluation Metric]

**训练集构建**：
1. 从原始视频提取768帧（按比例采样）
2. 随机裁剪160×160
3. 数据增强：PhotoMetricDistortion, Rotate
4. Normalize: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
[Repo: AFSD/anet_video_cls/configs/membank/anet.py, AFSD/anet_video_cls/membank/membank_dataset.py]

**测试集构建**：
1. 从原始视频提取768帧
2. 中心裁剪160×160
3. Normalize同训练集
[Repo: AFSD/anet_video_cls/test.py]

**预处理与缓存**：
```bash
# 训练集帧提取
python preprocess/extract_anet_frames.py 1 \
    --video_dir ./data/anet/anet_train \
    --output_dir ./data/anet/afsd_anet_768frames/training

# 验证集帧提取
python preprocess/extract_anet_frames.py 1 \
    --video_dir ./data/anet/anet_val \
    --output_dir ./data/anet/afsd_anet_768frames/validation
```
- **输出**：每个视频一个文件夹，包含768张jpg图片
[Repo: README.md, preprocess/extract_anet_frames.py]

#### D. 模型与依赖

**基础模型/Backbone**：
- **名称**：ViT-B (vit_base_patch16_224)
- **版本**：VideoMAE v2预训练权重
- **放置路径**：`./pretrained/vit-b.pth`
- **配置键**：`model.backbone.finetune`
[Repo: AFSD/anet_video_cls/configs/membank/anet.py line 38]

**关键模块**：
| 模块 | 配置 | 代码位置 |
|-----|------|---------|
| Backbone | ViT-B, n_segment=384 | `AFSD/anet_video_cls/configs/membank/anet.py` |
| SRM Neck | in_channels=768 | `AFSD/anet_video_cls/configs/membank/anet.py` neck[0] |
| Transformer Neck | 3 layers, dim=768, num_heads=16 | `AFSD/anet_video_cls/configs/membank/anet.py` neck[1] |
| TDM | 5 stages | `AFSD/anet_video_cls/configs/membank/anet.py` neck[2] |
| FPN | 6 levels | `AFSD/anet_video_cls/configs/membank/anet.py` neck[3] |
| Action Head | num_classes=201 | `AFSD/anet_video_cls/configs/membank/anet.py` action_head |
[Repo: AFSD/anet_video_cls/configs/membank/anet.py]

**训练策略**：
- **优化器**：Adam, lr=1e-4, weight_decay=1e-4
- **Backbone学习率倍率**：0.04
- **学习率调度**：MultiStepLR, milestones=[8, 10], gamma=0.1
- **Batch size**：1 per GPU × 4 GPUs = 4
- **训练轮数**：10 epochs
[Repo: AFSD/anet_video_cls/configs/anet.yaml, AFSD/anet_video_cls/configs/membank/anet.py, tools/vittad_anet.sh]

**随机性控制**：
- **seed**：2020
[Repo: AFSD/anet_video_cls/configs/anet.yaml line 33]

#### E. 评价指标与论文主表预期结果

**指标定义**：
- **mAP@tIoU**：在给定tIoU阈值下的mean Average Precision
- **tIoU阈值**：[0.5:0.05:0.95]（10个阈值）
- **Avg**：10个阈值的平均mAP
- **计算方式**：使用ActivityNet官方评测代码
[Paper: Section 4; Repo: AFSD/evaluation/eval_detection.py]

**论文主结果数值**（Table 9, ViT-TAD with ViT-B 160×160）：

| tIoU | 0.5 | 0.75 | 0.95 | Avg |
|------|-----|------|------|-----|
| mAP (%) | 55.87 | 38.47 | 8.80 | 37.40 |

[Paper: Table 9]

**复现预期**：以论文主表数值为准，即Avg mAP = 37.40%

#### F. 环境与硬件需求

**软件环境**：同E1
[Repo: requirements/vittad.yml]

**硬件要求**：
- **GPU**：4× GPU（论文提到8× TITAN Xp用于THUMOS14，ActivityNet使用4 GPU）
- **显存**：【推断】约10-12GB per GPU
- **磁盘**：ActivityNet-1.3原始视频约500GB，提取帧后约100GB
[Repo: tools/vittad_anet.sh --ngpu 4]

**训练时长**：【未知】

#### G. 可直接照做的主实验复现步骤

**步骤1：获取代码与安装依赖**
同E1步骤1

**步骤2：获取数据**
```bash
# 创建数据目录
mkdir -p data/anet/anet_train data/anet/anet_val

# 下载ActivityNet-1.3视频（从TALLFormer或官方渠道）
# 将训练视频放入 data/anet/anet_train/
# 将验证视频放入 data/anet/anet_val/
```
[Repo: README.md]

**步骤3：数据预处理**
```bash
# 提取训练集帧（768帧/视频）
python preprocess/extract_anet_frames.py 1 \
    --video_dir ./data/anet/anet_train \
    --output_dir ./data/anet/afsd_anet_768frames/training

# 提取验证集帧
python preprocess/extract_anet_frames.py 1 \
    --video_dir ./data/anet/anet_val \
    --output_dir ./data/anet/afsd_anet_768frames/validation
```
- **预期产物**：
  - `data/anet/afsd_anet_768frames/training/*/image_*.jpg`
  - `data/anet/afsd_anet_768frames/validation/*/image_*.jpg`
[Repo: README.md, preprocess/extract_anet_frames.py]

**步骤4：获取预训练权重**
同E1步骤4

**步骤5：训练**
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12447

python3 AFSD/anet_video_cls/membank/train_membank.py \
    AFSD/anet_video_cls/configs/anet.yaml \
    --lw=1 --cw=1 --piou=0.6 --ssl 0. \
    --checkpoint_path workdir/vittad/anet \
    --addi_config AFSD/anet_video_cls/configs/membank/anet.py \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --ngpu 4
```
- **预期产物**：
  - `workdir/vittad/anet/checkpoint-{epoch}.ckpt`
  - `workdir/vittad/anet/log.txt`
[Repo: tools/vittad_anet.sh]

**步骤6：测试**
```bash
epoch=10
output_json=anet-epoch_${epoch}-anet_rgb.json

python3 AFSD/anet_video_cls/test.py \
    AFSD/anet_video_cls/configs/anet.yaml \
    --nms_sigma=0.85 --ngpu=4 \
    --checkpoint_path workdir/vittad/anet/checkpoint-${epoch}.ckpt \
    --output_json=$output_json \
    --addi_config AFSD/anet_video_cls/configs/membank/anet.py
```
- **预期产物**：`output/${output_json}`
[Repo: tools/vittad_anet.sh]

**步骤7：评测**
```bash
for classifier in "builtin" "cuhk" "gt"; do
    output_json=anet-epoch_${epoch}-anet_rgb-${classifier}.json
    python3 AFSD/anet_video_cls/eval.py output/$output_json \
        --workspace workdir/vittad/anet --epoch ${epoch}_${classifier}
done
```
- **预期输出**：
```
mAP at tIoU 0.5 is XX.XX
mAP at tIoU 0.75 is XX.XX
mAP at tIoU 0.95 is XX.XX
Average mAP: XX.XX
```
[Repo: tools/vittad_anet.sh, AFSD/anet_video_cls/eval.py]

**步骤8：主表指标对齐**
对比论文Table 9中ViT-TAD (ViT-B, 160×160)的结果：
- 使用CUHK分类器时：Avg mAP ≈ 37.40%
[Paper: Table 9]

#### H. 可复现性判断

**结论**：部分可复现

**依据清单**：
| 项目 | 状态 | 说明 |
|-----|------|------|
| 数据可得性 | ⚠️ | ActivityNet-1.3需从TALLFormer或官方渠道获取，数据量大 |
| 预训练权重 | ⚠️ | 需手动下载 |
| 训练脚本 | ✅ | 完整提供 |
| 测试脚本 | ✅ | 完整提供 |
| 配置文件 | ✅ | 完整提供 |
| 评测代码 | ✅ | 完整提供 |
| 外部分类器 | ✅ | CUHK分类器结果已提供 |

**补救路径**：
1. **数据获取**：参考TALLFormer仓库的数据下载说明
2. **存储空间**：ActivityNet-1.3数据量大，需要足够的磁盘空间
3. **只跑评测**：如果作者提供训练好的checkpoint，可直接运行测试脚本

#### I. 主实验专属排错要点

1. **视频路径约定**：
   - 原始视频在`data/anet/anet_train/`和`data/anet/anet_val/`
   - 提取帧在`data/anet/afsd_anet_768frames/`

2. **视频名称处理**：
   - 代码中对视频名称有特殊处理（去除前缀`v_`或`0000000`）
   - 见`AFSD/anet_video_cls/test.py` lines 102-105

3. **测试脚本GPU设置**：
   - `AFSD/anet_video_cls/test.py`中硬编码了`torch.cuda.set_device(pid+4)`
   - 需根据实际GPU配置修改

4. **分类器选择**：
   - 论文使用CUHK分类器获得最终结果
   - `cuhk_val_simp_share.json`已提供在`data/annots/anet/`

---

### 【E3 主实验标题：FineAction时序动作检测】

#### A. 这个主实验在回答什么问题
- **实验目的**：验证ViT-TAD在细粒度动作检测数据集FineAction上的性能
- **核心结论**：ViT-TAD with ViT-B达到17.20% average mAP
- **论文证据位置**：Table 10 (Comparison with state-of-the-art methods on FineAction), Section 4.4
[Paper: Table 10, Section 4.4]

#### B-I. 【不可复现】

**结论**：不可复现

**原因**：
代码仓库中**未提供**FineAction数据集的相关代码、配置文件和训练/测试脚本。

**证据**：
1. README.md只提到THUMOS14和ActivityNet-1.3的复现步骤
2. 搜索仓库中所有文件，未找到任何包含"fineaction"、"fine_action"或"fine-action"的代码
3. 没有FineAction的配置文件
4. 没有FineAction的数据预处理脚本

**论文主结果数值**（Table 10, ViT-TAD with ViT-B 160×160）：

| tIoU | 0.5 | 0.75 | 0.95 | Avg |
|------|-----|------|------|-----|
| mAP (%) | 32.61 | 15.85 | 2.68 | 17.20 |

[Paper: Table 10]

**补救路径**：
1. 联系作者获取FineAction的代码和配置
2. 参考THUMOS14的代码结构，自行实现FineAction的数据加载和训练流程【需要较多工作量】
3. 等待作者更新仓库

---

## 3. 主实验一致性检查

### 论文主表指标是否能被仓库脚本直接产出同款结果

| 实验 | 论文指标 | 仓库脚本 | 一致性 |
|-----|---------|---------|--------|
| E1 (THUMOS14) | Avg mAP=69.5% | `tools/thumos/test_af.py` 输出 "sum average" | ✅ 一致 |
| E2 (ActivityNet-1.3) | Avg mAP=37.40% | `AFSD/anet_video_cls/eval.py` 输出 "Average mAP" | ✅ 一致 |
| E3 (FineAction) | Avg mAP=17.20% | 【无脚本】 | ❌ 无法验证 |

### 多个主实验是否共享同一套预处理与评测入口

- **预处理**：E1和E2使用不同的预处理脚本
  - E1: `preprocess/extract_thumos_*_videos_in_8fps.sh`（视频转8FPS）
  - E2: `preprocess/extract_anet_frames.py`（提取768帧）
  
- **评测**：E1和E2使用不同的评测入口
  - E1: `tools/Eval_Thumos14/eval_detection.py`
  - E2: `AFSD/evaluation/eval_detection.py`
  
- **共用组件**：
  - 预训练权重：`pretrained/vit-b.pth`
  - Backbone代码：`vedatad/models/backbones/videomae.py`
  - 环境配置：`requirements/vittad.yml`

### 最小复现路径

**如果只想最快验证主表分数，建议按以下顺序：**

1. **优先复现E1 (THUMOS14)**
   - 原因：数据量小（约20GB），训练时间相对较短，代码完整
   - 命令：
   ```bash
   # 环境准备
   conda env create -f requirements/vittad.yml && conda activate vittad && pip install -v -e .
   
   # 数据准备（假设已下载原始视频）
   bash preprocess/extract_thumos_val_videos_in_8fps.sh
   bash preprocess/extract_thumos_test_videos_in_8fps.sh
   
   # 训练
   bash tools/dist_trainval.sh configs/vittad.py "0,1,2,3,4,5,6,7"
   
   # 测试
   CUDA_VISIBLE_DEVICES=0 python tools/thumos/test_af.py --framerate 8 configs/vittad.py workdir/vittad/epoch_1200_weights.pth
   ```

2. **其次复现E2 (ActivityNet-1.3)**
   - 原因：数据量大，但代码完整
   - 需要更多存储空间和下载时间

3. **E3 (FineAction) 目前无法复现**

---

## 4. 未知项与需要补充的最小信息

| 问题 | 必要性 | 缺失后果 |
|-----|--------|---------|
| FineAction的训练/测试代码和配置 | 必须 | 无法复现E3实验 |
| 训练好的checkpoint是否公开 | 建议 | 无法快速验证结果，必须从头训练 |
| 具体训练时长 | 建议 | 无法估算复现所需时间 |
| 是否使用混合精度训练 | 可选 | 可能影响显存使用和训练速度 |

**最关键问题**：
1. **FineAction代码缺失**：这是论文三个主实验之一，但仓库完全没有相关代码。建议联系作者或等待仓库更新。
2. **预训练权重下载**：百度网盘链接可能对海外用户不友好，建议作者提供其他下载方式（如Google Drive或Hugging Face）。

---

*报告生成时间：2026-01-28*
*基于仓库：https://github.com/MCG-NJU/ViT-TAD (main branch)*
