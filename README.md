# Video Summarizer

使用 **本地 VLM** (OpenAI 兼容 API) 生成带时间戳的 **Markdown** 视频摘要。

基于 [Video-Browser](https://github.com/chrisx599/Video-Browser) 论文的**金字塔感知架构**实现：
- **Stage 1**: 稀疏采样识别章节边界
- **Stage 2**: 密集采样生成详细摘要

## 特性

- 🎬 **支持任意长度视频** - 自动分段处理（默认每10分钟分段）
- 📝 **带时间戳的 Markdown 输出** - 章节表格 + 详细内容
- 🎯 **智能章节分割** - AI 识别内容主题转换点
- 💾 **转录和抽帧结果缓存** - 避免重复处理
- 🔊 **使用本地 Whisper** - HTTP API 接口 (http://127.0.0.1:18181/v1/audio/)
- 🔌 **兼容任意 OpenAI API 格式 VLM** - 本地部署 (LM Studio/Ollama) 或云服务

## 安装

```bash
# 克隆仓库
git clone <repo-url>
cd video-summarizer

# 使用 uv 创建虚拟环境
uv venv

# 激活虚拟环境 (Windows)
.venv\Scripts\activate
# 激活虚拟环境 (Linux/Mac)
# source .venv/bin/activate

# 安装依赖
uv pip install -r requirements.txt
```

### 依赖说明

本项目使用以下开源库：

| 库 | 许可证 | 用途 |
|---|--------|------|
| [decord](https://github.com/dmlc/decord) | Apache-2.0 | 视频帧提取 |
| [openai](https://github.com/openai/openai-python) | Apache-2.0 | API 客户端 |
| Whisper | - | 本地 HTTP 服务语音识别 |

## 使用前提

⚠️ **运行前请确保以下服务已启动：**

### 1. 语音识别服务 (Whisper)

需要启动本地 Whisper HTTP 服务：

```bash
# 默认端口: 18181
# 可通过环境变量配置: WHISPER_API_BASE
```

### 2. 视觉语言模型服务 (VLM)

需要启动兼容 OpenAI API 格式的 VLM 服务：

| 推荐模型 | 启动方式 | 说明 |
|---------|---------|------|
| **Qwen3-VL** | 玲珑星核 | 推荐，效果较好 |
| **GLM-4.6V** | 玲珑星核 | 可用，但占用资源较大可能报错 |

```bash
# 默认端口: 1234
# 可通过环境变量配置: OPENAI_BASE_URL
```

> 💡 **提示**: 使用玲珑星核管理本地模型，可同时启动 Whisper 和 VLM 服务。
> ⚠️ **注意**: GLM-4.6V 模型体积较大，显存不足时可能导致 OOM 错误，建议优先使用 Qwen3-VL。

## 使用方法

### 方式 1: 命令行交互

```bash
python summarize.py
```

然后按提示输入视频路径和 API Key。

### 方式 2: 环境变量配置

```bash
# 设置 API Key
set OPENAI_API_KEY=your-api-key  # Windows
export OPENAI_API_KEY=your-api-key  # Linux/Mac

# 运行
python -m video_summarizer
```

### 方式 3: Python API

```python
from video_summarizer import VideoSummarizerPipeline, SummarizerConfig

config = SummarizerConfig(
    api_key="your-api-key",
    base_url="http://127.0.0.1:1234/v1"
)

pipeline = VideoSummarizerPipeline(config)
result = pipeline.summarize("./my_video.mp4")

print(f"生成了 {len(result.chapters)} 个章节")
print(f"输出文件: {result.output_path}")
```

## 配置

复制 `.env.example` 为 `.env` 并填写：

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=http://127.0.0.1:1234/v1
```

或在代码中配置：

```python
from video_summarizer import SummarizerConfig

config = SummarizerConfig(
    api_key="your-key",
    base_url="http://127.0.0.1:1234/v1",
    model="your-model-name",  # 可选，自动检测
    segment_duration=600,     # 分段时长（秒）
    sparse_frame_count=20,    # 稀疏采样帧数
    dense_fps=0.5,            # 密集采样帧率
    max_chapters=8,           # 最大章节数
)
```

### 支持的 VLM 服务

- **本地部署**: LM Studio, Ollama, vLLM
- **云服务**: 任何 OpenAI 兼容 API

## 输出示例

生成的 Markdown 文件格式：

```markdown
# 视频摘要: tech_talk.mp4

- **文件**: `tech_talk.mp4`
- **时长**: 32分15秒
- **章节数**: 5

## 整体摘要
本视频是关于大语言模型推理优化的技术分享...

## 章节概览

| 时间 | 章节 | 时长 |
|------|------|------|
| 00:00 | 开场与演讲者介绍 | 3分30秒 |
| 03:30 | KV缓存原理 | 8分45秒 |
...

## 章节详情

### [00:00 - 03:30] 开场与演讲者介绍

**摘要:** 演讲者介绍自己的背景和研究方向...

**关键台词:**
> "大家好，我是..."
> "今天我们要讨论..."

**主题标签:** 自我介绍, 议程预览
```

## 项目结构

```
video-summarizer/
├── video_summarizer/
│   ├── __init__.py
│   ├── cli.py              # 命令行界面
│   ├── config.py           # 配置管理
│   ├── pipeline.py         # 主流程
│   ├── extractors/
│   │   ├── metadata.py     # 视频元信息
│   │   ├── audio.py        # 音频转录 (HTTP API)
│   │   └── frames.py       # 帧提取 (金字塔采样)
│   ├── analyzers/
│   │   └── chapters.py     # 章节分析 (VLM)
│   ├── utils/
│   │   └── vlm_client.py   # VLM API 客户端
│   └── output/
│       └── formatter.py    # Markdown 输出
├── tests/                  # 测试文件
├── docs/plans/             # 实现计划
├── requirements.txt
├── summarize.py            # 快速启动脚本
└── README.md
```

## 运行测试

```bash
pytest tests/ -v
```

## 致谢

- [Video-Browser](https://github.com/chrisx599/Video-Browser) - 论文实现参考与灵感来源
- [decord](https://github.com/dmlc/decord) - 高效视频解码
- [Kimi](https://www.moonshot.cn/) - 项目开发过程中使用的 AI 编程助手

## License

MIT License
