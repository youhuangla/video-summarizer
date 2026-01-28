# Whisper 服务配置指南

## 问题诊断

### 症状
音频转录返回 `0 speech segments` 或 `{"error": "failed to read audio data"}`

### 根本原因
Whisper.cpp Server 未以 API 模式启动，仅提供 Web UI。

**验证方法：**
```bash
curl http://127.0.0.1:18181/
# 返回 HTML 页面 = 只有 Web UI，没有 API
curl http://127.0.0.1:18181/inference
# 返回 404 = API 未启用
```

## 正确启动方式

### Whisper.cpp Server
```bash
# 下载 whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp

# 编译 server 模式
cmake -B build -DWHISPER_BUILD_SERVER=ON
cmake --build build --config Release

# ✅ 启动 API 服务（关键：添加 --api 参数）
./build/bin/Release/whisper-server.exe \
    -p 18181 \
    -m models/ggml-base.bin \
    --api

# 参数说明：
# -p 18181          : 端口号
# -m <model>        : 模型文件路径
# --api             : 启用 HTTP API（必需！）
```

### 验证 API 是否工作
```bash
# 测试 API 端点
curl http://127.0.0.1:18181/inference \
  -F file="@test.mp3" \
  -F response_format="json"

# 预期返回：{"text": "转录文本"}
```

## 备选方案

如果无法启用 HTTP API，可以：

### 方案 1：使用 OpenAI API
```python
config = SummarizerConfig(
    api_key="your-openai-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4o"
)
```

### 方案 2：使用本地 whisper 库
修改 `audio.py` 使用 `openai-whisper` 直接转录（无需 HTTP 服务）。

### 方案 3：仅使用视觉分析
当前已实现：当音频转录失败时，仍基于视频帧生成章节摘要。

## 相关链接

- [Whisper.cpp Server 文档](https://github.com/ggerganov/whisper.cpp/blob/master/examples/server/README.md)
- [Whisper.cpp API 参考](https://github.com/ggerganov/whisper.cpp/tree/master/examples/server#api-endpoints)
