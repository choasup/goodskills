# AMD GPU 部署 HuggingFace 模型为 OpenAI API

在 AMD GPU (ROCm) 服务器上部署 HuggingFace 模型，提供 OpenAI 兼容的 `/v1/chat/completions` API。  
采用 **transformers + FastAPI** 方案，适用于 vLLM 不支持的新架构模型。

---

## 适用场景

- AMD GPU + ROCm 环境（Radeon Instinct / MI 系列）
- vLLM 不支持目标模型架构时的备选方案
- 需要 OpenAI 兼容 API，支持流式和非流式输出
- 支持 `<think>` 思维链模型（自动分离 `reasoning_content` 和 `content`）

---

## 快速开始

### 1. 环境检查

SSH 到服务器后依次确认：

```bash
# GPU 状态
rocm-smi

# PyTorch + ROCm
python3 -c "import torch; print(torch.__version__); print('GPU count:', torch.cuda.device_count())"

# transformers 版本（新架构需要较新版本）
python3 -c "import transformers; print(transformers.__version__)"

# 模型文件完整性
ls -lh /path/to/model/*.safetensors
```

如果 transformers 版本过低：
```bash
pip install --upgrade transformers
```

### 2. 安装依赖

```bash
pip install fastapi uvicorn
```

### 3. 部署

将 `scripts/api_server.py` 上传到服务器，修改开头两个变量：

```python
MODEL_PATH = "/path/to/your/model"   # 模型本地路径
MODEL_NAME = "your-model-name"        # API 中显示的模型名
```

#### 模型加载类的选择

| 模型类型 | 使用的类 |
|---------|---------|
| 纯文本模型（GPT/LLaMA/Qwen等） | `AutoModelForCausalLM` |
| 多模态模型（Qwen3.5/LLaVA等） | `AutoModelForImageTextToText` |

看模型 `config.json` 中的 `architectures` 字段判断。默认使用 `AutoModelForImageTextToText`，如果是纯文本模型需要手动改。

### 4. 启动

```bash
# 后台启动（推荐）
nohup python3 api_server.py > api_server.log 2>&1 &

# 指定单卡启动
CUDA_VISIBLE_DEVICES=0 python3 api_server.py

# 前台启动（调试用）
python3 api_server.py
```

### 5. 验证

#### 服务器上验证

> ⚠️ 如果服务器有 Privoxy 代理，**必须**加 `--noproxy '*'`

```bash
# 模型列表
curl --noproxy '*' -s http://127.0.0.1:8001/v1/models | python3 -m json.tool

# 非流式调用
curl --noproxy '*' -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"your-model-name","messages":[{"role":"user","content":"你好"}],"max_tokens":256}' \
  | python3 -m json.tool

# 流式调用
curl --noproxy '*' -N http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"your-model-name","messages":[{"role":"user","content":"你好"}],"max_tokens":256,"stream":true}'
```

#### Python 调用（服务器上需先清代理）

```python
import os
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    os.environ.pop(k, None)

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=256
)
print("思考:", response.choices[0].message.reasoning_content)
print("回答:", response.choices[0].message.content)
```

#### 本地通过 SSH 隧道访问

```bash
# 终端1：建隧道（保持运行）
ssh -p <SSH_PORT> -L 8001:127.0.0.1:8001 -N <USER>@<SERVER_IP>

# 终端2：正常调用
curl -s http://localhost:8001/v1/models | python3 -m json.tool
```

---

## API 说明

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/models` | GET | 获取可用模型列表 |
| `/v1/chat/completions` | POST | 聊天补全（兼容 OpenAI 格式） |
| `/health` | GET | 健康检查 |

### 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | 必填 | 模型名 |
| `messages` | array | 必填 | 对话消息列表 |
| `temperature` | float | 0.7 | 温度 |
| `top_p` | float | 0.9 | Top-P 采样 |
| `max_tokens` | int | 4096 | 最大生成长度 |
| `stream` | bool | false | 是否流式输出 |

### 响应中的特殊字段

- `reasoning_content`：思维链内容（模型的思考过程）
- `content`：最终回答

---

## 踩坑备忘

| 问题 | 原因 | 解决 |
|------|------|------|
| vLLM 报 `not supported for model type` | vLLM 版本不支持新架构 | 改用 transformers + FastAPI |
| `libcuda.so.1` 找不到 | ROCm 没有 CUDA 库，正常 | 忽略，不影响推理 |
| 502 Bad Gateway | 平台网关不转发端口 | 用 SSH 隧道 `-L` |
| 500 Privoxy Error | 服务器 HTTP 代理拦截 | curl 加 `--noproxy '*'`；Python 清 env |
| `<think>` 标签分离异常 | `skip_special_tokens` 行为不一致 | 已在代码中处理三种格式 |
| 推理慢 ~1 tok/s | transformers 原生推理无优化 | 可尝试 flash-linear-attention |
| 端口被占用 | 其他服务占了 | `netstat -tlnp \| grep PORT` 查看，换端口或 kill |
