"""
API 调用示例（非流式 + 流式）
使用前先 pip install openai

如果在部署服务器上运行，需要清除代理环境变量（脚本已自动处理）。
如果在本地运行，需要先建立 SSH 隧道：
  ssh -p <SSH_PORT> -L 8001:127.0.0.1:8001 -N <USER>@<SERVER_IP>
"""

import os

# 清除代理（服务器上有 Privoxy 会拦截请求）
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    os.environ.pop(k, None)

from openai import OpenAI

# ============ 修改这里 ============
BASE_URL = "http://127.0.0.1:8001/v1"  # 本地隧道或服务器本机
MODEL_NAME = "your-model-name"
# ==================================

client = OpenAI(base_url=BASE_URL, api_key="not-needed")


def test_non_stream():
    """非流式调用"""
    print("=" * 50)
    print("非流式调用测试")
    print("=" * 50)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "用一句话介绍你自己"}],
        temperature=0.7,
        max_tokens=512,
    )

    msg = response.choices[0].message
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        print(f"\n💭 思考过程:\n{msg.reasoning_content}")
    print(f"\n💬 回答:\n{msg.content}")
    print(f"\nTokens: {response.usage.total_tokens}")


def test_stream():
    """流式调用"""
    print("\n" + "=" * 50)
    print("流式调用测试")
    print("=" * 50)

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "1+1等于几？请简短回答"}],
        temperature=0.7,
        max_tokens=512,
        stream=True,
    )

    print("\n💭 思考过程:")
    reasoning_done = False
    for chunk in stream:
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            print(delta.reasoning_content, end="", flush=True)
        if delta.content:
            if not reasoning_done:
                print(f"\n\n💬 回答:")
                reasoning_done = True
            print(delta.content, end="", flush=True)
    print()


def test_multi_turn():
    """多轮对话"""
    print("\n" + "=" * 50)
    print("多轮对话测试")
    print("=" * 50)

    messages = [
        {"role": "system", "content": "你是一个有帮助的助手，请简短回答。"},
        {"role": "user", "content": "什么是Python？"},
    ]

    r1 = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, max_tokens=256
    )
    print(f"\n用户: 什么是Python？")
    print(f"助手: {r1.choices[0].message.content}")

    messages.append({"role": "assistant", "content": r1.choices[0].message.content})
    messages.append({"role": "user", "content": "它和Java有什么区别？"})

    r2 = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, max_tokens=256
    )
    print(f"\n用户: 它和Java有什么区别？")
    print(f"助手: {r2.choices[0].message.content}")


if __name__ == "__main__":
    test_non_stream()
    test_stream()
    test_multi_turn()
