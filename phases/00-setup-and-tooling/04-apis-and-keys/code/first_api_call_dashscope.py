"""
第四课练习：使用阿里云百炼（DashScope）API 调用大模型

百炼兼容 OpenAI SDK 格式，所以可以用 openai 库直接调用。
也可以用阿里官方的 dashscope SDK。

使用 .env 文件管理密钥（无需手动 export）：
  1. 在项目根目录创建 .env 文件，写入: DASHSCOPE_API_KEY=sk-你的密钥
  2. 确保 .env 已加入 .gitignore
  3. 运行本脚本即可自动加载

获取密钥：https://bailian.console.aliyun.com/ → API-KEY 管理
"""

import os
import json
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

# 自动查找项目根目录的 .env 文件并加载
env_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(env_path)



def call_with_openai_sdk():
    """方式 1：用 OpenAI SDK（百炼兼容 OpenAI 接口）"""
    try:
        from openai import OpenAI
    except ImportError:
        print("请先安装: pip install openai")
        return

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("请先设置环境变量: export DASHSCOPE_API_KEY='sk-你的密钥'")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "Hello，你是谁"}],
        max_tokens=256,
    )

    print(f"模型回复: {response.choices[0].message.content}")
    print(f"Token 用量: {response.usage.prompt_tokens} 输入, {response.usage.completion_tokens} 输出")


def call_with_dashscope_sdk():
    """方式 2：用阿里官方 dashscope SDK"""
    try:
        import dashscope
        from dashscope import Generation
    except ImportError:
        print("请先安装: pip install dashscope")
        return

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("请先设置环境变量: export DASHSCOPE_API_KEY='sk-你的密钥'")
        return

    dashscope.api_key = api_key

    response = Generation.call(
        model="qwen-plus",
        messages=[{"role": "user", "content": "Hello，你是谁"}],
        max_tokens=256,
        result_format="message",
    )

    if response.status_code == 200:
        content = response.output.choices[0].message.content
        print(f"模型回复: {content}")
        print(f"Token 用量: {response.usage.input_tokens} 输入, {response.usage.output_tokens} 输出")
    else:
        print(f"请求失败: {response.code} - {response.message}")


def call_raw_http():
    """方式 3：原始 HTTP 请求（不依赖任何 SDK）"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("请先设置环境变量: export DASHSCOPE_API_KEY='sk-你的密钥'")
        return

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = json.dumps({
        "model": "qwen-plus",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Hello，你是谁"}],
    }).encode()

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        content = result["choices"][0]["message"]["content"]
        usage = result["usage"]
        print(f"模型回复: {content}")
        print(f"Token 用量: {usage['prompt_tokens']} 输入, {usage['completion_tokens']} 输出")


if __name__ == "__main__":
    print("=== 阿里云百炼 API 调用演示 ===\n")
#
#     print("--- 方式 1: OpenAI SDK（兼容模式）---")
#     call_with_openai_sdk()
#
#     print("\n--- 方式 2: DashScope 官方 SDK ---")
#     call_with_dashscope_sdk()

    print("\n--- 方式 3: 原始 HTTP 请求 ---")
    call_raw_http()
