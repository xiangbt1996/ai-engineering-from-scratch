import os
import json
import urllib.request
from openai import OpenAI

api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    print("请设置 DASHSCOPE_API_KEY 环境变量")
    exit(1)

# === 第一步：访问 Qdrant 向量数据库（第二个容器）===
print("--- 测试访问 Qdrant 容器 ---")
try:
    req = urllib.request.Request("http://qdrant:6333/collections")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
        print(f"Qdrant 连接成功！当前集合数: {len(data['result']['collections'])}")
except Exception as e:
    print(f"Qdrant 连接失败: {e}")

# === 第二步：调用百炼 API ===
print("\n--- 调用百炼 API ---")
client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
response = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Hello，你是谁"}],
    max_tokens=256,
)
print(f"模型回复: {response.choices[0].message.content}")
print(f"Token 用量: {response.usage.prompt_tokens} 输入, {response.usage.completion_tokens} 输出")
