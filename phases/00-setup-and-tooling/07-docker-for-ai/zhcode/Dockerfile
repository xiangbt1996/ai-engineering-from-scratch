FROM python:3.12-slim

RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com openai

WORKDIR /app

COPY app.py .

CMD ["python", "app.py"]
