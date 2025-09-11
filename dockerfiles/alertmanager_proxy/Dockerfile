FROM python:3.9-slim-buster

WORKDIR /app

COPY dockerfiles/alertmanager_proxy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "plugins/cd4ml/alertmanager_helper/webhook_proxy.py"]