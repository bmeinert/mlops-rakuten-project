FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends git gosu \
  && rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory /app && \
    git config --global --add safe.directory '*'

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir "dvc[s3]"

COPY dockerfiles/preprocessing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dockerfiles/preprocessing/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

COPY . .

# Set PYTHONPATH for possibility to use other scripts
ENV PYTHONPATH="${PYTHONPATH}:/app"

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "plugins/cd4ml/data_processing/run_preprocessing.py"]