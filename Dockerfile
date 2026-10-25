FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY peftquant ./peftquant
COPY pipeline ./pipeline
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir .
ENTRYPOINT ["peftquant"]
CMD ["bench"]
