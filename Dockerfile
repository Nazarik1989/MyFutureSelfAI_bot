FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 future-self \
    && useradd --uid 10001 --gid 10001 --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin future-self \
    && install -d -m 0700 -o 10001 -g 10001 /data /data/knowledge
COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
RUN pip install --no-cache-dir .

USER 10001:10001
HEALTHCHECK --interval=60s --timeout=20s --start-period=30s --retries=3 \
    CMD ["python", "-m", "future_self.doctor"]
CMD ["sh", "-c", "alembic upgrade head && future-self-bot"]
