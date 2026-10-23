FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY requirements ./requirements
RUN if [ -f requirements/analytics.lock ]; then \
      pip install --no-cache-dir --require-hashes -r requirements/analytics.lock \
      && pip install --no-cache-dir --no-deps .; \
    else pip install --no-cache-dir '.[analytics]'; fi \
    && pip freeze --all > /app/installed-packages.txt \
    && useradd --create-home --uid 10001 shelfsense \
    && mkdir /app/outputs \
    && chown shelfsense:shelfsense /app/outputs
COPY dashboard ./dashboard
COPY configs ./configs
COPY examples ./examples

USER shelfsense
EXPOSE 8000 8501
CMD ["uvicorn", "shelfsense.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

FROM app AS test
USER root
RUN pip install --no-cache-dir 'httpx>=0.27,<1'
COPY tests ./tests
COPY scripts ./scripts
USER shelfsense
RUN python -m unittest discover -s tests -v \
    && python -m shelfsense smoke --output-dir outputs/build-smoke

FROM test AS postgres-test
USER root
RUN apt-get update && apt-get install -y --no-install-recommends postgresql-15 \
    && rm -rf /var/lib/apt/lists/*
USER shelfsense
RUN /usr/lib/postgresql/15/bin/initdb -D /tmp/test-postgres --auth=trust --no-locale --encoding=UTF8 \
    && (/usr/lib/postgresql/15/bin/pg_ctl -D /tmp/test-postgres -l /tmp/test-postgres.log -o '-h 127.0.0.1 -k /tmp -p 55432' start \
        || { cat /tmp/test-postgres.log; exit 1; }) \
    && SHELFSENSE_TEST_POSTGRES_URL=postgresql://shelfsense@127.0.0.1:55432/postgres \
       PYTHONPATH=tests python -m unittest test_analytics.PostgresTests -v \
    && /usr/lib/postgresql/15/bin/pg_ctl -D /tmp/test-postgres stop

FROM app AS lockbuilder
USER root
RUN pip install --no-cache-dir uv==0.9.4 \
    && uv pip compile pyproject.toml --extra analytics --extra test --generate-hashes \
       --output-file /app/requirements/analytics.lock

FROM scratch AS dependency-lock
COPY --from=lockbuilder /app/requirements/analytics.lock /analytics.lock

FROM app AS runtime
