FROM postgres:17.11-trixie

RUN apt-get update \
  && apt-get install -y --no-install-recommends postgresql-17-postgis-3 postgresql-17-h3 \
  && rm -rf /var/lib/apt/lists/*

COPY init.sql /docker-entrypoint-initdb.d/
