# Multi-arch build of the DrugCentral source database engine: PostgreSQL 16 + the RDKit cartridge.
# Mirrors the published unmtransinfo/drugcen-2027-db recipe (RDKit Release_2026_03_6 -> extension 4.8.0)
# so the full source archive restores unchanged on arm64 hosts, where that amd64-only image cannot run.
# Build: docker build -f environments/ec2/source-db.Dockerfile -t drugcentral-source-db:rdkit-2026.03 .
FROM postgres:16 AS build
ARG RDKIT_TAG=Release_2026_03_6
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libboost-all-dev libeigen3-dev libpq-dev \
        postgresql-server-dev-16 libfreetype6-dev python3-dev python3-numpy \
        libjpeg-dev libpng-dev libtiff-dev zlib1g-dev libxml2-dev libexpat1-dev libffi-dev libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${RDKIT_TAG} https://github.com/rdkit/rdkit.git /tmp/rdkit
WORKDIR /tmp/rdkit/build
RUN PG_ROOT="$(pg_config --bindir | sed 's#/bin$##')" \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DPython3_EXECUTABLE="$(which python3)" \
        -DRDK_BUILD_PYTHON_WRAPPERS=OFF -DRDK_BUILD_RDRAWPYTHON=OFF \
        -DRDK_BUILD_INCHI_SUPPORT=ON -DRDK_BUILD_AVALON_SUPPORT=ON -DRDK_BUILD_PGSQL=ON \
        -DPostgreSQL_ROOT="${PG_ROOT}" \
        -DPostgreSQL_INCLUDE_DIR="$(pg_config --includedir-server)" \
        -DPostgreSQL_LIBRARY="$(pg_config --libdir)/libpq.so" \
    && make -j"$(nproc)" && make install

FROM postgres:16
RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-iostreams1.83.0 libboost-serialization1.83.0 libboost-system1.83.0 libboost-regex1.83.0 \
        libfreetype6 libxml2 libexpat1 libsqlite3-0 zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/lib/ /usr/local/lib/
COPY --from=build /usr/lib/postgresql/16/lib/rdkit.so /usr/lib/postgresql/16/lib/rdkit.so
COPY --from=build /usr/share/postgresql/16/extension/rdkit* /usr/share/postgresql/16/extension/
RUN ldconfig
