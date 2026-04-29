# Builder stage: compile DHSVM
FROM ubuntu:22.04 AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        libnetcdf-dev \
        libx11-dev \
        tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /matilija/src
RUN curl --fail -L -o dhsvm3.2.tar.gz https://codeload.github.com/pnnl/DHSVM-PNNL/tar.gz/refs/tags/v3.2 \
    && tar -xzf dhsvm3.2.tar.gz \
    && mv DHSVM-PNNL-3.2 dhsvm \
    && rm dhsvm3.2.tar.gz

WORKDIR /matilija/src/dhsvm/build
RUN cmake -D CMAKE_BUILD_TYPE:STRING=Release \
    -D DHSVM_USE_X11:BOOL=ON \
    -D DHSVM_USE_NETCDF:BOOL=ON \
    .. \
    && cmake --build .

# Runtime stage: copy the built binary and runtime libraries only
FROM ubuntu:22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libnetcdf19 \
        libx11-6 \
        libhdf5-103-1 \
        libcurl4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /matilija
COPY --from=builder /matilija/src/dhsvm/build/DHSVM /matilija/src/dhsvm/build/DHSVM
