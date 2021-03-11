FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq \
    libboost-dev \
    libboost-program-options-dev \
    libzmq3-dev \
    doxygen \
    libspdlog-dev \
    libyaml-cpp-dev \
    libgtest-dev \
    libfmt-dev \
    python3-pip \
    curl \
    cmake \
    pkg-config \
    libfmt-dev

RUN pip3 install \
    meson \
    ninja \
    mako \
    six

RUN mkdir -p volk && \
    curl -Lo volk.tar.gz https://github.com/gnuradio/volk/archive/v2.2.1.tar.gz && \
    tar xzf volk.tar.gz -C volk --strip-components=1 && \
    cmake -DCMAKE_BUILD_TYPE=Release -S ./volk/ -B build && \
    cd build && \
    cmake --build . && \
    cmake --build . --target install && \
    ldconfig && \
    cd ../ && \
    rm -rf build
