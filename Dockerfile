# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM ubuntu:24.04

# The distroless uv image holds /uv and /uvx at its root. Pin the patch tag:
# :latest and :0.12 both move.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

ARG DEBIAN_FRONTEND="noninteractive"
# No LIBGL_ALWAYS_INDIRECT: indirect GLX is refused by modern X servers, so it
# fails context creation under xvfb and leaves rgb_array frames empty.
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Qt 6.5+ will not load its xcb platform plugin without libxcb-cursor0; noble
# dropped libgl1-mesa-glx for libgl1; libgl1-mesa-dri is the software GL an
# xvfb display needs to create a context.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
        libgl1 libglu1-mesa libglib2.0-0 libsm6 libxrender1 libxext6 \
        libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 \
        libxcb-shape0 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 \
        libdbus-1-3 libegl1 libopengl0 libgl1-mesa-dri fontconfig mesa-utils xvfb \
    && rm -rf /var/lib/apt/lists/*

# Unpinned, uv installs the newest interpreter allowed by requires-python (3.14
# today), not noble's 3.12. The venv sits outside the workdir so a bind-mounted
# source tree cannot shadow it.
ENV UV_PYTHON=3.12 \
    UV_MANAGED_PYTHON=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /f1tenth_gym
COPY . /f1tenth_gym

# Not --frozen/--locked: both hard-fail when uv.lock is absent, and it is
# gitignored. --no-group gpu drops the CUDA stack that `gpu` in default-groups
# would otherwise pull.
RUN uv sync --no-group gpu

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ENTRYPOINT ["/bin/bash"]
