FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel AS devel

# Install necessary dependencies (for building TriWild)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git cmake g++ libgmp3-dev

# Set up TriWild
RUN mkdir /thirdparty && cd /thirdparty && git clone "https://github.com/wildmeshing/TriWild.git"
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN mkdir /thirdparty/TriWild/build && cd /thirdparty/TriWild/build && cmake .. && make

# Set up Python environment
RUN conda install -y -c conda-forge \
    pytorch-lightning \
    intel-openmp
RUN pip install \
    protobuf \
    libigl==2.5.1 \
    matplotlib \
    gpytoolbox \
    scipy \
    svgpathtools \
    svgwrite \
    opencv-python-headless \
    opencv-contrib-python-headless \
    "numpy<1.26.4" \
    largesteps \
    meshio \
    imageio \
    pybind11 \
    pyquaternion \
    tqdm
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1%2Bcu117.html

# Copying...
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS release

# Install runtime library
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=devel /opt/conda /opt/conda
COPY --from=devel /thirdparty/TriWild/build/TriWild /usr/local/bin/triwild
ENV TRIWILD_PATH=/usr/local/bin/triwild
WORKDIR /workspace
