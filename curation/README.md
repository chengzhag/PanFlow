
# PanFlow Data Curation Pipeline

This directory contains the code for curating high-quality panoramic video clips with accurate camera pose from the [360-1M](https://github.com/MattWallingford/360-1M). This pipeline is mostly designed to be run on a cluster due to the large data size and computational cost, but we also provide options to run on local machine.

## 🛠️ Installation

```bash
conda create -n panflow_curation python=3.12
conda activate panflow_curation
pip install -r requirements.txt
```

Follow the guide "Use your own video..." in [PanSplat](https://github.com/chengzhag/PanSplat#our-video-data) to install [stella_vslam](https://github.com/stella-cv/stella_vslam?tab=readme-ov-file).

<details>
<summary>Alternatively, if you are on a cluster without sudo, try install from source code.</summary>

As an example, we install to `${HOME}/usr/local`:

```bash

# Prerequisites for cluster: example 1
module load cuda/11.7
conda activate panflow_curation
export LD_LIBRARY_PATH=${HOME}/usr/local/lib64:${HOME}/usr/local/lib:$LD_LIBRARY_PATH
export PATH=${HOME}/usr/local/bin:$PATH
module load cmake/3.29.3
module load gcc/10.2.0

# Prerequisites for cluster: example 2
module load cuda/11.7.0
conda activate panflow_curation
export LD_LIBRARY_PATH=${HOME}/usr/local/lib64:${HOME}/usr/local/lib:$LD_LIBRARY_PATH
export PATH=${HOME}/usr/local/bin:$PATH
module load cmake/3.24.2
module load gcc/10.3.0

# Create target directories
mkdir -p ${HOME}/usr/local

# Download and install ffmpeg from source.
cd /tmp
rm -rf nasm-2.16.03
wget https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/nasm-2.16.03.tar.xz
tar -xvf nasm-2.16.03.tar.xz
rm nasm-2.16.03.tar.xz
cd nasm-2.16.03
./configure --prefix=${HOME}/usr/local
make -j4 && make install

cd /tmp
rm -rf x264
git clone --depth 1 https://code.videolan.org/videolan/x264.git
cd x264
./configure --enable-shared --prefix=${HOME}/usr/local
make -j4 && make install

cd /tmp
rm -rf libvpx
git clone https://chromium.googlesource.com/webm/libvpx
cd libvpx
git checkout v1.15.0
./configure --enable-vp9 --enable-vp8 --enable-shared --prefix=${HOME}/usr/local
make -j4 && make install

cd /tmp
rm -rf ffmpeg
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg
git checkout n4.4.2
./configure --prefix=${HOME}/usr/local \
            --extra-cflags="-I${HOME}/usr/local/include" \
            --extra-ldflags="-L${HOME}/usr/local/lib" \
            --enable-gpl \
            --enable-nonfree \
            --enable-libx264 \
            --enable-libvpx \
            --enable-shared
make -j4 && make install

# Download and install tbb from source.
cd /tmp
rm -rf oneTBB
git clone https://github.com/uxlfoundation/oneTBB
cd oneTBB
mkdir build
cd build
cmake \
    -DCMAKE_INSTALL_PREFIX=${HOME}/usr/local \
    -DCMAKE_PREFIX_PATH=${HOME}/usr/local \
    -DTBB_TEST=OFF \
    ..
make -j4 && make install

# Download and install yaml-cpp from source.
cd /tmp
rm -rf yaml-cpp
git clone https://github.com/jbeder/yaml-cpp
cd yaml-cpp
mkdir build
cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${HOME}/usr/local \
    -DYAML_BUILD_SHARED_LIBS=ON \
    -DCMAKE_PREFIX_PATH=${HOME}/usr/local \
    ..
make -j4 && make install

# Download and install Eigen from source.
cd /tmp
rm -rf eigen-3.3.7
wget -q https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2
tar xf eigen-3.3.7.tar.bz2 && rm -rf eigen-3.3.7.tar.bz2
cd eigen-3.3.7
mkdir build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${HOME}/usr/local \
    ..
make -j4 && make install

# Download, build and install OpenCV from source.
cd /tmp
rm -rf opencv-4.5.5
rm -rf extra
rm -rf opencv_contrib-4.5.5
# Download OpenCV
wget -q https://github.com/opencv/opencv/archive/4.5.5.zip
unzip -q 4.5.5.zip && rm -rf 4.5.5.zip
# Download aruco module (optional)
wget -q https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.5.zip -O opencv_contrib-4.5.5.zip
unzip -q opencv_contrib-4.5.5.zip && rm -rf opencv_contrib-4.5.5.zip
mkdir extra && mv opencv_contrib-4.5.5/modules/aruco extra
rm -rf opencv_contrib-4.5.5
# Build and install OpenCV
cd opencv-4.5.5
mkdir build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${HOME}/usr/local \
    -DBUILD_DOCS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_JASPER=OFF \
    -DBUILD_OPENEXR=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PROTOBUF=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_dnn=OFF \
    -DBUILD_opencv_ml=OFF \
    -DBUILD_opencv_python_bindings_generator=OFF \
    -DENABLE_CXX11=ON \
    -DENABLE_FAST_MATH=ON \
    -DWITH_EIGEN=ON \
    -DWITH_FFMPEG=ON \
    -DWITH_TBB=ON \
    -DWITH_OPENMP=ON \
    -DOPENCV_EXTRA_MODULES_PATH=/tmp/extra \
    -DCMAKE_PREFIX_PATH=${HOME}/usr/local \
    ..
make -j4 && make install

# Download and install SuiteSparse from source.
cd /tmp
rm -rf mpfr-4.2.1
wget https://www.mpfr.org/mpfr-current/mpfr-4.2.1.tar.xz
tar -xf mpfr-4.2.1.tar.xz
cd mpfr-4.2.1
./configure --prefix=$HOME/usr/local
make -j4 && make install

cd /tmp
rm -rf SuiteSparse
git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
cd SuiteSparse
git checkout v7.10.1
mkdir build
cd build
cmake \
    -DCMAKE_INSTALL_PREFIX=${HOME}/usr/local \
    -DCMAKE_PREFIX_PATH=${HOME}/usr/local \
    ..
make -j4 && make install

# Download, build and install the custom FBoW from source.
cd /tmp
rm -rf FBoW
git clone https://github.com/stella-cv/FBoW.git
cd FBoW
mkdir build
cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${HOME}/usr/local \
    -DCMAKE_PREFIX_PATH=${HOME}/usr/local \
    ..
make -j4 && make install

# Download, build and install g2o.
cd /tmp
rm -rf g2o
git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o
git checkout 20230223_git
mkdir build
cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${HOME}/usr/local \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_UNITTESTS=OFF \
    -DG2O_USE_CHOLMOD=OFF \
    -DG2O_USE_CSPARSE=ON \
    -DG2O_USE_OPENGL=OFF \
    -DG2O_USE_OPENMP=OFF \
    -DG2O_BUILD_APPS=OFF \
    -DG2O_BUILD_EXAMPLES=OFF \
    -DG2O_BUILD_LINKED_APPS=OFF \
    -DCMAKE_PREFIX_PATH=${HOME}/usr/local \
    ..
make -j4 && make install

# Download, build and install backward-cpp.
cd /tmp
rm -rf backward-cpp
git clone https://github.com/bombela/backward-cpp.git
cd backward-cpp
git checkout 5ffb2c879ebdbea3bdb8477c671e32b1c984beaa
mkdir build
cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${HOME}/usr/local \
    -DCMAKE_PREFIX_PATH=${HOME}/usr/local \
    ..
make -j4 && make install

# Install stella_vslam core library
mkdir ${HOME}/lib
cd ${HOME}/lib
git clone --recursive https://github.com/stella-cv/stella_vslam.git
cd stella_vslam
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_INSTALL_PREFIX=${HOME}/usr/local \
      -DCMAKE_PREFIX_PATH=${HOME}/usr/local \
      ..
make -j4 && make install

# Install stella_vslam executables
cd ${HOME}/lib
git clone --recursive https://github.com/stella-cv/stella_vslam_examples.git
cd stella_vslam_examples
rm -rf build
mkdir build
cd build
cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DUSE_STACK_TRACE_LOGGER=ON \
    -DCMAKE_PREFIX_PATH=${HOME}/usr/local \
    ..
make -j4
curl -sL "https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow" -o orb_vocab.fbow

```

</details>
<br>

## 📂 Data Curation

### Download Data

Follow [360-1M](https://github.com/MattWallingford/360-1M) to download the [Filtered Subset](https://huggingface.co/datasets/mwallingford/360-1M/blob/main/Filtered_24k.parquet) to `Filtered_24k.parquet`. Then run the following command to download the videos:

```bash
python download_local.py --in_path Filtered_24k.parquet --out_dir ../data/PanFlow/videos/
```

This script is adapted from [360-1M](https://github.com/MattWallingford/360-1M). Due to the consistent changes in yt-dlp's downloading mechanism to comply with YouTube's anti-scraping mechanism, the script may require some adjustments from time to time. Total size is around 1 TB.

### Clip Detection

Adapt `hpc_curation.py` as needed for your cluster environment. Then run the following command on the cluster to detect clips in the videos.
```bash
python hpc_curation.py --steps video_check format_check detect_scenes slam_clips motion_score
```
We recommend running this step on a cluster with multiple CPU cores. Alternatively, you can run it with `batch_curation.py` on local machine, but it will be much slower. This will output the metadata of the detected clips to `data/PanFlow/meta/`.

### Watermark Score

Run the following command on local machine to compute the watermark score for each clip.
```bash
python batch_curation.py --steps watermark_score
```
We recommend running this step on a local machine with a GPU. This step will update the watermark score for each clip to `data/PanFlow/meta/`.

### Pose Estimation

Run the following command on the cluster to estimate the camera pose for each clip.
```bash
python hpc_curation.py --steps slam_pose
```
We recommend running this step on a cluster with multiple CPU cores. This step will output the camera pose for each clip to `data/PanFlow/slam_pose/` and update the metadata in `data/PanFlow/meta/`.

### Filter Clips

Run the following command on local machine to filter the clips.
```bash
python filter_clips.py
```
This will output the metadata of the filtered data to `data/PanFlow/filter_clips/`.

### Caption

We use the official tool of [CogVideo](https://github.com/zai-org/CogVideo/tree/main/tools/caption) to generate captions for each clip. Please follow their instructions to set up the environment and adapt their script to add generated captions to the metadata files in `data/PanFlow/filter_clips/`.
