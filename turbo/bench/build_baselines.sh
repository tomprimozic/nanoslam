#!/bin/bash
# build the patched baselines (bench/patches applied on a `bench` branch of each checkout in $LIB)
# usage: build_baselines.sh [LIB]   (default ~/my/lib); logs in $LIB/build_logs
LIB=${1:-$HOME/my/lib}
J=$(nproc)
LOG=$LIB/build_logs; mkdir -p $LOG
step() { echo "== $1"; shift; ( "$@" ) > $LOG/$STEP.log 2>&1 && echo "   ok" || { echo "   FAILED (see $LOG/$STEP.log)"; tail -5 $LOG/$STEP.log; }; }

STEP=pangolin; step "Pangolin v0.6" bash -c "
  cd $LIB && [ -d Pangolin ] || git clone -q --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git
  cd Pangolin && mkdir -p build && cd build &&
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$LIB/Pangolin/install -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF \
    -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_PANGOLIN_FFMPEG=OFF -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_FLAGS='-include cstdint -include limits' &&
  make -j$J install"
export Pangolin_DIR=$LIB/Pangolin/install/lib/cmake/Pangolin

STEP=orbslam2; step "ORB-SLAM2" bash -c "cd $LIB/ORB_SLAM2 && sed -i 's/make -j[0-9]*$/make -j$J/' build.sh && bash build.sh && cd build && make -j$J mono_bench"
STEP=orbslam3; step "ORB-SLAM3" bash -c "cd $LIB/ORB_SLAM3 && sed -i 's/make -j[0-9]*$/make -j$J/' build.sh && bash build.sh && cd build && make -j$J mono_bench"
STEP=dso; step "DSO" bash -c "cd $LIB/dso && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$J dso_bench"

STEP=opencv; step "OpenCV 4.5.5 + contrib (minimal, for ov2slam's xfeatures2d)" bash -c "
  cd $LIB && [ -d opencv-4.5.5 ] || git clone -q --depth 1 --branch 4.5.5 https://github.com/opencv/opencv.git opencv-4.5.5
  [ -d opencv_contrib-4.5.5 ] || git clone -q --depth 1 --branch 4.5.5 https://github.com/opencv/opencv_contrib.git opencv_contrib-4.5.5
  mkdir -p opencv-4.5.5/build && cd opencv-4.5.5/build &&
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$LIB/opencv-4.5.5/install -DOPENCV_EXTRA_MODULES_PATH=$LIB/opencv_contrib-4.5.5/modules \
    -DBUILD_LIST=core,imgproc,imgcodecs,features2d,calib3d,flann,video,xfeatures2d,highgui,videoio -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_opencv_python3=OFF -DWITH_CUDA=OFF -DWITH_IPP=OFF -DOPENCV_ENABLE_NONFREE=OFF -DWITH_FFMPEG=OFF &&
  make -j$J install"
STEP=opengv; step "OpenGV" bash -c "
  cd $LIB && [ -d opengv ] || git clone -q https://github.com/laurentkneip/opengv.git
  cd opengv && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$LIB/opengv/install && make -j$J install"
STEP=ov2slam_thirdparty; step "ov2slam third party (iBoW-LCD, Sophus, Ceres 2.0)" bash -c "cd $LIB/ov2slam && sed -i 's/make -j[0-9]*/make -j$J/' build_thirdparty.sh && bash build_thirdparty.sh"
STEP=ov2slam; step "ov2slam standalone" bash -c "cd $LIB/ov2slam/standalone && mkdir -p build && cd build &&
  cmake .. -DOpenCV_DIR=$LIB/opencv-4.5.5/install/lib/cmake/opencv4 && make -j$J ov2slam_bench"
echo BUILD_DONE
