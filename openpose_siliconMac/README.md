## OpenPose Setup Guide (macOS for Silicon Mac)

### 1. Install Dependencies

```bash
brew install cmake protobuf boost
brew install --cask xquartz
brew install openblas
brew install opencv
```

---

### 2. Build with CMake

```bash
mkdir build && cd build
```

```bash
cmake .. \
  -DBUILD_PYTHON=ON \
  -DUSE_MKL=OFF \
  -DUSE_CUDNN=OFF \
  -DBLAS=Open \
  -DOpenBLAS_INCLUDE_DIR=/opt/homebrew/opt/openblas/include \
  -DOpenBLAS_LIB=/opt/homebrew/opt/openblas/lib/libopenblas.dylib \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-I/opt/homebrew/opt/openblas/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/opt/homebrew/opt/openblas/lib" \
  -DPYTHON_EXECUTABLE=$(which python3)
```

---

### 3. Compile OpenPose

```bash
make -j$(sysctl -n hw.logicalcpu)
```

---

### 4. Run OpenPose (Single Image Folder)

```bash
./build/examples/openpose/openpose.bin \
  --image_dir ./videos/video_4/extracted_frames \
  --model_pose COCO \
  --net_resolution "-1x256" \
  --scale_number 1 \
  --scale_gap 0.3 \
  --render_pose 1 \
  --display 0 \
  --disable_blending \
  --write_json ./videos/video_4/keypoints_json \
  --write_images ./videos/video_4/pose_frames \
  --disable_multi_thread
```

> 💡 **Output**:  
> - JSON keypoints → `./videos/video_4/keypoints_json/`  
> - Rendered pose images → `./videos/video_4/pose_frames/`


## License
OpenPose is freely available for free non-commercial use, and may be redistributed under these conditions. Please, see the [license](./LICENSE) for further details. Interested in a commercial license? Check this [FlintBox link](https://cmu.flintbox.com/#technologies/b820c21d-8443-4aa2-a49f-8919d93a8740). For commercial queries, use the `Contact` section from the [FlintBox link](https://cmu.flintbox.com/#technologies/b820c21d-8443-4aa2-a49f-8919d93a8740) and also send a copy of that message to [Yaser Sheikh](mailto:yaser@cs.cmu.edu).
