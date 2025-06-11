## Mutual Self-Attention + ControlNet

### Environment Setting

Create anaconda environment
```
conda create -n {env_name} python=3.9
```

Install required dependencies
```
pip install -r requirement.txt
```

Install pytorch
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 1. Sample Inference
The notebook `masactrl_controlnet.ipynb` demonstrates our method for pose-guided image editing.

### 2. Experiments
The notebook `sampling.ipynb` and `sampling_p2p.ipynb` provide a way to batch-generate sample datasets.

The notebook `EvalMetrics.ipynb` provides a way to calculate metrics from generated samples.