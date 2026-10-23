# Software and hardware environment

Recorded research environment:

| Component | Version or device |
|---|---|
| Python | 3.13.14 (64-bit) |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.16.1 |
| CUDA runtime reported by PyTorch | 12.8 |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU |
| pandas | 2.2.3 |
| NumPy | 2.1.3 |
| scikit-learn | 1.6.1 |
| NLTK | 3.9.1 |
| Stanza | 1.14.0 |
| SciPy | 1.16.3 |

`requirements.txt` and `constraints-cuda.txt` provide the project's dependency specifications. The complete recorded package list is in `provenance/datasets/abstract/pip_freeze.txt`. GPU training additionally requires compatible NVIDIA drivers and the pretrained model and tokenizer assets. These versions describe the experiment environment; installation on a different machine may require platform-specific adjustments.
