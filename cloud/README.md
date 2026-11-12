# GPU image build gate

Run `build_and_push.sh` from the repository root in a Linux CI/CodeBuild worker with Docker, AWS CLI, ECR push permission, and an authenticated `sa-east-1` account. Set `AWS_ACCOUNT_ID` and `ECR_REPOSITORY`. The command prints the digest-pinned ECR URI required by `SageMakerConfig`.

The pinned base is the [official PyTorch 2.7.1 CUDA 11.8/cuDNN 9 runtime](https://github.com/orgs/pytorch/packages/container/pytorch/431042143?tag=2.7.1-cuda11.8-cudnn9-runtime). The hash lock was compiled for Linux x86-64 and Python 3.11 with `uv pip compile pyproject.toml --constraint cloud/torch.constraints --python-version 3.11 --python-platform x86_64-manylinux_2_28 --generate-hashes --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match -o cloud/requirements-linux-py311.lock`.

Record the digest, `pip check`, and `python -m tabtune_baseline.image_self_test` output in your run notes. Build checks alone do not verify the host driver, GPU device, model checkpoint route, or end-to-end scores. The real `sa-east-1` smoke remains required before larger runs.
