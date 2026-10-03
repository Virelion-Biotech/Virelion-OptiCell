# GPU runner setup

OptiCell includes a manual GitHub Actions workflow named **Real GPU Cellpose E2E**.
It intentionally targets a self-hosted runner labeled:

- `self-hosted`
- `linux`
- `x64`
- `opticell-gpu`

The runner must expose a working NVIDIA driver/runtime such that both
`nvidia-smi` and `torch.cuda.is_available()` succeed.

Once attached, trigger the workflow from GitHub Actions. The job runs both the
real Cellpose engine smoke and the real Streamlit UI path with Cellpose selected.

This closes the software/CI wiring gap. It does not provide GPU hardware by
itself; the runner remains deployment infrastructure.
