from __future__ import annotations

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    values = torch.arange(1 << 20, device="cuda", dtype=torch.float32)
    output = values * 2.0 + 1.0
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise SystemExit("NCU smoke workload produced invalid output")
    print(float(output[0].item()))


if __name__ == "__main__":
    main()
