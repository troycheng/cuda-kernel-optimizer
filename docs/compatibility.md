# Compatibility

The CPU/static suite targets Python 3.10 and 3.12. GPU execution uses Linux facilities for process groups, signals, resource locking, and tool supervision. macOS can run static checks; native Windows GPU execution is not supported.

| Path | Requirement | Evidence boundary |
|---|---|---|
| CUDA C++ | Compatible driver, Toolkit, compiler, and exact target architecture | Generated binary must remain bound to source and build identity |
| CUTLASS / CuTe | A checkout and APIs that support the exact target | Version labels do not replace compile probes |
| Triton | Compatible Python, framework, Triton, and GPU target | IR, launch, dispatch, and generated binary identity may all matter |
| Nsight Compute | Supported NCU export or collection tool | Counter permission is optional and reported explicitly |
| Nsight Systems | Supported exported SQLite dialect or collection tool | Private `.nsys-rep` bytes are not reverse-engineered |
| PyTorch Profiler | Supported Chrome trace dialect | Unknown versions, interpretation-critical fields, or units fail closed; non-critical extensions remain unmodeled |

Architecture-specific claims require an exact compute capability match. A numerically adjacent SM does not inherit another target's features. Use a local compile probe and current official documentation for capability questions.

The repository includes opt-in physical RTX 5090 tests. They are separate from the default CPU/static suite. See the [RTX 5090 test guide](https://github.com/troycheng/cuda-kernel-optimizer/blob/v1.4.1/tests/gpu/sm120/README.md) and the installed [compatibility reference](https://github.com/troycheng/cuda-kernel-optimizer/blob/v1.4.1/skills/cuda-kernel-optimizer/references/compatibility.md).
