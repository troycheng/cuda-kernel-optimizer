# Compatibility

The CPU/static suite is tested with Python 3.10 and 3.12 on Linux CI. Core
Controller scripts use POSIX facilities such as file locks, resource limits,
signals, and process groups. Linux is the supported GPU execution environment;
macOS can run CPU/static checks, while native Windows is not supported. Windows
users should use a Linux or WSL environment.

Kernel optimization also needs a working CUDA GPU and driver and the toolchain
required by the target implementation.

| Path | Requirement | Boundary |
|---|---|---|
| CUDA C++ | `nvcc`, compatible driver/toolkit, target architecture | Generated binary and compiler evidence must be bound to the tested source |
| CUTLASS / CuTe | Compatible CUTLASS checkout and architecture support | Public APIs and target-specific routing take precedence over version labels alone |
| Triton | Compatible Python, PyTorch, Triton, and GPU target | Autotune, IR, launch configuration, and generated binary identity may all matter |
| Nsight Compute | Compatible `ncu` for profiling or report import | Counter access is optional; unavailable access must be reported explicitly |

## RTX 5090 and SM120

The repository includes an opt-in physical RTX 5090 lane. It is not run by the
default CPU/static test command. Historical target-side profiling returned
`ERR_NVGPUCTRPERM`; the workflow recorded that degradation without changing
permissions or driver policy.

## NCU report import

Read-only report analysis needs a compatible Nsight Compute executable and an
existing report file. It does not launch the profiled program and cannot prove
that the current host can collect counters.

Exact observed versions, architecture capability rules, Triton and CUTLASS
routing, and primary upstream sources are maintained in the
[canonical compatibility reference](https://github.com/troycheng/cuda-kernel-optimizer/blob/main/skills/cuda-kernel-optimizer/references/compatibility.md).

The physical GPU fixture and opt-in commands are documented in the
[RTX 5090 test guide](https://github.com/troycheng/cuda-kernel-optimizer/blob/main/tests/gpu/sm120/README.md).

## Schema identities

New unversioned schemas use the standalone repository namespace. Existing
versioned pre-V1 schema IDs under the archived `cuda-optimized-skill/schema/v*` and
`cuda-optimized-skill/schemas/v*` namespaces remain unchanged because they are
stable protocol identifiers, not installation URLs. Future incompatible schemas
must use a new versioned namespace in this repository instead of rewriting those
legacy IDs.
