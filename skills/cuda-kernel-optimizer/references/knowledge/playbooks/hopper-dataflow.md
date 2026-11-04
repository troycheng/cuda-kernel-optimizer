# Hopper data movement and compute overlap

Use this note when a matched card points to a data-supply or producer/consumer question. It describes alternatives, not a required profiling sequence. Read existing code and reports first; collect only the missing evidence that could change the next decision.

## Locate the lost time

Low Tensor Core utilization alone does not identify a pipeline problem. A small grid, tail tiles, non-matmul work, memory traffic and synchronization can produce similar symptoms. Check the dispatched kernel, problem shape and tile count before interpreting counters. Separate time inside that kernel from host launch gaps and unrelated work.

For a sustained mainloop, compare the work feeding each tile with the work consuming it. Does the consumer wait for data, does the producer wait for a free stage, or are both waiting on another dependency? Use existing generated code, register/shared-memory reports and available profile evidence. An aggregate stall percentage alone cannot distinguish these cases.

## Choose a change that addresses the cause

- Data arrives late: inspect address mapping and reuse before replacing loads. Ampere global-to-shared `cp.async` and Hopper tensor-map TMA are different paths. TMA reduces per-thread copy/address work, but requires valid tensor layout and synchronization; descriptor setup and tails may dominate a short kernel. It does not remove necessary memory traffic.
- Work can overlap: consider producer/consumer specialization or another stage only if there is independent work to overlap. More stages consume shared memory; larger tiles and consumers can increase registers. Include epilogue storage when estimating residency. A candidate that hides copy latency but reduces useful parallelism may lose overall.
- Registers are unevenly needed: `setmaxnreg` reallocates a CTA's register pool; it is not extra capacity. An increase can wait for registers to be released. Check the instruction's target/PTX requirements and warpgroup synchronization in the actual implementation, not just the GPU compute capability. Keep a working upstream implementation when possible.
- Several CTAs reuse a tile: cluster multicast may reduce duplicate transfers, but cluster residency, synchronization and existing cache reuse affect the benefit. Check these before proposing multicast. An H800/H100 cluster is not Blackwell Cluster Launch Control; they solve different problems.

The CUTLASS Hopper collective-builder example exposes tile, cluster, stage and schedule choices, including epilogue shared-memory carveout. Start from an implementation available in the deployed version. Change the parameter tied to the unresolved cause; do not sweep every combination or transplant another GPU's winning configuration.

## Carry principles, not measured gains

These trade-offs can inform another architecture, but Hopper WGMMA code and tuning parameters are not portable merely because a GPU supports Tensor Cores. Recheck instruction targets, generated code and resource limits for Ampere or either Blackwell family. For H800/H100/H200, use the installed device's memory capacity and actual interconnect rather than inheriting another SKU's specifications. Large-memory capacity may change feasible batching or parallelism; it does not establish a kernel speedup.

Offline, the distinctions above remain usable; an unverified backend or toolchain capability stays unknown. Compilation, correctness and performance answer different questions. Kernel improvements affect the workload only through its actual replacement boundary and critical path.

## Sources

- [Ampere asynchronous copy](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#asynchronous-data-copy-from-global-memory-to-shared-memory) and [Hopper TMA, occupancy and clusters](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html).
- [PTX setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg) and [Cluster Launch Control](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel).
- [CUTLASS collective builder, pinned source](https://github.com/NVIDIA/cutlass/blob/0d2b201e8c1c4a03efa6e9c468161916e2334725/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu).

Reviewed 2026-11-04. Source-based guidance, not an H800 performance validation.
