---
tag: NEW-nvidia-h100
kind: hardware
slug: nvidia-h100
name: NVIDIA H100
aliases: [H100, H100 Tensor Core GPU, H100 SXM5, H100 PCIe]
area: ai-hardware/gpus
level: core
summary: >-
  NVIDIA's data center GPU of the Hopper architecture, announced in March 2022, with FP8 Tensor Cores and a Transformer Engine, in SXM5 and PCIe versions.
history:
  year: 2022
  note: "Announced at GTC on 22 March 2022, with availability from the third quarter of 2022 (NVIDIA's announcement)."
same_as:
  wikipedia: Hopper_(microarchitecture)
sources:
  - {paper: "NVIDIA (2023)", title: "NVIDIA H100 Tensor Core GPU Architecture, whitepaper V1.04", locator: "Table 1 and Table 3", note: "the final figures of the SXM5 and PCIe versions, and the comparison with A100"}
  - {paper: "NVIDIA (2024)", title: "NVIDIA H100 Tensor Core GPU datasheet", locator: "p. 2"}
  - {paper: "NVIDIA (2022)", title: "NVIDIA H100 PCIe GPU product brief, PB-11133-001_v02", locator: "Table 1 and Table 2"}
  - {url: "https://nvidianews.nvidia.com/news/nvidia-announces-hopper-architecture-the-next-generation-of-accelerated-computing", title: "NVIDIA Announces Hopper Architecture, the Next Generation of Accelerated Computing", locator: "p. 1"}
vendor: NVIDIA
type: GPU
released: 2022-03-22
---

The NVIDIA H100 is a data center GPU of NVIDIA's Hopper architecture, announced at GTC in March 2022.
It is built on TSMC's 4N process with 80 billion transistors.
Its Tensor Cores add an 8-bit floating point format, FP8, which its Transformer Engine uses to speed up transformer models.
It comes as an SXM5 board and as a PCIe card, and the two differ in memory, bandwidth and power.

## Architecture

**The chip.** The full GH100 GPU is fabricated on TSMC's 4N process customized for NVIDIA, with 80 billion transistors and a die size of 814 mm2. {#c-die}
NVIDIA writes that the H100 GPUs are built mainly for data center and edge compute workloads for AI, HPC and data analytics, not for graphics processing. {#c-workloads}

**Two versions.** The H100 SXM5 has 132 streaming multiprocessors (SMs) and 80 GB of HBM3 memory in five stacks. {#c-sxm5}
The H100 PCIe has 114 SMs and 80 GB of HBM2e memory in five stacks. {#c-pcie}
So every figure below names its version.

| Figure | H100 SXM5 | H100 PCIe |
|---|---|---|
| Memory | {fact:nvidia-h100/memory.bytes.sxm5}, {fact:nvidia-h100/memory.type.sxm5} | {fact:nvidia-h100/memory.bytes.pcie}, {fact:nvidia-h100/memory.type.pcie} |
| Memory bandwidth | {fact:nvidia-h100/memory.bandwidth.sxm5} | {fact:nvidia-h100/memory.bandwidth.pcie} |
| Peak FP8 Tensor Core, dense | {fact:nvidia-h100/peak_flops.fp8_tensor.sxm5} | {fact:nvidia-h100/peak_flops.fp8_tensor.pcie} |
| Peak FP16 Tensor Core, dense | {fact:nvidia-h100/peak_flops.fp16_tensor.sxm5} | {fact:nvidia-h100/peak_flops.fp16_tensor.pcie} |
| Thermal design power | {fact:nvidia-h100/power.sxm5} | {fact:nvidia-h100/power.pcie} |

The two NVIDIA documents give different PCIe memory bandwidths, so the table shows both.
With the sparsity feature, NVIDIA gives an effective FP8 rate of {fact:nvidia-h100/peak_flops.fp8_tensor_sparse.sxm5} for the SXM5.
The rates in the table are dense, without that feature.

**Tensor Cores and FP8.** On a per SM basis, the H100 Tensor Cores deliver 2x the matrix multiply-accumulate rate of the A100 SM on equivalent data types, and 4x using FP8 compared with the earlier 16-bit types. {#c-mma}
The FP8 Tensor Cores take two FP8 input types: E4M3, with 4 exponent bits and 3 mantissa bits, and E5M2, with 5 exponent bits and 2 mantissa bits. {#c-fp8-types}
E4M3 has more precision and less dynamic range, and E5M2 has a wider dynamic range and less precision. {#c-fp8-range}
FP8 halves data storage requirements and doubles throughput compared with FP16 or BF16. {#c-fp8-half}

**Transformer Engine.** The Transformer Engine combines software and custom Hopper Tensor Core technology to accelerate the training and inference of transformer models. {#c-te}
It chooses between FP8 and 16-bit calculations and handles the re-casting and scaling between them in each layer. {#c-te-cast}
NVIDIA reports up to 9x faster AI training and up to 30x faster AI inference on large language models, compared with the A100. {#c-te-speedup}

**Memory.** NVIDIA writes that the SXM5's HBM3 delivers over 3 TB/sec, about 2x the memory bandwidth of the A100, and that the PCIe card's HBM2e gives over 2 TB/sec. {#c-hbm}
The exact figures are in the table above.

**Interconnect.** Fourth-generation NVLink gives {fact:nvidia-h100/interconnect.nvlink.sxm5} of total bandwidth for multi-GPU IO, which NVIDIA puts at 7x the bandwidth of PCIe Gen 5. {#c-nvlink}
Two H100 PCIe cards can be joined by three NVLink bridges, for 900 GB/s of bidirectional bandwidth between them. {#c-pcie-bridge}

**Programming model.** Hopper adds thread block clusters, a level of the CUDA hierarchy between thread blocks and grids. {#c-clusters}
Clusters let thread blocks on several SMs synchronize and share data. {#c-clusters-sm}
The Tensor Memory Accelerator moves data so that a few CUDA threads can drive the full memory bandwidth, while the others do general computation. {#c-tma}

## Use in AI

This section lists the model cards that name the H100 as training hardware; none did when this page was written.
NVIDIA's own figures for AI workloads are the Transformer Engine claims above.
At the announcement, NVIDIA also presented the DGX H100 system, with eight H100 GPUs. {#c-dgx}

## Comparison

NVIDIA's whitepaper compares the H100 with the A100, its predecessor, in Table 3.
The A100 there has 40 GB of HBM2 memory.

**Chip.** The A100 has 54.2 billion transistors on a die of 826 mm2, made on TSMC's 7 nm N7 process. {#c-a100-chip}
The H100 has {fact:nvidia-h100/transistors} transistors on a die of 814 mm2, on {fact:nvidia-h100/process}.

**SMs.** The A100 has 108 SMs, against 132 for the H100 SXM5. {#c-a100-sms}
NVIDIA counts this as a 22% increase in SMs. {#c-22}

**Memory.** The A100 has 1555 GB/sec of memory bandwidth from 40 GB of HBM2. {#c-a100-mem}
The H100 SXM5 has {fact:nvidia-h100/memory.bandwidth.sxm5} from 80 GB of HBM3.

**Compute.** The A100 peaks at 312 TFLOPS on FP16 Tensor Core operations, dense, and has no FP8 Tensor Core rate. {#c-a100-fp16}
The H100 SXM5 peaks at {fact:nvidia-h100/peak_flops.fp16_tensor.sxm5} on FP16 Tensor Core operations, dense, and at {fact:nvidia-h100/peak_flops.fp8_tensor.sxm5} with FP8.
NVIDIA puts the overall gain at about 6x the compute of the A100: more SMs, Tensor Cores 2x faster per SM, and another 2x from FP8 and the Transformer Engine. {#c-6x}

**Power.** The A100 has a thermal design power of 400 W. {#c-a100-tdp}
The H100 SXM5 has {fact:nvidia-h100/power.sxm5}, and the H100 PCIe has {fact:nvidia-h100/power.pcie}.

## History

NVIDIA announced the Hopper architecture and the H100 at GTC on 22 March 2022. {#c-announce}
Hopper is named for Grace Hopper and succeeds the NVIDIA Ampere architecture. {#c-ampere}
The announcement said the H100 would come in SXM and PCIe form factors. {#c-forms}
It gave availability starting in the third quarter of 2022. {#c-q3}

NVIDIA's datasheet of 2024 lists the H100 SXM next to the H100 NVL, a PCIe card with 94 GB of memory and 3.9 TB/s of memory bandwidth. {#c-nvl}
The H100 PCIe of 2022 is not in that datasheet.

## References
