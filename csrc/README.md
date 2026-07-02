# Native Source Layout

The native sources are grouped by device backend:

- `npu/`: Ascend/CANN custom operators and the NPU torch extension. The existing
  `a2e` and `e2a` ACLNN operators live here. See `npu/README.md` for build
  instructions.
- `gpu/`: reserved for GPU native sources.
