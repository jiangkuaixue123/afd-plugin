# Connector Package Layout

AFD connector implementations are grouped by backend:

- `gpu/`: GPU-only connector implementations. `p2pconnector` is implemented by
  `afd_plugin.connectors.gpu.p2p`.
- `npu/`: NPU-only connector implementations. `camp2pconnector` is implemented
  by `afd_plugin.connectors.npu.camp2p`.

Shared connector contracts, metadata containers, factory registration, and
backend-neutral helpers stay in `afd_plugin.connectors`.
