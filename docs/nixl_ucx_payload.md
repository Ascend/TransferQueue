# SimpleStorage NIXL-UCX Host Payload Transfer

After setting `payload_transfer` to `nixl-ucx`, all non-empty payloads are transferred through NIXL-UCX;
ZMQ handles control messages only. This document describes the configuration and usage.

## 1. Check RDMA Devices

Run the following on every node running TQ or a Ray worker:

```bash
ls /sys/class/infiniband
rdma link show
ibv_devinfo
```

`ls` should list RDMA devices, and the ports shown by `rdma link show` should be `ACTIVE`.
`ibv_devinfo` does not show the provider dynamic library name. If no device is present or a port is
not active, check the driver, `rdma-core`, provider, and container device mappings first.

## 2. Install NIXL and TQ

Install the NIXL wheel:

```bash
python -m pip install nixl
```

The NIXL wheel includes the UCX runtime, but the system must still provide `rdma-core`,
`libibverbs`, and the provider for the network adapter.

Install TQ from the source directory:

```bash
python -m pip install -e .
```

## 3. Check the NIXL-UCX Backend

Run:

```bash
python - <<'PY'
from nixl import nixl_agent, nixl_agent_config

config = nixl_agent_config(
    enable_prog_thread=True,
    enable_listen_thread=True,
    listen_port=0,
    backends=["UCX"],
)
agent = nixl_agent("tq-nixl-check", config)
assert "UCX" in agent.backends
print("NIXL UCX backend is available")
PY
```

The command should output `NIXL UCX backend is available`. If it fails, see the common issues at the end.

## 4. Enable SimpleStorage NIXL-UCX Transfer

Enable NIXL-UCX in the TQ configuration:

```yaml
backend:
  storage_backend: SimpleStorage
  SimpleStorage:
    payload_transfer:
      backend: nixl-ucx
      ucx_env_vars: {}
```

`ucx_env_vars: {}` means that TQ does not set additional UCX environment variables. TQ and Ray workers
continue to use the `UCX_*` variables inherited when they were started. To specify a transport, device,
or GID, add the corresponding variables to `ucx_env_vars`.

If NIXL initialization or a transfer fails, TQ reports the error directly and does not fall back to ZMQ.
If the transport is not restricted, or if `UCX_TLS` includes `tcp`, UCX may use TCP.

### Common UCX Configuration

| Variable | Purpose | Reference value |
| --- | --- | --- |
| `UCX_TLS` | Restrict the transports available to UCX | `<available rc_* transport>,tcp,sm,self` |
| `UCX_NET_DEVICES` | Specify the RDMA device and port | `<rdma_device>:<port>` |
| `UCX_IB_GID_INDEX` | Specify the RoCE GID index | `<gid_index>` |
| `UCX_MODULE_DIR` | Specify the UCX transport module directory in the NIXL wheel | `<ucx_module_dir>` |

Restart TQ/Ray after making changes. Set the device name and GID index for each node.

### Memory Registration

Before NIXL registers memory, check the system limit in the current shell:

```bash
ulimit -l
```

If the value is too small, set it to `unlimited` in the shell that starts TQ/Ray:

```bash
ulimit -l unlimited
```

This setting applies only to the current shell and its child processes.

## 5. Verify SimpleStorage NIXL-UCX Transfer

After enabling it, the StorageUnit startup log will contain:

```text
SimpleStorage payload transfer selected: nixl-ucx device=ucx-auto gid_index=ucx-auto tls=ucx-auto
```

After a cross-node PUT/GET completes, the GET content should match the PUT content. The log and data
validation only confirm that the NIXL-UCX path is usable; to confirm RDMA, also check the payload lane.
`rc_*` indicates RDMA, while a TCP lane indicates that TCP is being used.

## Common Issues

### RDMA Devices Are Ready, but NIXL-UCX Fails to Start

If `ibv_devinfo` shows RDMA devices and active ports but NIXL initialization fails, the log typically contains:

```text
no userspace device-specific driver found
failed to open ... libuct_ib ...
NIXL_ERR_BACKEND
```

First confirm that the provider for the network adapter is installed. If the provider is installed but the
error persists, use a NIXL wheel compatible with the system `rdma-core/provider`. If no suitable wheel is
available, follow the [official NIXL source build instructions](https://github.com/ai-dynamo/nixl#prerequisites-for-source-build-linux)
to build UCX with multi-thread and verbs enabled:

```bash
python -m pip install meson ninja pybind11 tomlkit
git clone https://github.com/openucx/ucx.git <ucx_source>
cd <ucx_source>
git checkout <nixl_supported_ucx_version>
./autogen.sh
./contrib/configure-release-mt \
  --prefix=<ucx_install_prefix> \
  --enable-shared \
  --disable-static \
  --with-verbs
make -j"$(nproc)"
make install
```

Then configure NIXL to use this UCX:

```bash
git clone https://github.com/ai-dynamo/nixl.git <nixl_source>
cd <nixl_source>
python -m pip install .
meson setup build \
  -Ducx_path=<ucx_install_prefix> \
  -Dprefix=<nixl_install_prefix> \
  -Dbuildtype=release
ninja -C build
ninja -C build install
python -m pip install build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl
```

### `libnixl.so` Cannot Be Found

If `import nixl` reports `libnixl.so: cannot open shared object file`, the NIXL Python extension
cannot find the NIXL shared library; UCX initialization has not started yet.

First locate `libnixl.so` in the wheel:

```bash
NIXL_SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
find "${NIXL_SITE}" -name libnixl.so
```

If it is found, add its containing directory to `LD_LIBRARY_PATH` in the same shell that starts TQ/Ray:

```bash
NIXL_LIB_DIR=/path/to/directory/containing/libnixl.so
export LD_LIBRARY_PATH="${NIXL_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

If it is not found, reinstall the NIXL wheel corresponding to the reported error. For example:

```bash
python -m pip install --no-cache-dir --force-reinstall --no-deps nixl-cu12
```
