# TransferQueue 端到端数据一致性校验实施计划 (Consistency Validation Master Plan)

## 1. 背景与目标 (Background & Objective)
本项目旨在开发一个端到端的自动化测试脚本 (`scripts/test_e2e_lifecycle_consistency.py`)，用于验证 `TransferQueue` 在复杂场景下的数据一致性和生命周期管理能力。
核心目标是确保在数据经过 **分片存储 (Sharding)**、**跨节点传输**、**分批写入 (Multi-round Put)**、**动态更新 (Update/Overwrite)** 等操作后，数据的完整性和状态机（Production/Consumption Status）的正确性。

## 2. 核心原则 (Core Principles)
1. **Public API Only**: 测试代码仅允许使用 `TransferQueueClient` 的公共接口（如 `put`, `get_meta`, `get_data`, `check_production_status` 等），**严禁**调用 `StorageUnit` 或 `Controller` 的内部私有方法。
2. **Complex Data Types**: 所有数据传输场景必须覆盖全量复杂数据类型（见下文）。
3. **Environment**: 必须模拟真实分布式环境，启动 **2个及以上 Storage Units** 以强制触发 Manager 的自动分片逻辑。

---

## 3. 详细实施方案 (Implementation Details)

### 3.1 测试环境配置
参考 `scripts/performance_test.py` 的初始化逻辑：
- **Ray Cluster**: 使用 `pytest` fixture 启动。
- **Components**:
    - **Controller**: 1个 (`polling_mode=True`)。
    - **Storage Units**: **2个** (Capacity=10000)，确保数据会分布在不同 Unit 上。
    - **Client**: 初始化一般 Client，配置 `AsyncSimpleStorageManager`。

### 3.2 通用复杂数据生成器 (Universal Data Generator)
实现 `generate_complex_data(indices, fields_subset=None)`，生成 `TensorDict`，必须包含：

| 类型 (Type) | 字段 (Field) | 特征 (Characteristics) |
| :--- | :--- | :--- |
| **Standard Tensor** | `tensor_f32`, `tensor_i64` | Float32/Int64, 标准形状 |
| **Nested Tensor** | `nested_jagged` | `layout=torch.jagged`, 变长样本 |
| | `nested_strided` | `layout=torch.strided` (若支持) |
| **Lists** | `list_int`, `list_str` | Python 原生列表 |
| **NumPy** | `np_array`, `np_obj` | 标准 Array 及 Object Array (混合类型) |
| **Special Values** | `special_val` | 包含 **NaN** 和 **Inf** (验证传输稳定性) |
| **Non-Tensor** | **`non_tensor_stack`** | 使用 `tensordict.tensorclass.NonTensorData` 封装 |

### 3.3 验证场景 (Verification Scenarios)

#### 场景一：核心读写一致性 (Core Consistency)
- **操作**: Put 写入上述全量复杂数据 -> Get 读取。
- **验证**: 
    - 输入输出的 Hash 值完全一致 (使用结构无关 Hash)。
    - `NaN` 保持为 `NaN`，`Inf` 保持为 `Inf`。
    - `NonTensorData` 解包后内容无损。

#### 场景二：跨分片操作与复杂更新 (Cross-Partition & Complex Update)
- **配置**: 依赖 Manager 自动将不同 Indices 分片到 2 个 Storage Units。
- **步骤**:
    1. **Put A**: Indices `0-19` (含全量复杂字段)。
    2. **Put B**: Indices `20-39` (含全量复杂字段)。
    3. **Update (Cross-Shard)**: Indices `10-29` (跨越分片边界)。
        - **Modify**: 修改 `nested_jagged` (变长), `non_tensor_stack` 等字段的值。
        - **Add**: 新增字段 `new_extra_tensor` 和 `new_extra_non_tensor`。
    4. **Get Full**: Indices `0-39`。
- **验证**:
    - `0-9`: 保持 Put A 旧值。
    - `10-29` (Update区): 旧字段更新成功，**新字段存在且正确**。
    - `30-39`: 保持 Put B 旧值。

#### 场景三：生命周期状态管理 (Status Lifecycle)
- **重点**: 验证 **分字段多轮 Put** 对 `Production Status` 的影响。
- **步骤**:
    1. **Round 1 Put**: Indices `0-9`, 仅写入 `Set_A` 字段。
        - Check Production(`Set_A`): **True**。
        - Check Production(`Set_B`): **False**。
        - Check Production(`Set_A` + `Set_B`): **False**。
    2. **Round 2 Put**: Indices `0-9`, 补全 `Set_B` 字段。
        - Check Production(`Set_A` + `Set_B`): **True**。
    3. **Consumption**:
        - Check Consumption: **False**。
        - Get Data (`Set_A` + `Set_B`).
        - Check Consumption: **True**。

#### 场景四：自定义元数据持久化 (Custom Metadata)
- **操作**: `put` 数据 -> `set_custom_meta` (上传 Sample-level dict) -> `get_meta`。
- **验证**: 读取到的 `custom_meta` 与上传内容完全一致。

#### 场景五：重置与清理 (Reset & Clear)
- **Reset Consumption**: 
    - 消费后调用 `reset_consumption`。
    - 验证状态变回 `Not Consumed`。
    - 验证数据可再次 `get_meta` 获取。
- **Clear Partition**: 
    - 调用 `clear_partition`。
    - 验证数据物理删除 (`get_meta` 返回空或 `check_production` 为 False)。

---

## 4. 执行指南 (Execution Guide)
1. **脚本位置**: `scripts/test_e2e_lifecycle_consistency.py`
2. **运行命令**: 
   ```bash
   ./venv/bin/python -m pytest scripts/test_e2e_lifecycle_consistency.py -v
   ```
3. **依赖**: 确保 `pytest`, `pytest-asyncio` 已安装。

## 5. 注意事项 (Notes)
- 保持 `client.py` 的接口纯净性，如果发现 Client 功能不足以支持测试（如由 Sync/Async 接口缺失导致），应先在 Client 层补充对应公共接口，而非在测试脚本中 Hack 内部实现。
