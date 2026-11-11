# GEM 矿机（宝石币 · BNB Chain）

本地 GPU 挖矿工具：出题 → 挖 nonce → 生成每轮提交文件。

## 题目

合约 `0xBEB8833c5a5290678e67Af65a4c12f9402bA404F`（BSC 主网）

```solidity
hash = keccak256(abi.encode(epochSeed, wallet, nonce))
submit if hash <= epochTarget        // submitBatch(uint256[])，单笔最多 64
```

- 每轮 30 分钟（+5 分钟宽限），每轮换 seed
- 每钱包每轮最多 99 票
- 必须持有 .ai Agent ID（SPACE ID 注册表 `0x60B9262456dE011F7d15a54CdD8c48bAaa8B022F` 的 balanceOf > 0）
- 轮末任何人可 `anchorEpoch()` 结算，之后 `claim(epoch)` 领奖

## 用法

```bash
python gem.py status      # 看本轮 seed / target / 票数 / 剩余时间
python gem.py auto        # 等 startMining，开挖后每轮自动挖并输出 epoch-N_*.txt
python gem.py mine --tickets 99
python gem.py calldata    # 打印 submitBatch calldata
```

依赖：Python 3 + OpenCL 驱动（AMD/NVIDIA），无需安装额外库。

## 每轮输出

```
epoch-<N>_agentA.txt   → w1-w4（见 agents.json）
epoch-<N>_文.txt       → 文
epoch-<N>_儿.txt       → 儿
epoch-<N>_大.txt       → 大
```

格式（agent 可直接解析）：

```
## 钱包名  地址  (99 票)
111111111,222222222,...   ← 99 个 nonce，逗号分隔
```

把这一行数字直接作为 `submitBatch(uint256[])` 的参数，用对应钱包签名发送。

## 提交步骤（每个钱包）

1. `submitBatch(nonces[0:64])`
2. `submitBatch(nonces[64:])`
3. 轮末调 `anchorEpoch()`（谁都能调）
4. `claim(epoch)` 领奖

## 文件

| 文件 | 说明 |
|---|---|
| `gem.py` | 主程序 |
| `engine.py` + `keccak.cl` | OpenCL keccak-256 引擎（96 字节 preimage） |
| `wallets.json` | 钱包列表 |
| `agents.json` | 每个 agent 负责哪些钱包 |
| `sample_脱敏_全部.txt` | 输出格式示例（假数据） |

## 性能

RX 570 ≈ 179 MH/s。初始难度 2^26 → 约 0.37 秒/票，99 票约 37 秒/钱包。
