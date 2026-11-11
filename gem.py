"""GEM (宝石币) PoW miner — BSC, local GPU.

Puzzle: keccak256(abi.encode(epochSeed, wallet, nonce)) <= epochTarget
Submit: submitBatch(uint256[]) from the SAME wallet (must hold .ai Agent ID).
"""
import argparse, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine

GEM = "0xBEB8833c5a5290678e67Af65a4c12f9402bA404F"
RPC = ["https://bsc.publicnode.com", "https://bsc-dataseed.bnbchain.org"]
SEL = {
    "started": "0x3c8e4b0b",  # placeholder, replaced below by computed selectors
}


def keccak_selector(sig: str) -> str:
    # minimal keccak256 via sha3 library? use eth-hash if present, else fallback table
    return SELECTORS[sig]


SELECTORS = {
    "started()": "0x1f2698ab",
    "epoch()": "0x900cf0cf",
    "epochSeed()": "0x0cfe406e",
    "epochTarget()": "0x9f6f37d8",
    "epochTickets()": "0x6476c7f7",
    "epochPool()": "0x0695563f",
    "epochStart()": "0x15e5a1e5",
    "ticketsOf(uint64,address)": "0x947f09c7",
    "submitBatch(uint256[])": "0xf18858cf",
    "claim(uint64)": "0xaab8ab0c",
    "EPOCH_SECONDS()": "0x5262265a",
    "MAX_TICKETS_PER_WALLET()": "0x229df6d6",
}


def rpc(method, params, url=None):
    urls = [url] if url else RPC
    last = None
    for u in urls:
        for _ in range(2):
            try:
                req = urllib.request.Request(u, data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
                                             headers={"content-type": "application/json"})
                with urllib.request.urlopen(req, timeout=15) as r:
                    j = json.loads(r.read().decode())
                if "result" in j:
                    return j["result"]
                last = j.get("error")
            except Exception as e:
                last = str(e)
    raise RuntimeError("rpc failed: %s" % (last,))


def call(fn, *args_hex):
    data = SELECTORS[fn] + "".join(args_hex)
    return rpc("eth_call", [{"to": GEM, "data": data}, "latest"])


def state():
    started = int(call("started()"), 16)
    epoch = int(call("epoch()"), 16)
    seed = call("epochSeed()")
    target = int(call("epochTarget()"), 16)
    tickets = int(call("epochTickets()"), 16)
    pool = int(call("epochPool()"), 16)
    start = int(call("epochStart()"), 16)
    dur = int(call("EPOCH_SECONDS()"), 16)
    cap = int(call("MAX_TICKETS_PER_WALLET()"), 16)
    return dict(started=started, epoch=epoch, seed=seed, target=target,
                tickets=tickets, pool=pool, start=start, dur=dur, cap=cap)


def cmd_status(_):
    s = state()
    left = s["start"] + s["dur"] - int(time.time())
    tries = ((1 << 256) - 1) // s["target"] if s["target"] else 0
    print("started      :", s["started"])
    if not s["started"]:
        print("状态：未开挖（等项目方 startMining）")
        return
    print("epoch        :", s["epoch"])
    print("seed         :", s["seed"])
    print("target       : 0x%064x" % s["target"])
    print("难度(期望哈希): %.3fM" % (tries / 1e6))
    print("本轮票数      :", s["tickets"], "| 奖池:", s["pool"] / 1e18)
    print("剩余时间      : %d 秒" % left)


def build_batch_calldata(nonces):
    """submitBatch(uint256[]) calldata"""
    body = "".join("%064x" % n for n in nonces)
    return SELECTORS["submitBatch(uint256[])"] + ("%064x" % 32) + ("%064x" % len(nonces)) + body


def cmd_calldata(a):
    data = json.load(open(a.out, encoding="utf-8"))
    for item in data:
        print("== %s %s (epoch %s, %d 票)" % (item["name"], item["wallet"], item.get("epoch"), len(item["nonces"])))
        for i in range(0, len(item["nonces"]), 64):
            chunk = item["nonces"][i:i + 64]
            print("  batch %d (%d 票):" % (i // 64 + 1, len(chunk)))
            print("  " + build_batch_calldata(chunk))
        print()


def cmd_mine(a):
    s = state()
    if not s["started"] or s["pool"] == 0:
        print("还没开挖 / 本轮无奖池，退出")
        return
    wallets = json.load(open(a.wallets, encoding="utf-8"))
    gpu = engine.GPU(os.path.join(HERE, "keccak.cl"))
    out = []
    for w in wallets:
        addr = w["address"].lower().replace("0x", "").rjust(64, "0")
        prefix = s["seed"].replace("0x", "") + addr
        job = engine.Job(gpu, w["address"], prefix, "0x%064x" % s["target"])
        nonces = []
        base = 0
        t0 = time.time()
        while len(nonces) < a.tickets and time.time() - t0 < a.seconds:
            found = job.scan(base, a.batch)
            base += a.batch
            for n in found:
                if n not in nonces:
                    nonces.append(n)
                    print("  ", w["name"], "票", len(nonces), "nonce", n, flush=True)
                    if len(nonces) >= a.tickets:
                        break
        batches = [build_batch_calldata(nonces[i:i + 64]) for i in range(0, len(nonces), 64)]
        out.append({"name": w["name"], "wallet": w["address"], "epoch": s["epoch"],
                    "nonces": nonces, "batches": batches, "count": len(nonces)})
        print("%s: %d 票, %.1fs" % (w["name"], len(nonces), time.time() - t0), flush=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)
    print("saved ->", a.out)


def git_publish(paths, epoch):
    """把本轮文件推到 GitHub（agent 可直接用 raw URL 读取）"""
    import subprocess
    repo = os.path.dirname(os.path.abspath(__file__))
    try:
        subprocess.run(["git", "-C", repo, "add"] + list(paths), check=True, capture_output=True, timeout=60)
        subprocess.run(["git", "-C", repo, "commit", "-m", "epoch %d tickets" % epoch], capture_output=True, timeout=60)
        r = subprocess.run(["git", "-C", repo, "push"], capture_output=True, timeout=120, text=True)
        if r.returncode == 0:
            print("pushed -> https://github.com/StarsN/gem-miner", flush=True)
        else:
            print("git push failed:", (r.stderr or "")[:200], flush=True)
    except Exception as e:
        print("git publish failed:", e, flush=True)


def write_epoch_files(epoch, data, outdir):
    """One file per epoch PER AGENT GROUP (agent-friendly)."""
    groups = json.load(open(os.path.join(HERE, "agents.json"), encoding="utf-8"))
    by_name = {it["name"]: it for it in data}
    made = []
    for gname, members in groups.items():
        items = [by_name[m] for m in members if m in by_name]
        if not items:
            continue
        base = os.path.join(outdir, "epoch-%d_%s" % (epoch, gname))
        json.dump({
            "chain": "BSC",
            "contract": GEM,
            "epoch": epoch,
            "function": "submitBatch(uint256[])",
            "max_per_tx": 64,
            "agent": gname,
            "steps": [
                "1) 对每个钱包，用该钱包发 submitBatch(nonces[0:64])",
                "2) 同一钱包再发 submitBatch(nonces[64:])",
                "3) 本轮结束后（+5 分钟宽限）调用 anchorEpoch() 结算",
                "4) 结算后调用 claim(epoch) 领取奖励",
            ],
            "wallets": [{"name": it["name"], "address": it["wallet"], "count": len(it["nonces"]),
                         "nonces": it["nonces"], "batches": it["batches"]} for it in items],
        }, open(base + ".json", "w", encoding="utf-8"), indent=2)

        lines = ["# GEM 第 %d 轮 · %s 提交数据" % (epoch, gname), "",
                 "合约 %s（BSC）" % GEM,
                 "函数 submitBatch(uint256[])，每笔最多 64 票，必须用对应钱包发。", ""]
        for it in items:
            lines += ["## %s  %s  (%d 票)" % (it["name"], it["wallet"], len(it["nonces"])), ""]
            lines += ["nonces:", ",".join(str(n) for n in it["nonces"]), ""]
            for i, b in enumerate(it["batches"]):
                lines += ["calldata batch%d (%d 票): %s" % (i + 1, min(64, len(it["nonces"]) - i * 64), b), ""]
        open(base + ".txt", "w", encoding="utf-8").write("\n".join(lines))
        made.append(base)
    print("saved -> " + ", ".join(made), flush=True)
    if made:
        git_publish([b + ".json" for b in made] + [b + ".txt" for b in made], epoch)


def cmd_auto(a):
    print("等待 startMining ...", flush=True)
    while True:
        try:
            s = state()
            if s["started"]:
                print("!!! MINING STARTED !!! epoch=%d pool=%.0f" % (s["epoch"], s["pool"] / 1e18), flush=True)
                open(os.path.join(HERE, "STARTED.flag"), "w").write(str(int(time.time())))
                break
            print("[%s] not started" % time.strftime("%H:%M:%S"), flush=True)
        except Exception as e:
            print("poll error:", e, flush=True)
        time.sleep(a.interval)
    last = -1
    while True:
        try:
            s = state()
            if s["epoch"] != last and s["pool"] > 0:
                last = s["epoch"]
                print("=== 第 %d 轮开始 ===" % s["epoch"], flush=True)
                cmd_mine(a)
                data = json.load(open(a.out, encoding="utf-8"))
                write_epoch_files(s["epoch"], data, os.path.dirname(a.out))
        except Exception as e:
            print("auto error:", e, flush=True)
        time.sleep(a.interval)


def cmd_watch(a):
    last = None
    while True:
        try:
            s = state()
            if s["started"] and last != 1:
                print("!!! MINING STARTED !!! epoch 0  seed", s["seed"], flush=True)
                open(os.path.join(HERE, "STARTED.flag"), "w").write(str(int(time.time())))
            if s["started"]:
                print("[%s] epoch=%d tickets=%d pool=%.0f target_tries=%.2fM" % (
                    time.strftime("%H:%M:%S"), s["epoch"], s["tickets"], s["pool"] / 1e18,
                    (((1 << 256) - 1) // s["target"]) / 1e6 if s["target"] else 0), flush=True)
            else:
                print("[%s] not started" % time.strftime("%H:%M:%S"), flush=True)
            last = s["started"]
        except Exception as e:
            print("watch error:", e, flush=True)
        time.sleep(a.interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "mine", "watch", "calldata", "auto"])
    ap.add_argument("--wallets", default=os.path.join(HERE, "wallets.json"))
    ap.add_argument("--tickets", type=int, default=99)
    ap.add_argument("--seconds", type=int, default=240)
    ap.add_argument("--batch", type=int, default=1 << 23)
    ap.add_argument("--out", default=os.path.join(HERE, "tickets.json"))
    ap.add_argument("--interval", type=int, default=30)
    a = ap.parse_args()
    {"status": cmd_status, "mine": cmd_mine, "watch": cmd_watch, "calldata": cmd_calldata, "auto": cmd_auto}[a.cmd](a)


if __name__ == "__main__":
    main()
