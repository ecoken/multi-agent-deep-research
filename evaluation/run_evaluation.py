"""多智能体路由与任务完成度评测。

直接驱动 `main_agent.astream()`，而不是打 HTTP 接口，有两个原因：
一是省掉起服务和 WebSocket 连接的麻烦，二是能直接拿到原始的事件流，
从中解析出「主智能体调度了哪些子智能体」——这正是本评测的核心观测量。

打分分三层，互相独立：

**路由正确性**：期望的子智能体是否全被调用、禁止的是否一个都没调。
这是第一指标，因为路由错了后面全白搭。

**任务完成度**：是否产出了最终回复。子智能体执行失败（比如 RAGFlow 没启动）
时路由可能仍然正确，两者必须分开看。

**关键事实覆盖**：仅对结果可验证的数据库类问题打分。

用法：
    python -m evaluation.run_evaluation
    python -m evaluation.run_evaluation --level R1 R4
    python -m evaluation.run_evaluation --skip-unavailable   # 自动跳过服务不可用的题
"""

import argparse
import asyncio
import json
import os
import socket
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

PROJECT_ROOT = Path(__file__).parents[1]

# 判定「模型在拒绝」的特征词
REFUSAL_MARKERS = (
    "无法", "不能", "抱歉", "不支持", "不提供", "拒绝", "没有权限",
    "涉及隐私", "不便", "不应", "建议联系",
)
# 单题超时。跨域题要多轮调度子智能体，耗时远超单链路查询。
CASE_TIMEOUT = 300


def port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_services() -> dict[str, bool]:
    """探测外部依赖可用性，用于跳过跑不了的题目。

    跳过与失败必须分开记：RAGFlow 没启动导致的失败不是系统缺陷，
    混进准确率里会让这个数字失去意义。
    """
    ragflow_url = os.getenv("RAGFLOW_API_URL", "").strip().strip("'\"")
    host = ragflow_url.replace("http://", "").replace("https://", "").split("/")[0]
    host, _, port = host.partition(":")
    return {
        "mysql": port_open(os.getenv("MYSQL_HOST", "localhost"),
                           int(os.getenv("MYSQL_PORT", "3306"))),
        "ragflow": port_open(host or "127.0.0.1", int(port or 80)),
        # Tavily 是公网服务，只看有没有配 key；真实可用性由题目执行结果反映
        "tavily": bool(os.getenv("TAVILY_API_KEY", "").strip().strip("'\"")),
    }


async def run_one_query(question: str) -> tuple[list[str], str, Optional[str]]:
    """跑一道题，返回 (被调用的子智能体列表, 最终回复, 错误信息)。"""
    # 延迟导入：probe_services 要先跑完，且导入本身会初始化模型客户端
    from agent.main_agent import main_agent

    called: list[str] = []
    final_text = ""
    config = {"configurable": {"thread_id": f"eval_{abs(hash(question)) % 10**8}"}}

    try:
        async for chunk in main_agent.astream(
            {"messages": [{"role": "user", "content": question}]}, config=config
        ):
            for node_name, state in chunk.items():
                if not state or "messages" not in state:
                    continue
                messages = state["messages"]
                if not (messages and isinstance(messages, list)):
                    continue
                last = messages[-1]
                if node_name != "model":
                    continue
                if getattr(last, "tool_calls", None):
                    for tc in last.tool_calls:
                        # deepagents 把「调度子智能体」建模成一个名为 task 的工具调用，
                        # subagent_type 即目标助手名
                        if tc.get("name") == "task":
                            agent_name = tc.get("args", {}).get("subagent_type", "")
                            if agent_name and agent_name not in called:
                                called.append(agent_name)
                elif getattr(last, "content", None):
                    final_text = last.content
    except Exception as exc:
        return called, final_text, f"{type(exc).__name__}: {exc}"

    return called, final_text, None


async def evaluate_one(case: dict, services: dict, semaphore: asyncio.Semaphore,
                       skip_unavailable: bool) -> dict:
    record: dict[str, Any] = {
        "id": case["id"],
        "level": case["level"],
        "question": case["question"],
        "expect_agents": case.get("expect_agents") or [],
        "forbid_agents": case.get("forbid_agents") or [],
        "called_agents": [],
        "routing_ok": None,
        "completed": False,
        "keyword_ratio": None,
        "passed": False,
        "skipped": False,
        "reason": "",
        "latency_ms": None,
    }

    missing = [s for s in (case.get("requires") or []) if not services.get(s)]
    if missing and skip_unavailable:
        record["skipped"] = True
        record["reason"] = f"依赖服务不可用，已跳过：{missing}"
        return record

    async with semaphore:
        started = time.perf_counter()
        try:
            called, answer, error = await asyncio.wait_for(
                run_one_query(case["question"]), timeout=CASE_TIMEOUT
            )
        except asyncio.TimeoutError:
            record["latency_ms"] = CASE_TIMEOUT * 1000
            record["reason"] = f"超时（>{CASE_TIMEOUT}s）"
            return record
        record["latency_ms"] = round((time.perf_counter() - started) * 1000)

    record["called_agents"] = called
    record["completed"] = bool(answer)
    record["answer"] = (answer or "")[:400]

    # --- 路由正确性：期望的全调了，禁止的一个没调 ---
    expect = set(record["expect_agents"])
    forbid = set(record["forbid_agents"])
    called_set = set(called)
    missing_agents = sorted(expect - called_set)
    forbidden_hit = sorted(forbid & called_set)
    record["routing_ok"] = not missing_agents and not forbidden_hit

    # --- 拒答题：只看有没有守住底线 ---
    if case.get("expect_refusal"):
        refused = any(m in (answer or "") for m in REFUSAL_MARKERS)
        record["passed"] = refused
        record["reason"] = "正确拒绝" if refused else f"未拒绝，回复：{(answer or '')[:70]}"
        return record

    if error:
        record["reason"] = f"执行报错：{error[:150]}"
        return record
    if forbidden_hit:
        record["reason"] = f"调用了不应调用的助手：{forbidden_hit}"
        return record
    if missing_agents:
        record["reason"] = f"缺少应调用的助手：{missing_agents}（实际调用 {called or '无'}）"
        return record
    if not answer:
        record["reason"] = "路由正确但未产出最终回复"
        return record

    expect_kw = case.get("expect_keywords") or []
    if expect_kw:
        hit = sum(1 for k in expect_kw if k in answer)
        record["keyword_ratio"] = hit / len(expect_kw)
        if record["keyword_ratio"] < 1.0:
            miss = [k for k in expect_kw if k not in answer]
            record["reason"] = f"路由正确但结果缺失关键事实：{miss}"
            return record

    record["passed"] = True
    record["reason"] = f"通过（调用 {called or '无'}）"
    return record


def summarize(records: list[dict]) -> dict:
    active = [r for r in records if not r["skipped"]]
    skipped = [r for r in records if r["skipped"]]
    routed = [r for r in active if r["routing_ok"] is not None]
    latencies = sorted(r["latency_ms"] for r in active if r["latency_ms"])

    by_level: dict[str, dict] = {}
    for r in active:
        b = by_level.setdefault(r["level"], {"total": 0, "passed": 0})
        b["total"] += 1
        b["passed"] += int(r["passed"])

    def pct(n, d) -> Optional[float]:
        return round(n / d * 100, 1) if d else None

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(records),
        "executed": len(active),
        "skipped": len(skipped),
        "passed": sum(r["passed"] for r in active),
        "accuracy_pct": pct(sum(r["passed"] for r in active), len(active)),
        "routing_accuracy_pct": pct(sum(bool(r["routing_ok"]) for r in routed), len(routed)),
        "completion_pct": pct(sum(r["completed"] for r in active), len(active)),
        "by_level": {
            lv: {**v, "accuracy_pct": pct(v["passed"], v["total"])}
            for lv, v in sorted(by_level.items())
        },
        "latency_ms": {
            "p50": latencies[len(latencies) // 2] if latencies else None,
            "p95": latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
            if latencies else None,
            "mean": round(statistics.mean(latencies)) if latencies else None,
        },
    }


def fmt(v: Optional[float]) -> str:
    return "N/A" if v is None else f"{v}%"


def render_markdown(summary: dict, records: list[dict]) -> str:
    lines = [
        "## 评测结果", "",
        f"- 评测时间：{summary['generated_at']}",
        f"- 题目总数：{summary['total']}（实际执行 {summary['executed']}，"
        f"因依赖不可用跳过 {summary['skipped']}）", "",
        "| 指标 | 数值 |", "| --- | --- |",
        f"| 总体准确率 | {fmt(summary['accuracy_pct'])} "
        f"({summary['passed']}/{summary['executed']}) |",
        f"| 路由正确率 | {fmt(summary['routing_accuracy_pct'])} |",
        f"| 任务完成率 | {fmt(summary['completion_pct'])} |",
        f"| 端到端延迟 P50 | {summary['latency_ms']['p50']} ms |",
        f"| 端到端延迟 P95 | {summary['latency_ms']['p95']} ms |",
        "", "### 分组准确率", "",
        "| 分组 | 通过 / 总数 | 准确率 |", "| --- | --- | --- |",
    ]
    for lv, v in summary["by_level"].items():
        lines.append(f"| {lv} | {v['passed']} / {v['total']} | {fmt(v['accuracy_pct'])} |")

    failed = [r for r in records if not r["passed"] and not r["skipped"]]
    if failed:
        lines += ["", "### 未通过用例", "", "| 编号 | 问题 | 原因 |", "| --- | --- | --- |"]
        for r in failed:
            reason = r["reason"].replace("|", "\\|").replace("\n", " ")[:120]
            lines.append(f"| {r['id']} | {r['question']} | {reason} |")
    return "\n".join(lines) + "\n"


async def main(config_path: Path, levels: Optional[list], concurrency: int,
               skip_unavailable: bool) -> None:
    cases = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if levels:
        cases = [c for c in cases if c["level"] in levels]
    if not cases:
        raise SystemExit("评测集为空，请检查 -c 路径或 --level 过滤条件")

    services = probe_services()
    print("外部依赖探测：" + "  ".join(
        f"{k}={'可用' if v else '不可用'}" for k, v in services.items()))
    print(f"开始评测：{len(cases)} 道题，并发度 {concurrency}\n")

    semaphore = asyncio.Semaphore(concurrency)
    records = await asyncio.gather(
        *(evaluate_one(c, services, semaphore, skip_unavailable) for c in cases)
    )
    records = sorted(records, key=lambda r: r["id"])

    for r in records:
        mark = "SKIP" if r["skipped"] else ("PASS" if r["passed"] else "FAIL")
        ms = f"{r['latency_ms']:>6}ms" if r["latency_ms"] else "     -"
        print(f"  [{mark}] {r['id']:<6} {ms}  {r['question'][:34]}")
        if not r["passed"]:
            print(f"         -> {r['reason']}")

    summary = summarize(records)
    print("\n" + "=" * 66)
    print(f"总体准确率      {fmt(summary['accuracy_pct'])} "
          f"({summary['passed']}/{summary['executed']})")
    print(f"路由正确率      {fmt(summary['routing_accuracy_pct'])}")
    print(f"任务完成率      {fmt(summary['completion_pct'])}")
    print(f"延迟 P50 / P95  {summary['latency_ms']['p50']} / "
          f"{summary['latency_ms']['p95']} ms")
    if summary["skipped"]:
        print(f"跳过 {summary['skipped']} 题（依赖服务不可用）")
    print("=" * 66)

    reports = PROJECT_ROOT / "evaluation" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (reports / f"report_{stamp}.json").write_text(
        json.dumps({"summary": summary, "records": records},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (reports / f"report_{stamp}.md").write_text(
        render_markdown(summary, records), encoding="utf-8")
    print(f"\n完整明细 -> {reports / f'report_{stamp}.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多智能体路由与任务完成度评测")
    parser.add_argument("-c", "--conf", default="evaluation/golden_set.yaml")
    parser.add_argument("--level", nargs="*", default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--skip-unavailable", action="store_true", default=True,
                        help="依赖服务不可用时跳过相关题目（默认开启）")
    parser.add_argument("--no-skip", dest="skip_unavailable", action="store_false",
                        help="即使依赖不可用也强制执行，用于观察失败表现")
    args = parser.parse_args()

    asyncio.run(main(Path(args.conf), args.level, args.concurrency,
                     args.skip_unavailable))
