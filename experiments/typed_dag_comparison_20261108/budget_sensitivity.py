"""在主实验完成后登记并执行更高预算的奇偶任务敏感性检查。

这个检查明确属于后续分析，不能替换每阶段 256 次的主实验，也不能称为
预注册主结论。仍使用同样的 30 个配对种子，只将每阶段评估增至 1024 次。
冻结的提案器、原语、容量限制和模块入库策略全部通过原实现复用。
"""
from __future__ import annotations

import itertools
import json
from run_comparison import HERE, PLAN, dump, sources, run_one, summarize, check_witnesses


def main():
    frozen = json.loads((HERE / "protocol_frozen.json").read_text(encoding="utf-8"))
    assert frozen["sources"] == sources() and frozen["protocol"] == PLAN
    primary = json.loads((HERE / "artifacts/formal/audit.json").read_text(encoding="utf-8"))
    assert primary["all_checks_passed"] and primary["runs"] == 240
    plan = {
        "phase": "post_primary_parity_budget_sensitivity",
        "planned_after_primary_results": True,
        "reason": "Neither library condition solved parity stages 2 or 3 at 256 evaluations per stage; assess sensitivity to a fourfold budget increase.",
        "families": ["parity"],
        "seeds": PLAN["formal_seeds"],
        "libraries": PLAN["libraries"],
        "backends": PLAN["backends"],
        "evaluations_per_stage": 1024,
        "other_settings": "Unchanged frozen primary protocol; same seeds, not an additional independent seed sample.",
        "sources": sources(),
        "stopping_rule": "Complete all 120 paired-condition runs once, irrespective of outcomes. Keep primary results unchanged."
    }
    # 在第一个延长预算候选生成之前写入方案；重复启动只接受完全相同的方案。
    path = HERE / "budget_sensitivity_protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == plan
    else:
        dump(path, plan)
    check_witnesses()
    directory = HERE / "artifacts/parity_budget1024"
    for family, library, seed in itertools.product(plan["families"], plan["libraries"], plan["seeds"]):
        # 继续交替后端的执行先后；两边各自初始化相同随机种子。
        backends = plan["backends"] if seed % 2 == 0 else list(reversed(plan["backends"]))
        for backend in backends:
            destination = directory / f"{family}_{library}_{seed}_{backend}"
            if (destination / "result.json").exists():
                continue
            record = run_one(backend, library, family, seed, plan["evaluations_per_stage"], destination)
            print(f"{library} {seed} {backend}: {[s['success'] for s in record['stages']]}", flush=True)
    result = summarize(directory)
    assert result["run_count"] == 120 and result["paired_stages"] == 180
    assert not result["backend_pair_mismatches"]
    print("Completed 120 runs across 30 paired seeds; 180 matched stages.", flush=True)


if __name__ == "__main__":
    main()
