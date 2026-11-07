"""只读核验匿名补充实验发布包；不导入 ATF，不训练，不改写任何证据。"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import struct
import sys
import zipfile
from collections import Counter, defaultdict


# 本文件只使用标准库。禁止调用旧实验的审计脚本，因为旧脚本会写报告，
# 其中一个还会用当前环境覆盖历史 environment.json。这里独立读取冻结证据。
class VerificationError(Exception):
    """可安全展示给读者的核验失败；消息不包含匹配到的个人信息。"""


def require(condition, message):
    # 不使用 assert，避免 python -O 意外关闭发布核验。
    if not condition:
        raise VerificationError(message)


def close(actual, expected, message):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected))
            and math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-14), message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def checked_path(root, relative):
    # 清单必须使用相对 POSIX 路径；中文文件名合法。拒绝越界与符号链接，
    # 防止所谓只读审计被清单诱导到发布包之外读取用户文件。
    require(isinstance(relative, str) and "\\" not in relative, "清单路径必须使用正斜杠")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and ".." not in path.parts and ":" not in relative,
            "清单包含越界路径")
    candidate = root.joinpath(*path.parts)
    require(candidate.resolve().is_relative_to(root), "清单路径解析后越界")
    require(all(not parent.is_symlink() for parent in [candidate, *candidate.parents]
                if parent.is_relative_to(root)), "清单包含符号链接")
    require(candidate.is_file(), f"发布文件缺失：{relative}")
    return candidate


def verify_manifest(root):
    manifest_name = "release_manifest.json"
    manifest = read_json(root / manifest_name)
    require(manifest.get("version") == 1, "发布清单版本不支持")
    entries = manifest.get("files")
    require(isinstance(entries, dict) and entries, "发布清单没有文件")
    require(manifest_name not in entries, "发布清单不应循环记录自身哈希")
    history = defaultdict(set)
    for relative, entry in entries.items():
        path = checked_path(root, relative)
        require(isinstance(entry, dict) and re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")),
                f"发布哈希格式错误：{relative}")
        require(sha256(path) == entry["sha256"], f"发布内容哈希不一致：{relative}")
        require(isinstance(entry.get("transformation"), str), f"缺少转换说明：{relative}")
        original = entry.get("source_sha256")
        require(original is None or re.fullmatch(r"[0-9a-f]{64}", original),
                f"来源哈希格式错误：{relative}")
        # 原协议记录的是历史字节，不是匿名副本的字节。二者分开校验，
        # 不要求匿名化后的文件与旧哈希相同，更不修改旧协议伪造一致性。
        history[entry["sha256"]].add(relative)
        if original is not None:
            history[original].add(relative)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == set(entries) | {manifest_name}, "发布目录存在未列入清单的文件")
    return entries, history


def verify_anonymity(root, entries):
    # 公开核验器只检查通用路径与邮箱模式，不内置任何可还原的作者身份。
    # 特定身份的检查由发布者在包外进行。失败仅报告文件名，不打印匹配内容。
    # 通用用户目录规则分段写出，避免规则的源码文本被自身识别为路径；
    # Python 合并后的正则与原规则完全相同，不跳过核验器，也不减少覆盖范围。
    patterns = [re.compile(r"[A-Za-z]:[\\/]+(?:Users|Research)[\\/]+", re.I),
                re.compile(r"/" r"Users/" r"[^/\s]+/", re.I),
                re.compile(r"[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}")]
    text_suffixes = {".py", ".txt", ".md", ".json", ".jsonl", ".csv", ".log",
                     ".html", ".svg", ".gexf", ".toml", ".ini", ".cfg", ".yml", ".yaml"}
    failed = []
    for relative in [*entries, "release_manifest.json"]:
        if Path(relative).suffix.lower() not in text_suffixes:
            continue
        value = (root / relative).read_text(encoding="utf-8-sig")
        if any(pattern.search(value) for pattern in patterns):
            failed.append(relative)
    require(not failed, "匿名扫描发现需处理文件：" + ", ".join(failed))


def verify_historical_hash(history, expected, label):
    require(expected in history, f"历史来源未包含在发布清单映射中：{label}")


def verify_type_search(base, history):
    folder = base / "type_search"
    protocol = read_json(folder / "protocol_frozen.json")
    summary = read_json(folder / "formal/summary.json")
    audit = read_json(folder / "formal/audit.json")
    rows = [read_json(p) for p in sorted((folder / "formal").glob("*/*/result.json"))]
    expected = {(seed, mode) for seed in protocol["formal_seeds"] for mode in protocol["modes"]}
    require(len(rows) == len(expected) == summary["n_attempted"] == 90, "类型实验运行数量不一致")
    require({(r["run_seed"], r["type_mode"]) for r in rows} == expected, "类型实验种子／组别缺失或重复")
    require(all(r["status"] == "completed" for r in rows), "类型实验存在未完成运行")
    csv_rows = read_csv(folder / "formal/per_seed.csv")
    csv_map = {(int(r["run_seed"]), r["type_mode"]): r for r in csv_rows}
    require(len(csv_rows) == len(csv_map) == 90 and set(csv_map) == expected, "类型逐种子 CSV 不完整")
    fixed_success = {"fine": 29, "coarse": 15, "flat": 1}
    for mode, expected_success in fixed_success.items():
        group = [r for r in rows if r["type_mode"] == mode]
        arm = summary["arms"][mode]
        successes = sum(bool(r["success"]) for r in group)
        require(len(group) == arm["n_completed"] == 30 and arm["n_failed"] == 0,
                f"类型实验组规模不一致：{mode}")
        require(successes == arm["successes"] == expected_success, f"类型实验成功数不一致：{mode}")
        require(sum(r["initial_accuracy"] == 1.0 for r in group) == arm["initial_successes"],
                f"类型实验初始成功数不一致：{mode}")
        for metric in ("loss", "accuracy", "loss_calls", "accuracy_calls", "callback_evaluations", "unique_candidates"):
            close(statistics.mean(r[metric] for r in group), arm[metric]["mean"], f"类型实验均值不一致：{mode}/{metric}")
        for result in group:
            seed = result["run_seed"]
            require(result["success"] == (result["accuracy"] == 1.0), "类型成功口径发生变化")
            events = read_jsonl(folder / f"formal/{mode}/{seed}/evaluation_events.jsonl")
            require(len(events) == result["callback_evaluations"], "类型评价回调总数不一致")
            require(sum(e["callback_kind"] == "loss" for e in events) == result["loss_calls"], "类型 loss 回调数不一致")
            require(sum(e["callback_kind"] == "accuracy" for e in events) == result["accuracy_calls"], "类型 accuracy 回调数不一致")
            require(len({e["fingerprint"] for e in events}) == result["unique_candidates"], "类型候选数不一致")
            observed = {}
            for event in events:
                value = (event["loss"], event["accuracy"])
                require(observed.setdefault(event["fingerprint"], value) == value, "同一冻结图出现不同评价")
            for metric in ("loss", "accuracy", "loss_calls", "accuracy_calls", "callback_evaluations", "unique_candidates"):
                close(csv_map[(seed, mode)][metric], result[metric], "类型 CSV 与逐运行记录不一致")
    catalog = read_jsonl(folder / "formal/candidate_catalog.jsonl")
    require(len(catalog) == audit["catalog_records"] == 1007, "类型候选目录记录数不一致")
    final_keys = [(r["run_seed"], r["type_mode"]) for r in catalog if r["kind"] == "final"]
    require(len(final_keys) == audit["catalog_final_records"] == 90 and set(final_keys) == expected,
            "类型候选目录最终图不完整")
    for name in ("runner_sha256", "source_sha256", "pilot_summary_sha256", "validation_sha256"):
        verify_historical_hash(history, protocol[name], f"type_search/{name}")
    for mode, digest in protocol["method_files_sha256"].items():
        verify_historical_hash(history, digest, f"type_search/methods/{mode}")
    for source, digest in read_json(folder / "environment.json")["core_source_hashes_collected_after_run"].items():
        verify_historical_hash(history, digest, "runtime_v0/" + source.replace("\\", "/"))
    return catalog


def verify_encoding(base, history, catalog):
    folder = base / "state_encoding/formal"
    summary = read_json(folder / "summary.json")
    analysis = read_json(folder / "analysis.json")
    rows = read_csv(folder / "candidate_results.csv")
    require(len(rows) == len(catalog) == summary["candidate_records"] == 1007, "状态编码候选数不一致")
    key = lambda r: (int(r["run_seed"]), r["type_mode"], r["kind"], r["fingerprint"])
    require(Counter(map(key, rows)) == Counter(map(key, catalog)), "状态编码使用了不同候选")
    steps = sum(int(r["checked_steps"]) for r in rows)
    require(steps == summary["checked_environment_steps"] == analysis["paired_environment_step_comparisons"] == 70670,
            "状态编码步数不一致")
    require(sum(int(r["initial_states"]) for r in rows) == summary["checked_initial_conditions"] == 20140,
            "状态编码初态数不一致")
    require(all(r["matched"] == "True" for r in rows) and summary["all_matched"] and analysis["all_matches"],
            "状态编码存在不匹配记录")
    require(analysis["total_native_graph_executions_in_semantic_audit"] == 2 * steps,
            "状态编码配对步数与执行次数混淆")
    finals = [r for r in rows if r["kind"] == "final"]
    require(len(finals) == summary["final_graphs"] == 90, "状态编码最终图数不一致")
    close(statistics.mean(int(r["extra_state_edges"]) for r in finals),
          analysis["mean_extra_state_edges_final_graphs"], "状态编码额外边均值不一致")
    pairs = defaultdict(lambda: defaultdict(dict))
    for row in read_csv(folder / "timing_raw.csv"):
        bucket = pairs[(row["run_seed"], row["type_mode"])][row["mode"]]
        require(row["repeat"] not in bucket and float(row["seconds"]) > 0, "编码计时重复或非正值")
        bucket[row["repeat"]] = float(row["seconds"])
    require(len(pairs) == summary["timed_final_graphs"] == 90, "编码计时模型数不一致")
    ratios = []
    for pair in pairs.values():
        require(set(pair) == {"implicit", "explicit"}, "编码计时缺少一侧")
        require(len(pair["implicit"]) == len(pair["explicit"]) == summary["timing_repeats"] == 5,
                "编码计时重复数不一致")
        require(set(pair["implicit"]) == set(pair["explicit"]), "编码计时重复编号不配对")
        ratios.append(statistics.median(pair["explicit"].values()) / statistics.median(pair["implicit"].values()))
    close(statistics.median(ratios), analysis["median_paired_timing_ratio"], "编码计时比率不一致")
    close(statistics.median(ratios), summary["median_explicit_over_implicit_ratio"], "编码汇总计时比率不一致")
    notes = []
    for protocol_name in ("protocol.json", "timing_protocol.json"):
        protocol = read_json(folder / protocol_name)
        verify_historical_hash(history, protocol["catalog_sha256"], "state_encoding/冻结候选目录")
        for source, digest in protocol["source_sha256"].items():
            if (digest not in history and protocol_name == "protocol.json"
                    and source.replace("\\", "/").endswith("/run_encoding.py")
                    and digest == "dd914d6fd3633a9e84cfd03215cb0a81b40a1f3ab0dace53c1fc7802cb05a372"):
                # 原报告明确披露：语义核验后仅增加了 timing-only 入口；早期
                # 脚本字节没有保存。保留此历史缺口，不能用新源码冒充旧快照。
                notes.append("state_encoding：早期语义阶段脚本快照未随包保存；计时阶段脚本及冻结运行时来源已核对。")
            else:
                verify_historical_hash(history, digest, "state_encoding/" + Path(source.replace("\\", "/")).name)
    return notes


def verify_scaling(base, history):
    folder = base / "scaling"
    protocol = read_json(folder / "frozen_protocol.json")
    completion = read_json(folder / "completion_manifest.json")
    formal = read_json(folder / "formal_results.json")
    cases = {(r["width"], r["seed"]) for r in formal}
    require(len(formal) == len(cases) == completion["formal_graphs"] == 30, "规模实验图数不一致")
    require(cases == {tuple(c) for c in protocol["formal_cases"]}, "规模实验案例不符合冻结协议")
    require(not cases.intersection(map(tuple, protocol["pilot_cases_excluded"])), "规模实验混入 pilot")
    require(completion["complete"] and completion["all_validated"], "规模实验未完整通过")
    verify_historical_hash(history, completion["protocol_sha256"], "scaling/冻结协议")
    require(protocol["code_sha256"] == completion["code_sha256_after"], "规模实验运行前后源码不同")
    for source, digest in protocol["code_sha256"].items():
        verify_historical_hash(history, digest, "scaling/" + source.replace("\\", "/"))
    for source, digest in completion["outputs"].items():
        verify_historical_hash(history, digest, "scaling/" + source)
    per_graph = read_csv(folder / "per_graph.csv")
    table = {(int(r["width"]), int(r["seed"])): r for r in per_graph}
    require(len(per_graph) == len(table) == 30 and set(table) == cases, "规模逐图 CSV 不完整")
    raw = read_csv(folder / "timing_samples.csv")
    require(len(raw) == 1260, "规模原始计时不是 1260 条")
    raw_map = {(int(r["width"]), int(r["seed"]), r["metric"], int(r["repetition"])): float(r["milliseconds"]) for r in raw}
    require(len(raw_map) == len(raw), "规模原始计时键重复")
    expected_raw = set()
    for row in formal:
        width, seed = row["width"], row["seed"]
        require(row["operation_nodes"] == 8 * width and row["total_nodes"] == 8 * width + 2
                and row["edges"] == 17 * width, "规模图节点／边计数不一致")
        require(row["node_function_calls_per_inference"] == row["operation_nodes"] * 32, "规模 batch 调用数不一致")
        require(row["max_abs_serial_parallel"] == 0 and row["max_abs_serial_reference"] < 1e-12
                and row["max_abs_parallel_reference"] < 1e-12, "规模运行数值核验不通过")
        for metric, values in row["samples_ms"].items():
            require(len(values) == 7 and all(math.isfinite(v) and v > 0 for v in values), "规模计时重复数或数值错误")
            close(statistics.median(values), row[metric], "规模逐图中位数错误")
            close(table[(width, seed)][metric], row[metric], "规模逐图 CSV 中位数错误")
            for repetition, value in enumerate(values, 1):
                raw_key = (width, seed, metric, repetition)
                expected_raw.add(raw_key)
                require(raw_key in raw_map, "规模原始计时缺项")
                close(raw_map[raw_key], value, "规模原始计时与 JSON 不同")
        close(row["serial_ms"] / row["parallel4_ms"], row["parallel_speedup"], "规模加速比错误")
    require(set(raw_map) == expected_raw, "规模原始计时多出未知项")
    aggregates = read_csv(folder / "summary.csv")
    require(len(aggregates) == len(protocol["widths"]) == 6, "规模汇总尺寸数不一致")
    for summary in aggregates:
        group = [r for r in formal if r["width"] == int(summary["width"])]
        require(len(group) == int(summary["graph_seeds"]) == 5, "规模每尺寸种子数不一致")
        for metric in group[0]:
            if metric != "samples_ms" and (metric.endswith("_ms") or metric.endswith("_bytes") or metric == "parallel_speedup"):
                values = [r[metric] for r in group]
                for suffix, expected in (("median", statistics.median(values)), ("min", min(values)), ("max", max(values))):
                    close(summary[f"{metric}_{suffix}"], expected, "规模汇总统计不一致")


def npz_shapes(path):
    # 只读 NPY 头部，不加载对象，不反序列化 pickle，也不依赖 NumPy。
    # zipfile.open 会检查解压 CRC；这里读完每项以发现压缩成员损坏。
    result = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            require(member.filename.endswith(".npy") and member.file_size < 128 * 1024 * 1024,
                    "轨迹归档包含异常成员")
            data = archive.read(member)
            require(data[:6] == b"\x93NUMPY", "轨迹数组头部不合法")
            version = tuple(data[6:8])
            require(version in {(1, 0), (2, 0), (3, 0)}, "轨迹数组格式版本不支持")
            size = 2 if version == (1, 0) else 4
            length = struct.unpack("<H" if size == 2 else "<I", data[8:8 + size])[0]
            header = ast.literal_eval(data[8 + size:8 + size + length].decode("utf-8" if version == (3, 0) else "latin1"))
            require(isinstance(header.get("descr"), str) and "O" not in header["descr"], "轨迹数组禁止对象类型")
            name = member.filename[:-4]
            require(name not in result, "轨迹数组名称重复")
            result[name] = tuple(header["shape"])
    return result


def verify_action(base, history):
    folder = base / "action_example"
    protocol = read_json(folder / "protocol_formal.json")
    result = read_json(folder / "results_formal.json")
    boundary = read_json(folder / "boundary_checks_formal.json")
    require(result["passed"] and result["protected_sources_unchanged"], "动作核验未完整通过")
    require(result["source_sha256"] == protocol["source_sha256"], "动作核验运行前后来源不同")
    for source, digest in protocol["source_sha256"].items():
        verify_historical_hash(history, digest, "action_example/" + Path(source.replace("\\", "/")).name)
    require(len(result["tasks"]) == 1 and result["tasks"][0]["task"] == "bird", "动作核验任务范围变化")
    task = result["tasks"][0]
    steps = sum(r["steps"] for r in task["outcomes"])
    require(steps == task["steps_checked"] == result["total_steps"] == 20480, "动作核验步数不一致")
    require(len(task["outcomes"]) == task["episodes"] == 20 and task["all_matched"], "动作核验回合数或一致性错误")
    require(all(error == 0 for error in task["max_absolute_error_features_scores_state"]), "动作核验存在数值误差")
    require(task["steps_also_checked_against_preexisting_saved_trace"] == 2048, "动作旧轨迹核验步数不一致")
    require(len(boundary) == result["boundary_probes"] == result["boundary_reordered_probes"] == 49,
            "动作边界核验数量不一致")
    require(len({(row["case"], row["u"]) for row in boundary}) == 49, "动作边界探针重复")
    require(all(all(error == 0 for error in row["max_errors_features_scores_state"]) for row in boundary),
            "动作边界探针存在数值误差")
    graph = read_json(folder / "native_graph_bird.json")
    require(len(graph["nodes"]) == task["native_adapter_nodes"] == 316
            and len(graph["edges"]) == task["native_adapter_edges"] == 565, "动作适配图规模不一致")
    require(task["frozen_graph_nodes"] == 314 and task["frozen_graph_edges"] == 411, "动作冻结图规模不一致")
    shapes = npz_shapes(folder / "native_trace_bird_formal.npz")
    expected = {"episode": (steps,), "step": (steps,), "u": (steps,), "action": (steps,),
                "old_state": (steps, 4), "new_state": (steps, 4), "features": (steps, task["features"]),
                "scores": (steps, task["actions"]), "permits": (steps, task["actions"])}
    require(shapes == expected, "动作归档轨迹数组形状与正式结果不同")


def main():
    parser = argparse.ArgumentParser(description="只读核验匿名发布包，结果仅打印到终端。")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="发布包根目录，默认是本脚本所在目录")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        entries, history = verify_manifest(root)
        verify_anonymity(root, entries)
        base = root if (root / "type_search").is_dir() else root / "experiments/rebuttal_20261106"
        catalog = verify_type_search(base, history)
        notes = verify_encoding(base, history, catalog)
        verify_scaling(base, history)
        verify_action(base, history)
    except VerificationError as error:
        print("FAIL: " + str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        # 异常正文可能含绝对路径或文件内容，因此只展示异常类型。
        print("FAIL: 发布材料无法解析（" + type(error).__name__ + "）", file=sys.stderr)
        return 1
    print(f"PASS: {len(entries)} 个发布文件哈希及匿名扫描通过。")
    print("PASS: 类型 90 次／29,15,1 成功／1007 条候选；编码 70670 对步骤。")
    print("PASS: 规模 30 图／1260 条原始计时；动作 20480 步／49+49 边界探针。")
    print("PASS: 已随包保存的来源与原协议哈希映射一致；未执行训练或写入文件。")
    for note in notes:
        print("NOTE: " + note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
