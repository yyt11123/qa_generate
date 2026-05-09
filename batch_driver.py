"""
批量处理 md -> 镜像目录结构下的独立 JSON -> 最终 CSV

核心机制:
- 输出为**镜像目录**: 每个源 md 对应一个独立的 <stem>_qa.json, 按源目录结构放进 --output-dir
- 并发: ThreadPoolExecutor, 通过 --workers 调节
- 断点续传: state.json 记录 completed_files / next_qa_id
- 异常隔离: 任意文件失败不影响其它 worker, 失败记入 error_list.txt
- 全局 qa_id 递增: 由主线程顺序分配, 避免并发条件竞争
- 最终导出: 递归扫描镜像目录 -> 扁平 6 列 CSV(utf-8-sig)

用法:
    # 批量生成(默认输出到 ./qa_output/)
    python batch_driver.py run --root 保司文件2.0 --workers 8

    # 指定输出根
    python batch_driver.py run --root 保司文件2.0 --output-dir dist/qa --workers 8

    # 跑完顺带导出 CSV
    python batch_driver.py run --root 保司文件2.0 --workers 8 --export-csv qa_pairs.csv

    # 仅从已有的 qa_output 导出 CSV
    python batch_driver.py export --output-dir qa_output --output qa_pairs.csv

    # 重跑: 把已有的 state/output-dir/error_log/rejected_log/csv 备份为 .bak.<时间戳>
    python batch_driver.py run --root 保司文件2.0 --reset --workers 8
"""

import argparse
import json
import shutil
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

import qa_generate as qg

load_dotenv()

DEFAULT_WORKERS = 8
DEFAULT_OUTPUT_DIR = "qa_output"
CSV_COLUMNS = ["分类", "question", "answers", "document", "page", "text/img/table"]


# ============== state.json 管理 ==============
class StateStore:
    """
    state.json 结构:
    {
      "completed_files": ["保司文件2.0/...rel/path.md", ...],
      "next_qa_id": 123,
      "updated_at": "2026-05-08T12:34:56"
    }
    """

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self._lock = threading.Lock()
        if state_path.exists():
            data = json.loads(state_path.read_text(encoding="utf-8"))
            self.completed = set(data.get("completed_files", []))
            self.next_qa_id = int(data.get("next_qa_id", 1))
        else:
            self.completed = set()
            self.next_qa_id = 1

    def is_done(self, rel_path: str) -> bool:
        return rel_path in self.completed

    def allocate_ids(self, n: int) -> int:
        with self._lock:
            start = self.next_qa_id
            self.next_qa_id += n
            return start

    def mark_done(self, rel_path: str):
        with self._lock:
            self.completed.add(rel_path)
            self._flush_locked()

    def _flush_locked(self):
        tmp = self.state_path.with_suffix(".json.tmp")
        payload = {
            "completed_files": sorted(self.completed),
            "next_qa_id": self.next_qa_id,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.state_path)


# ============== 错误日志 ==============
class ErrorLogger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def log(self, rel_path: str, err: Exception):
        tb = traceback.format_exception_only(type(err), err)[-1].strip()
        line = f"{datetime.now().isoformat(timespec='seconds')}\t{rel_path}\t{tb}\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)


# ============== 被拒 QA 调试日志 ==============
class RejectedLogger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def log_many(self, rel_path: str, company: str, doc_name: str, rejections: list):
        if not rejections:
            return
        now = datetime.now().isoformat(timespec="seconds")
        lines = []
        for rej in rejections:
            lines.append(
                json.dumps(
                    {
                        "ts": now,
                        "rel_path": rel_path,
                        "company": company,
                        "doc_name": doc_name,
                        "reason": rej.get("reason"),
                        "question": rej.get("question"),
                        "answer": rej.get("answer"),
                        "evidence": rej.get("evidence"),
                        "category": rej.get("category"),
                        "prompt_type": rej.get("prompt_type"),
                    },
                    ensure_ascii=False,
                )
            )
        blob = "\n".join(lines) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(blob)


# ============== 镜像路径 & 写独立 JSON ==============
def mirror_output_path(
    md_path: Path, root: Path, output_dir: Path
) -> Path:
    """
    源 .md 的镜像输出路径:
      源:  <root>/a/b/foo.md
      出:  <output_dir>/a/b/foo_qa.json
    md 不在 root 子树里时(比如 symlink 外跳), 按文件名落在 output_dir 根.
    """
    try:
        rel = md_path.resolve().relative_to(root)
    except ValueError:
        rel = Path(md_path.name)
    stem = rel.stem
    return output_dir.joinpath(*rel.parent.parts) / f"{stem}_qa.json"


def write_mirror_json(out_path: Path, payload: dict):
    """原子写: <name>.json.tmp -> rename -> <name>.json"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(out_path)


# ============== 单文件处理 ==============
def _process_one(
    md_path: Path,
    rel_path: str,
    root: Path,
    output_dir: Path,
    client: OpenAI,
    base_dir: str,
    state: StateStore,
    err_logger: ErrorLogger,
    rej_logger: RejectedLogger,
) -> dict:
    """
    在 worker 线程里被调用. 任何异常都被捕获, 通过返回值汇报.
    """
    try:
        result = qg.generate_qa_for_md(md_path, client, base_dir=base_dir)
    except Exception as e:
        err_logger.log(rel_path, e)
        return {
            "rel": rel_path, "status": "error",
            "written": 0, "rejected": 0, "error": str(e),
        }

    if result["status"] == "skipped_short":
        state.mark_done(rel_path)
        return {
            "rel": rel_path, "status": "short",
            "written": 0, "rejected": 0, "error": None,
        }

    records = result["records"]
    rejected = result["rejected"]
    doc_info = result.get("doc_info", {})
    company = doc_info.get("company", "")
    doc_name = doc_info.get("doc_name", "")

    if rejected:
        rej_logger.log_many(rel_path, company, doc_name, rejected)

    if not records:
        err_logger.log(
            rel_path,
            RuntimeError(f"empty_after_validation rejected={len(rejected)}"),
        )
        return {
            "rel": rel_path, "status": "empty",
            "written": 0, "rejected": len(rejected),
            "error": "empty_after_validation",
        }

    start_id = state.allocate_ids(len(records))
    enriched = [
        {"qa_id": f"qa_{start_id + i:05d}", **rec} for i, rec in enumerate(records)
    ]

    out_path = mirror_output_path(md_path, root, output_dir)
    payload = {
        "source_rel_path": rel_path,
        "company": company,
        "doc_name": doc_name,
        "md_length": doc_info.get("md_length"),
        "mode": doc_info.get("mode"),
        "n_chunks": doc_info.get("n_chunks"),
        "n_qa_per_chunk": doc_info.get("n_qa_per_chunk"),
        "n_qa_requested": doc_info.get("n_qa_requested"),
        "n_duplicates_removed": doc_info.get("n_duplicates_removed"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "qa_count": len(enriched),
        "qa_pairs": enriched,
    }
    try:
        write_mirror_json(out_path, payload)
    except Exception as e:
        err_logger.log(rel_path, e)
        return {
            "rel": rel_path, "status": "error",
            "written": 0, "rejected": len(rejected), "error": str(e),
        }

    state.mark_done(rel_path)
    return {
        "rel": rel_path, "status": "ok",
        "written": len(records), "rejected": len(rejected), "error": None,
        "out_path": str(out_path),
        "n_chunks": doc_info.get("n_chunks", 1),
    }


# ============== --reset 备份 ==============
def backup_existing(paths, tag: str = None) -> list:
    """
    把存在的文件 / 目录重命名为 <name>.bak.<tag>. 不存在的跳过.
    目录也可以备份(rename 级别, 原地移走).
    """
    tag = tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    backups = []
    for p in paths:
        p = Path(p)
        if p.exists():
            if p.is_dir():
                bak = p.parent / f"{p.name}.bak.{tag}"
            else:
                bak = p.with_suffix(p.suffix + f".bak.{tag}")
            p.rename(bak)
            backups.append((p, bak))
    return backups


# ============== 命令: run ==============
def cmd_run(args):
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[错误] root 不存在: {root}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    state_path = Path(args.state)
    error_path = Path(args.error_log)
    rejected_path = Path(args.rejected_log)

    if args.reset:
        reset_targets = [state_path, output_dir, error_path, rejected_path]
        if args.export_csv:
            reset_targets.append(Path(args.export_csv))
        backups = backup_existing(reset_targets)
        if backups:
            print("[reset] 已备份旧文件/目录:")
            for src, bak in backups:
                print(f"  {src} -> {bak}")
        else:
            print("[reset] 没有旧文件/目录需要备份")
        print()

    output_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(root.rglob("*.md"))
    if not md_files:
        print(f"[警告] {root} 下没有 .md 文件")
        return

    state = StateStore(state_path)
    err_logger = ErrorLogger(error_path)
    rej_logger = RejectedLogger(rejected_path)

    base_dir_name = root.name

    pending = []
    for p in md_files:
        try:
            rel = p.resolve().relative_to(root.parent).as_posix()
        except ValueError:
            rel = p.as_posix()
        if state.is_done(rel):
            continue
        pending.append((p, rel))

    print(f"共发现 {len(md_files)} 个 md, 已完成 {len(state.completed)}, "
          f"本次待处理 {len(pending)} 个")
    print(f"并发度: {args.workers}")
    print(f"输出根目录: {output_dir}")
    print(f"起始 qa_id: qa_{state.next_qa_id:05d}")
    print()

    if not pending:
        print("没有待处理文件, 直接结束。")
        return

    client = OpenAI(api_key=qg.API_KEY, base_url=qg.BASE_URL)

    stats = {"ok": 0, "short": 0, "empty": 0, "error": 0, "written_total": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _process_one,
                md_path, rel_path, root, output_dir,
                client, base_dir_name,
                state, err_logger, rej_logger,
            ): rel_path
            for md_path, rel_path in pending
        }
        with tqdm(total=len(futures), ncols=100, desc="批量生成") as pbar:
            for fut in as_completed(futures):
                rel = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    err_logger.log(rel, e)
                    stats["error"] += 1
                    pbar.update(1)
                    continue
                stats[res["status"]] += 1
                stats["written_total"] += res["written"]
                pbar.set_postfix_str(
                    f"ok={stats['ok']} err={stats['error']} "
                    f"short={stats['short']} empty={stats['empty']}"
                )
                pbar.update(1)

    print()
    print("=== 批量处理完成 ===")
    print(f"成功:       {stats['ok']} 个文件")
    print(f"过短跳过:   {stats['short']} 个文件")
    print(f"全部被过滤: {stats['empty']} 个文件 (详见 {error_path})")
    print(f"异常失败:   {stats['error']} 个文件 (详见 {error_path})")
    print(f"累计写入:   {stats['written_total']} 条 QA")
    print(f"下次起始 qa_id: qa_{state.next_qa_id:05d}")
    print(f"输出根目录:    {output_dir}")
    print(f"拒样详情:      {rejected_path}")

    if args.export_csv:
        print()
        export_csv_from_dir(output_dir, Path(args.export_csv))


# ============== 命令: export ==============
def iter_qa_files(output_dir: Path):
    """递归扫出所有 *_qa.json"""
    yield from sorted(output_dir.rglob("*_qa.json"))


def export_csv_from_dir(output_dir: Path, csv_path: Path):
    """镜像目录 -> 扁平 6 列 CSV(utf-8-sig)"""
    import pandas as pd

    if not output_dir.exists():
        print(f"[错误] 输出目录不存在: {output_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    n_files = 0
    for qa_file in iter_qa_files(output_dir):
        n_files += 1
        try:
            payload = json.loads(qa_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[警告] 解析失败, 跳过: {qa_file}: {e}")
            continue
        qa_pairs = payload.get("qa_pairs", [])
        for rec in qa_pairs:
            rows.append(
                {
                    "分类": rec.get("category", ""),
                    "question": rec.get("question", ""),
                    "answers": rec.get("answers", ""),
                    "document": rec.get("document", ""),
                    "page": rec.get("page", "N/A"),
                    "text/img/table": rec.get("support_type", "text"),
                }
            )

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[export] 扫描 {n_files} 个 _qa.json, 合计 {len(df)} 行 -> "
          f"{csv_path} (utf-8-sig)")


def cmd_export(args):
    export_csv_from_dir(Path(args.output_dir), Path(args.output))


# ============== 入口 ==============
def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="批量处理 md, 每个 md 写成独立的 _qa.json")
    pr.add_argument("--root", required=True, help="md 根目录, 例: 保司文件2.0")
    pr.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"镜像输出根目录, 默认 {DEFAULT_OUTPUT_DIR}",
    )
    pr.add_argument("--state", default="state.json", help="断点续传状态文件")
    pr.add_argument("--error-log", default="error_list.txt", help="异常文件日志")
    pr.add_argument(
        "--rejected-log",
        default="rejected_log.jsonl",
        help="被 validate_qa 拒掉的样本调试日志",
    )
    pr.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并发线程数")
    pr.add_argument(
        "--export-csv", default=None,
        help="跑完后同时从镜像目录导出 CSV, 填目标路径则启用",
    )
    pr.add_argument(
        "--reset",
        action="store_true",
        help="开跑前把 state/output-dir/error_log/rejected_log/CSV 备份为 .bak.<时间戳>",
    )
    pr.set_defaults(func=cmd_run)

    pe = sub.add_parser("export", help="从镜像目录导出最终 6 列 CSV")
    pe.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"镜像根目录, 默认 {DEFAULT_OUTPUT_DIR}",
    )
    pe.add_argument("--output", required=True, help="目标 csv 路径")
    pe.set_defaults(func=cmd_export)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
