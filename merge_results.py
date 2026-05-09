"""
把 batch_driver.py 生成的镜像目录下的所有 _qa.json 合并成一个嵌套结构的大 JSON.

输入:
    <qa_dir>/
        保司文件2.0/
            万通/
                xxx/
                    内部繳費指引_qa.json
                    yyy_qa.json
            保诚/
                zzz_qa.json

输出(默认):
    {
      "_meta": {"merged_at": "...", "total_qa": 123, "total_files": 12},
      "保司文件2.0": {
        "万通": {
          "xxx": {
            "内部繳費指引": [ {...qa...}, ... ],
            "yyy": [ ... ]
          }
        },
        "保诚": {
          "zzz": [ ... ]
        }
      }
    }

用法:
    python merge_results.py --input qa_output --output merged.json

    # 平铺(所有 QA 扁平展开到一个列表, 不要嵌套层级)
    python merge_results.py --input qa_output --output merged_flat.json --flat
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def iter_qa_files(root: Path):
    yield from sorted(root.rglob("*_qa.json"))


def strip_suffix(name: str, suffix: str = "_qa") -> str:
    """把文件名 stem 里的 _qa 后缀砍掉, 让键名干净"""
    return name[: -len(suffix)] if name.endswith(suffix) else name


def insert_nested(tree: dict, rel_parts: list, leaf_key: str, qa_pairs: list):
    """
    按 rel_parts 一层层深入 tree, 到叶子把 qa_pairs 挂到 leaf_key 下.
    如果同一层既有子目录又有同名文件导致冲突, 用 "__files__" 兜底, 见下方处理.
    """
    node = tree
    for i, part in enumerate(rel_parts):
        existing = node.get(part)
        if existing is None:
            node[part] = {}
            node = node[part]
        elif isinstance(existing, dict):
            node = existing
        else:
            # 前面把同名挂成了叶子(list), 现在要下钻, 退到 __files__ 里
            node[part] = {"__files__": existing}
            node = node[part]

    # leaf_key 可能已经存在(同目录下重名), 退到 __duplicates__
    if leaf_key in node:
        node.setdefault("__duplicates__", []).append(
            {leaf_key: node[leaf_key]}
        )
        node["__duplicates__"].append({leaf_key: qa_pairs})
    else:
        node[leaf_key] = qa_pairs


def build_nested(qa_dir: Path) -> tuple:
    """返回 (nested_tree, total_qa, total_files)"""
    tree = {}
    total_qa = 0
    total_files = 0
    for qa_file in iter_qa_files(qa_dir):
        total_files += 1
        try:
            payload = json.loads(qa_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[警告] 解析失败, 跳过: {qa_file}: {e}", file=sys.stderr)
            continue
        qa_pairs = payload.get("qa_pairs", [])
        total_qa += len(qa_pairs)

        rel = qa_file.relative_to(qa_dir)
        parts = list(rel.parent.parts)  # 目录层级
        leaf_key = strip_suffix(rel.stem)  # 去掉 _qa 后缀
        insert_nested(tree, parts, leaf_key, qa_pairs)

    return tree, total_qa, total_files


def build_flat(qa_dir: Path) -> tuple:
    """所有 QA 平铺到一个列表"""
    flat = []
    total_files = 0
    for qa_file in iter_qa_files(qa_dir):
        total_files += 1
        try:
            payload = json.loads(qa_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[警告] 解析失败, 跳过: {qa_file}: {e}", file=sys.stderr)
            continue
        flat.extend(payload.get("qa_pairs", []))
    return flat, len(flat), total_files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="qa_output", help="镜像目录根")
    ap.add_argument("--output", required=True, help="合并后的 JSON 输出路径")
    ap.add_argument(
        "--flat",
        action="store_true",
        help="平铺模式: 所有 QA 合成一个列表, 不保留目录嵌套",
    )
    args = ap.parse_args()

    qa_dir = Path(args.input).resolve()
    if not qa_dir.exists():
        print(f"[错误] 输入目录不存在: {qa_dir}", file=sys.stderr)
        sys.exit(1)

    if args.flat:
        data, total_qa, total_files = build_flat(qa_dir)
        output = {
            "_meta": {
                "merged_at": datetime.now().isoformat(timespec="seconds"),
                "total_files": total_files,
                "total_qa": total_qa,
                "mode": "flat",
                "source_dir": str(qa_dir),
            },
            "qa_pairs": data,
        }
    else:
        tree, total_qa, total_files = build_nested(qa_dir)
        output = {
            "_meta": {
                "merged_at": datetime.now().isoformat(timespec="seconds"),
                "total_files": total_files,
                "total_qa": total_qa,
                "mode": "nested",
                "source_dir": str(qa_dir),
            },
            "tree": tree,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[merge] {total_files} 个 _qa.json, 合计 {total_qa} 条 QA -> "
        f"{out_path} ({'flat' if args.flat else 'nested'})"
    )


if __name__ == "__main__":
    main()
