"""Manage RedSpark model-weight assets: status, download and cleanup.

Usage:
    python tools/model_weights.py status
    python tools/model_weights.py status --verify
    python tools/model_weights.py download --target base
    python tools/model_weights.py download --target all
    python tools/model_weights.py clean --target all --dry-run
    python tools/model_weights.py clean --target all --lfs-orphans

Weight binaries never enter Git history, and removing the working-tree file is
not enough to reclaim the disk: Git LFS keeps its own copy under
``.git/lfs/objects/``, and that copy keeps the space allocated until it is
removed too. ``clean`` always reports those copies and ``--lfs-orphans``
removes the ones no commit and no index entry references.

Every mode exits non-zero when a check fails, matching tools/verify_refactor.py.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = PROJECT_ROOT / "model" / "redspark" / "model-weights"
LFS_OBJECTS_DIR = PROJECT_ROOT / ".git" / "lfs" / "objects"

# Any of these means "this directory holds real weights"; .gitignore excludes them.
WEIGHT_SUFFIXES = (".safetensors", ".pt", ".bin")
# Kept next to the weights so a bare checkout still documents provenance.
PROVENANCE_FILES = ("config.json", "tokenizer.json", "chat_template.jinja")


@dataclass(frozen=True)
class Target:
    name: str
    repo_id: str
    directory: Path
    weight_file: str
    sha256: str
    size: int
    purpose: str


TARGETS: dict[str, Target] = {
    "base": Target(
        name="base",
        repo_id="openbmb/MiniCPM5-2B-Base",
        directory=WEIGHTS_ROOT / "base",
        weight_file="model.safetensors",
        sha256="d80717e7b8eb21ef43070244ecebd85d6694e4a33602fdb817f366bdb04e1e5a",
        size=5033557128,
        purpose="训练起点（继续预训练 / SFT）",
    ),
    "reference": Target(
        name="reference",
        repo_id="openbmb/MiniCPM5-2B",
        directory=WEIGHTS_ROOT / "minicpm5-2b-final-reference",
        weight_file="model-00000-of-00001.safetensors",
        sha256="14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c",
        size=5033557096,
        purpose="上游最终后训练版，仅作参考基线与加载回退",
    ),
}


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.2f} TiB"


def selected_targets(name: str) -> list[Target]:
    if name == "all":
        return list(TARGETS.values())
    return [TARGETS[name]]


def weight_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in WEIGHT_SUFFIXES)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


# --- Git LFS local copies -----------------------------------------------------
# A file staged through the LFS filter is written to .git/lfs/objects/<xx>/<yy>/<oid>.
# Deleting the working-tree file leaves that object behind, so the space stays used.


def lfs_object_oid(path: Path) -> str | None:
    """Return the LFS oid for an object file, or None when the path is not one.

    The on-disk layout is ``.git/lfs/objects/<oid[:2]>/<oid[2:4]>/<oid>``: the
    file name is already the full 64-hex oid, the two parent directories are
    only a fan-out for the filesystem.
    """
    oid = path.name
    if len(oid) == 64 and all(c in "0123456789abcdef" for c in oid):
        return oid
    return None


def lfs_objects() -> dict[str, Path]:
    if not LFS_OBJECTS_DIR.is_dir():
        return {}
    found: dict[str, Path] = {}
    for path in LFS_OBJECTS_DIR.rglob("*"):
        if path.is_file() and (oid := lfs_object_oid(path)):
            found[oid] = path
    return found


def lfs_tracked_oids() -> set[str]:
    """Oids referenced by the current checkout, any branch, or the index."""
    oids: set[str] = set()
    for line in _git("lfs", "ls-files", "--all", "--long").splitlines():
        parts = line.split()
        if parts and len(parts[0]) == 64:
            oids.add(parts[0])
    return oids


def oid_in_history(oid: str) -> bool:
    """True when the oid appears in any commit or in the staged index.

    The LFS oid is the *content* hash written inside a pointer file, so it is
    not a Git object name: `git rev-list --objects` can never match it. Only a
    content search (`git log -S`, `git grep --cached`) finds it.
    """
    if _git("log", "--all", "--oneline", "-S", oid).strip():
        return True
    return bool(_git("grep", "--cached", "-l", "-e", oid).strip())


def orphan_lfs_objects() -> dict[str, Path]:
    """LFS objects no ref and no index entry points at: safe to remove."""
    tracked = lfs_tracked_oids()
    return {
        oid: path
        for oid, path in lfs_objects().items()
        if oid not in tracked and not oid_in_history(oid)
    }


# --- Commands -----------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    print("== RedSpark 模型权重 ==")
    print()
    for target in selected_targets(args.target):
        files = weight_files(target.directory)
        total = sum(p.stat().st_size for p in files)
        if not target.directory.is_dir():
            state = "目录缺失"
        elif not files:
            state = "无权重"
        else:
            state = f"{len(files)} 个权重文件, {human(total)}"
        print(f"{target.name:<26} [{state}]  {target.directory.relative_to(PROJECT_ROOT)}")
        print(f"    上游      {target.repo_id} / {target.weight_file}")
        print(f"    期望体积  {human(target.size)}")
        present = [f for f in PROVENANCE_FILES if (target.directory / f).is_file()]
        missing = [f for f in PROVENANCE_FILES if f not in present]
        print(f"    溯源文件  {', '.join(present) or '（无）'}" + (f"   缺失: {', '.join(missing)}" if missing else ""))
        print(f"    用途      {target.purpose}")
        if args.verify and files:
            for path in files:
                actual = sha256_file(path)
                mark = "OK" if actual == target.sha256 else "不匹配"
                print(f"    sha256    {actual}  [{mark}]")
        print()

    objects = lfs_objects()
    if objects:
        tracked = lfs_tracked_oids()
        orphans = orphan_lfs_objects()
        total = sum(p.stat().st_size for p in objects.values())
        orphan_total = sum(p.stat().st_size for p in orphans.values())
        print("== Git LFS 本地副本 ==")
        print(f"    .git/lfs/objects: {len(objects)} 个对象, {human(total)}")
        print(f"    其中被跟踪:       {len(tracked)} 个")
        print(f"    孤儿对象:         {len(orphans)} 个, {human(orphan_total)}")
        for oid, path in sorted(orphans.items(), key=lambda kv: -kv[1].stat().st_size)[:10]:
            print(f"      {oid}  {human(path.stat().st_size)}")
        if len(orphans) > 10:
            print(f"      ... 另有 {len(orphans) - 10} 个")
        if orphans:
            print("    提示: 删工作区权重不会释放这些副本，用 clean --lfs-orphans 回收。")
        print()

    usage = shutil.disk_usage(PROJECT_ROOT)
    print("== 磁盘 ==")
    print(f"    可用 {human(usage.free)} / 总计 {human(usage.total)}")
    print("    说明: 部分环境（含 Windows 沙箱卷）的可用空间计数不反映删除，")
    print("          判断实际释放量请用 du -sh，而不是 df。")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("需要 huggingface_hub（随 transformers 安装）", file=sys.stderr)
        return 1

    failed = False
    for target in selected_targets(args.target):
        print(f"== 拉取 {target.name}: {target.repo_id} -> {target.directory.relative_to(PROJECT_ROOT)}")
        if args.dry_run:
            print(f"   [dry-run] snapshot_download(repo_id='{target.repo_id}', local_dir='{target.directory}')")
            print()
            continue
        target.directory.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=target.repo_id, local_dir=target.directory)

        path = target.directory / target.weight_file
        if not path.is_file():
            print(f"   下载后未找到 {target.weight_file}", file=sys.stderr)
            failed = True
        else:
            actual = sha256_file(path)
            if actual == target.sha256:
                print(f"   sha256 校验通过  {actual}")
            else:
                print(f"   sha256 不匹配\n     期望 {target.sha256}\n     实际 {actual}", file=sys.stderr)
                failed = True
        print()
    return 1 if failed else 0


def cmd_clean(args: argparse.Namespace) -> int:
    freed = 0
    print("== 清理权重文件 ==")
    for target in selected_targets(args.target):
        files = weight_files(target.directory)
        if not files:
            print(f"{target.name:<26} 无权重文件，跳过")
            continue
        for path in files:
            size = path.stat().st_size
            freed += size
            if args.dry_run:
                print(f"{target.name:<26} [dry-run] 将删除 {path.relative_to(PROJECT_ROOT)}  {human(size)}")
            else:
                path.unlink()
                print(f"{target.name:<26} 已删除 {path.relative_to(PROJECT_ROOT)}  {human(size)}")
        kept = [f for f in PROVENANCE_FILES if (target.directory / f).is_file()]
        if kept:
            print(f"{'':<26} 保留 {', '.join(kept)}")
    print()

    orphans = orphan_lfs_objects()
    print("== Git LFS 孤儿对象 ==")
    if not orphans:
        print("    无")
    else:
        total = sum(p.stat().st_size for p in orphans.values())
        for oid, path in sorted(orphans.items(), key=lambda kv: -kv[1].stat().st_size):
            size = path.stat().st_size
            if args.lfs_orphans:
                if not args.dry_run:
                    path.unlink()
                freed += size
                note = "  [dry-run]" if args.dry_run else "  已删除"
                print(f"    {oid}  {human(size)}{note}")
            else:
                print(f"    {oid}  {human(size)}")
        print(f"    合计 {len(orphans)} 个, {human(total)}")
        if not args.lfs_orphans:
            print("    未删除：加 --lfs-orphans 才会移除（它们不被任何提交或索引引用）。")
    print()
    print(f"本次{'预计' if args.dry_run else '实际'}释放 {human(freed)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    common = {"choices": (*TARGETS, "all")}

    status = sub.add_parser("status", help="报告权重与 LFS 副本状态")
    status.add_argument("--target", default="all", **common)
    status.add_argument("--verify", action="store_true", help="对权重文件做 sha256 全量校验（较慢）")
    status.set_defaults(func=cmd_status)

    download = sub.add_parser("download", help="拉取官方权重并校验 sha256")
    download.add_argument("--target", default="all", **common)
    download.add_argument("--dry-run", action="store_true", help="只打印将要执行的操作")
    download.set_defaults(func=cmd_download)

    clean = sub.add_parser("clean", help="删除权重文件，并可回收 LFS 孤儿副本")
    clean.add_argument("--target", default="all", **common)
    clean.add_argument("--dry-run", action="store_true", help="只列出将要删除的内容")
    clean.add_argument(
        "--lfs-orphans",
        action="store_true",
        help="同时删除 .git/lfs/objects 中无引用的副本（否则只报告）",
    )
    clean.set_defaults(func=cmd_clean)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
