import os
import shutil
from pathlib import Path

def backup_dirs_excluding_master(base_dir, backup_dir):
    base_dir = Path(base_dir)
    backup_dir = Path(backup_dir)

    mapping = {}

    if not base_dir.exists():
        raise ValueError(f"Base directory '{base_dir}' does not exist.")

    # Ensure backup root exists
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Go through each subdir in base
    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue

        backup_subdir = backup_dir / subdir.name

        # Register mapping regardless of whether it's already backed up
        mapping[str(subdir)] = str(backup_subdir)

        if backup_subdir.exists():
            print(f"Already backed up: {backup_subdir}")
            continue

        # Ignore function to skip -master dirs
        def ignore_master_dirs(dir_path, contents):
            return [item for item in contents
                    if (Path(dir_path) / item).is_dir() and item.startswith("master")]

        print(f"Backing up {subdir} -> {backup_subdir}")
        shutil.copytree(
            subdir,
            backup_subdir,
            ignore=ignore_master_dirs
        )

    return mapping

# Example usage
if __name__ == "__main__":
    base = "/net/scratch/zsarwar/exps"
    backup = "/net/projects/mmairegroup/zsarwar/backup"
    result = backup_dirs_excluding_master(base, backup)

    print("\nBackup mapping:")
    for src, dst in result.items():
        print(f"{src} -> {dst}")

"""

/net/scratch/zsarwar/exps/GPT_experts-16-topk-1-layers12-heads-12 -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-16-topk-1-layers12-heads-12
/net/scratch/zsarwar/exps/GPT_experts-8-topk-2-layers12-heads-12 -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-8-topk-2-layers12-heads-12
/net/scratch/zsarwar/exps/GPT_experts-32-topk-4-layers12-heads-12 -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-32-topk-4-layers12-heads-12
/net/scratch/zsarwar/exps/GPT_experts-32-topk-2-layers12-heads-12 -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-32-topk-2-layers12-heads-12
/net/scratch/zsarwar/exps/GPT_experts-8-topk-1-layers12-heads-12 -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-8-topk-1-layers12-heads-12
/net/scratch/zsarwar/exps/GPT_experts-16-topk-2-layers12-heads-12 -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-16-topk-2-layers12-heads-12
/net/scratch/zsarwar/exps/GPT_experts-4-topk-1-layers12-heads-12 -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-4-topk-1-layers12-heads-12
/net/scratch/zsarwar/exps/GPT_experts-7-topk-1-layers12-heads-12 -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-7-topk-1-layers12-heads-12
/net/scratch/zsarwar/exps/GPT_experts-27-topk-2-layers12-heads-12 -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-27-topk-2-layers12-heads-12
/net/scratch/zsarwar/exps/GPT_experts-3-topk-1-layers12-heads-12 -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-3-topk-1-layers12-heads-12
/net/scratch/zsarwar/exps/GPT_experts-1-topk-1-layers12-heads-12 -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-1-topk-1-layers12-heads-12
/net/scratch/zsarwar/exps/GPT_experts-16-topk-4-layers12-heads-12 -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-16-topk-4-layers12-heads-12

"""

"""
/net/scratch/zsarwar/exps/GPT_experts-1-topk-1-layers12-heads-12-lora -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-1-topk-1-layers12-heads-12-lora
/net/scratch/zsarwar/exps/GPT_experts-3-topk-1-layers12-heads-12-hetro -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-3-topk-1-layers12-heads-12-hetro
/net/scratch/zsarwar/exps/GPT_experts-8-topk-1-layers12-heads-12-base-moe -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-8-topk-1-layers12-heads-12-base-moe
/net/scratch/zsarwar/exps/GPT_experts-4-topk-1-layers12-heads-12-base-moe -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-4-topk-1-layers12-heads-12-base-moe
/net/scratch/zsarwar/exps/GPT_experts-1-topk-1-layers12-heads-12-base-moe -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-1-topk-1-layers12-heads-12-base-moe
/net/scratch/zsarwar/exps/GPT_experts-4-topk-1-layers12-heads-12-hetro -> /net/projects/mmairegroup/zsarwar/backup/GPT_experts-4-topk-1-layers12-heads-12-hetro

"""