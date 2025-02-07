import os
from collections import defaultdict

def group_dirs_by_depth(root_dir=None):
    if root_dir is None:
        root_dir = os.getcwd()  # Use current directory

    depth_dict = defaultdict(list)

    for dirpath, _, _ in os.walk(root_dir):
        depth = dirpath.count(os.sep) - root_dir.count(os.sep)
        depth_dict[depth].append(dirpath)

    return dict(depth_dict)

dirs_by_depth = group_dirs_by_depth()

for depth, dirs in sorted(dirs_by_depth.items()):
    print(f"Depth {depth}:")
    for d in dirs:
        print(f"  {d}")

