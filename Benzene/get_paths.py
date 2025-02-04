import os

def get_deepest_dirs(root_dir):
    deepest_dirs = []
    max_depth = 0

    for dirpath, dirnames, _ in os.walk(root_dir):
        depth = dirpath.count(os.sep)
        if depth > max_depth:
            max_depth = depth
            deepest_dirs = [dirpath]
        elif depth == max_depth:
            deepest_dirs.append(dirpath)

    return deepest_dirs

root_directory = "/hpc/home/baum612/TESTS/DUCC Hamiltonians/Benzene"  # Change this to your directory path
deepest_subdirs = get_deepest_dirs(root_directory)

for subdir in deepest_subdirs:
    print(subdir)

