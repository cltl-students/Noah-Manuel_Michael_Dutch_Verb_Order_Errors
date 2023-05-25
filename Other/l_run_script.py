import subprocess

while True:
    exit_code = subprocess.call(["python3", "tbd_l_get_most_likely_tree_discodop.py"])
    if exit_code == 0:
        break
