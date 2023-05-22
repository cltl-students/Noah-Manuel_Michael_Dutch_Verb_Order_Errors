import subprocess

while True:
    exit_code = subprocess.call(["python3", "l_discodop_get_most_likely_tree.py"])
    if exit_code == 0:
        break
