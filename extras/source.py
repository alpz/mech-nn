import os
import subprocess

def create_log_dir(root):
    for i in range(1000):
        name = str(i)
        path = os.path.join(root, name)

        if not os.path.isdir(path):
            os.makedirs(path)
            return path,i 

    raise ValueError("Dir not found. Limit exceeded",i)


def write_source_files(summ_dir):
    #stdout_log = os.path.join(summ_dir, 'stdout.txt')
    #redirect_stdout(stdout_log)
    if not os.path.isdir(summ_dir):
        os.makedirs(summ_dir)

    # write git diff, commit hash, redirect stdout
    #diff = os.path.join(summ_dir, 'git.diff')
    diff = os.path.join(summ_dir, "git.diff")
    diff2 = os.path.join(summ_dir, "git2.diff")

    if not os.path.isfile(diff):
        with open(diff, 'w') as fd:
            #subprocess.call(['git diff'], stdout=fd, stderr=fd, shell=True)
            subprocess.call(["git diff -- '***.py'"], stdout=fd, stderr=fd, shell=True)

        with open(diff2, 'w') as fd2:
            #subprocess.call(['git diff'], stdout=fd, stderr=fd, shell=True)
            subprocess.call(["git diff -- '*.py'"], stdout=fd2, stderr=fd2, shell=True)

    # write commit hash
    commit = os.path.join(summ_dir, 'commit.txt')
    if not os.path.isfile(commit):
        with open(commit, 'w') as fd:
            subprocess.call(['git rev-parse HEAD'], stdout=fd, stderr=fd, shell=True)