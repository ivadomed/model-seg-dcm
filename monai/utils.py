import subprocess


def get_git_branch_and_commit(dataset_path=None):
    """
    :return: git branch and commit ID, with trailing '*' if modified
    Taken from: https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/utils/sys.py#L476 
    and https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/utils/sys.py#L461
    """

    # branch info
    b = subprocess.Popen(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=dataset_path)
    b_output, _ = b.communicate()
    b_status = b.returncode

    if b_status == 0:
        branch = b_output.decode().strip()
    else:
        branch = "!?!"

    # commit info
    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=dataset_path)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        commit = output.decode().strip()
    else:
        commit = "?!?"

    p = subprocess.Popen(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=dataset_path)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        unclean = True
        for line in output.decode().strip().splitlines():
            line = line.rstrip()
            if line.startswith("??"):  # ignore ignored files, they can't hurt
                continue
            break
        else:
            unclean = False
        if unclean:
            commit += "*"

    return branch, commit
