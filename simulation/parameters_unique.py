# -----------------------------------------------------------------------------
# An evolutionary perspective on the prospect theory
# Copyright 2020 Nicolas P. Rougier & Aurélien Nioche
# Released under the BSD two-clauses license
# -----------------------------------------------------------------------------
import json
import time
import subprocess

# Default parameters
_data = {
    "seed"           : 123,
    "n_trial"        : 100,
    "n_agent"        : 1000,
    "n_epoch"        : 1000,
    "n_lottery"      : 1000,
    "mixture_rate"   : 0.25,
    "selection_rate" : 0.20,
    "mutation_rate"  : 0.02,
    "pmin"           : 1.00 - 0.75,
    "pmax"           : 1.00 + 0.75,
    "vmin"           : 0.00 - 0.80,
    "vmax"           : 0.00 + 0.80,
    "gridsize"       : 100,
    "timestamp"      : "",
    "git_branch"     : "",
    "git_hash"       : "",
}

def get_git_revision_hash():
    """ Get current git hash """
    answer = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    return answer.decode("utf8").strip("\n")

def get_git_revision_branch():
    """ Get current git branch """
    answer = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    return answer.decode("utf8").strip("\n")

def default():
    """ Get default parameters """
    _data["timestamp"] = time.ctime()
    _data["git_branch"] = get_git_revision_branch()
    _data["git_hash"] = get_git_revision_hash()
    return _data

def save(filename, data=None):
    """ Save parameters into a json file """
    if data is None:
       data = { name : eval(name) for name in _data.keys()
                if name not in ["timestamp", "git_branch", "git_hash"] }
    data["timestamp"] = time.ctime()
    data["git_branch"] = get_git_revision_branch()
    data["git_hash"] = get_git_revision_hash()
    with open(filename, "w") as outfile:
        json.dump(data, outfile)

def load(filename):
    """ Load parameters from a json file """
    with open(filename) as infile:
        data = json.load(infile)
    return data

def dump(data):
    for key, value in data.items():
        print(f"{key:15s} : {value}")

# -----------------------------------------------------------------------------
if __name__  == "__main__":
    save("test.txt", _data)
    data = load("test.txt")
    dump(data)
    locals().update(data)
    save("test.txt")
    

