import os.path
from os.path import dirname, join

from imodelsx import submit_utils

repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    "seed": [1, 2],
    "save_dir": [join(repo_dir, "results")],
    "use_cache": [1],  # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys
params_coupled_dict = {
    ("model_name", "alpha"): [
        ("ridge", 0.1),
        ("ridge", 1),
    ],
    ("model_name", "max_depth"): [("decision_tree", i) for i in range(2, 4)],
}

# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, "experiments", "01_train_model.py"),
    actually_run=True,
)
