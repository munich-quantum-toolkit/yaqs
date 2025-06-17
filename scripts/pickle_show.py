# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import pickle

# Adjust the path to wherever your file is saved
pickle_filepath = (
    "/Users/maximilianfrohlich/Documents/GitHub/mqt-yaqs/scripts/lindblad_mpo_results.pkl"  # or an absolute path
)

# Load the file
with open(pickle_filepath, "rb") as f:
    data = pickle.load(f)

# Now you can access its contents
parameters = data["parameters"]
z_expectation_values_mpo = data["z_expectation_values_mpo"]
lindblad_mpo_results = data["result"]


# Optional: print a few values
times, max_bond_dims = data["result"]["global"]["max_bond_dim", ()]
for _t, _d in zip(times, max_bond_dims):
    pass
