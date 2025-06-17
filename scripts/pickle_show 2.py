import pickle

# Adjust the path to wherever your file is saved
pickle_filepath = "/Users/maximilianfrohlich/Documents/GitHub/mqt-yaqs/scripts/lindblad_mpo_results.pkl"  # or an absolute path

# Load the file
with open(pickle_filepath, "rb") as f:
    data = pickle.load(f)

# Now you can access its contents
parameters = data["parameters"]
z_expectation_values_mpo = data["z_expectation_values_mpo"]
lindblad_mpo_results = data["result"]


print(data.keys())  # Shows: dict_keys(['parameters', 'result', 'z_expectation_values_mpo'])

# Optional: print a few values
print(parameters)
print(z_expectation_values_mpo.shape)
print(list(lindblad_mpo_results.keys()))
print(data["result"]["global"].keys())
times, max_bond_dims = data["result"]["global"][('max_bond_dim', ())]
print("Max bond dimensions at each step:")
for t, d in zip(times, max_bond_dims):
    print(f"  t = {t:.2f}: {d}")


