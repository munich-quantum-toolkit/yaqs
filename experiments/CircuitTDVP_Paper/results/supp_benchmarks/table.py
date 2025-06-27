# import os
# import glob
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import re

# def collect_bond_data(pickle_dir: str) -> pd.DataFrame:
#     """
#     Scan pickle files under pickle_dir, extract TEBD/TDVP bond stats,
#     and return a DataFrame with columns:
#     Name, TEBD Max bond, TEBD Total bond, TDVP Max bond, TDVP Total bond.
#     """
#     records = []
#     # Recursively find all .pickle files
#     for path in glob.glob(os.path.join(pickle_dir, '**', '*.pickle'), recursive=True):
#         try:
#             with open(path, 'rb') as f:
#                 data = pickle.load(f)
#         except Exception:
#             continue
#         if 'results' not in data:
#             continue

#         results = data['results']
#         # Assume one entry per method
#         if results is not None:
#             tebd_entry = results.get('TEBD', [([], None)])[0]
#             tdvp_entry = results.get('TDVP', [([], None)])[0]
#         else:
#             continue

#         bonds_tebd, exp_tebd = tebd_entry
#         bonds_tdvp, exp_tdvp = tdvp_entry

#         tebd_max   = float(np.nanmax(bonds_tebd))   if bonds_tebd else np.nan
#         tebd_total = float(np.nansum(bonds_tebd))   if bonds_tebd else np.nan
#         tdvp_max   = float(np.nanmax(bonds_tdvp))   if bonds_tdvp else np.nan
#         tdvp_total = float(np.nansum(bonds_tdvp))   if bonds_tdvp else np.nan

#         # Extract number of qubits from filename
#         name = os.path.splitext(os.path.basename(path))[0]
#         match = re.search(r'_(\d+)_bonds', name)
#         n_qubits = int(match.group(1)) if match else np.nan
    
#         if exp_tebd is not None:
#             error = np.abs(exp_tebd - exp_tdvp)
#         else:
#             error = np.nan
#         records.append({
#             'Name': name,
#             'Qubits': n_qubits,
#             'Error': error,
#             'TEBD Max bond': tebd_max,
#             'TEBD Total bond': tebd_total,
#             'TDVP Max bond': tdvp_max,
#             'TDVP Total bond': tdvp_total,
#             'Delta max': tebd_max - tdvp_max,
#             'Delta total': tebd_total - tdvp_total
#         })

#     return pd.DataFrame.from_records(records)


# def plot_graph_scaling(df: pd.DataFrame, output_prefix: str = 'graph_scaling'):
#     # Sort by qubit count
#     df = df.sort_values('Qubits')
#     print(df)
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#     # (a) Total bond dimension
#     axes[0].plot(df['Qubits'], df['TEBD Total bond'], 'o-', label='TEBD')
#     axes[0].plot(df['Qubits'], df['TDVP Total bond'], 's--', label='TDVP')
#     axes[0].set_xlabel('Number of qubits')
#     axes[0].set_ylabel('Total bond dimension')
#     axes[0].set_title('(a) Total bond vs qubit count')
#     axes[0].legend()
#     axes[0].grid(True)

#     # (b) Observable error (|exp_TDVP - exp_TEBD|)
#     axes[1].plot(df['Qubits'], df['Error'], 'd-', color='purple')
#     axes[1].set_xlabel('Number of qubits')
#     axes[1].set_ylabel('Absolute error in ⟨X_i X_j⟩')
#     axes[1].set_yscale('log')
#     axes[1].set_title('(b) TDVP–TEBD correlator error')
#     axes[1].grid(True)

#     plt.tight_layout()
#     plt.savefig(f'{output_prefix}.pdf')
#     plt.savefig(f'{output_prefix}.png', dpi=300)
#     plt.show()

# if __name__ == '__main__':
#     PICKLE_DIR = '.'  # Replace with your directory path
#     df = collect_bond_data(PICKLE_DIR)
#     df.to_csv('bond_summary.csv', index=False)
#     plot_graph_scaling(df)

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def collect_bond_data(pickle_dir: str) -> pd.DataFrame:
    records = []
    for path in glob.glob(os.path.join(pickle_dir, '**', '*.pickle'), recursive=True):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception:
            continue
        if 'results' not in data:
            continue

        results = data['results']
        tebd_entry = results.get('TEBD', [([], None)])[0]
        tdvp_entry = results.get('TDVP', [([], None)])[0]

        bonds_tebd, exp_tebd, fidelity = tebd_entry
        bonds_tdvp, exp_tdvp = tdvp_entry

        tebd_max   = float(np.nanmax(bonds_tebd))   if bonds_tebd else np.nan
        tebd_total = float(np.nansum(bonds_tebd))   if bonds_tebd else np.nan
        tdvp_max   = float(np.nanmax(bonds_tdvp))   if bonds_tdvp else np.nan
        tdvp_total = float(np.nansum(bonds_tdvp))   if bonds_tdvp else np.nan

        # Extract rows and cols from filename, e.g., cluster2d_3x5.pickle
        name = os.path.splitext(os.path.basename(path))[0]
        match = re.search(r'_(\d+)x(\d+)', name)
        if match:
            rows = int(match.group(1))
            cols = int(match.group(2))
            n_qubits = rows * cols
            label = f"{rows}×{cols}"
        else:
            rows = cols = n_qubits = np.nan
            label = name

        if exp_tebd is not None and exp_tdvp is not None:
            error = np.abs(exp_tebd - exp_tdvp)
        else:
            error = np.nan

        records.append({
            'Name': name,
            'Grid size': label,
            'Qubits': n_qubits,
            'Rows': rows,
            'Cols': cols,
            'Fidelity': fidelity,
            'TEBD Exp Val': exp_tebd,
            'TEBD Max bond': tebd_max,
            'TEBD Total bond': tebd_total,
            'TDVP Exp Val': exp_tdvp,
            'TDVP Max bond': tdvp_max,
            'TDVP Total bond': tdvp_total,
            'Delta max': tebd_max - tdvp_max,
            'Delta total': tebd_total - tdvp_total
        })

    return pd.DataFrame.from_records(records)

def plot_cluster2d_scaling(df: pd.DataFrame, output_prefix: str = 'cluster2d_scaling'):
    df = df.sort_values('Qubits')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) Total bond dimension
    axes[0].plot(df['Grid size'], df['TEBD Total bond'], 'o-', label='TEBD')
    axes[0].plot(df['Grid size'], df['TDVP Total bond'], 's--', label='TDVP')
    axes[0].set_xlabel('Grid size')
    axes[0].set_ylabel('Total bond dimension')
    axes[0].set_title('(a) Total bond vs 2D grid shape')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].tick_params(axis='x', rotation=45)

    # Only show ticks for square grids on left plot
    square_grids = df[df['Rows'] == df['Cols']]['Grid size'].values
    xticks_all = df['Grid size'].values
    axes[0].set_xticks(xticks_all)
    axes[0].set_xticklabels([
        label if label in square_grids else "" for label in xticks_all
    ])

    # (b) Observable error (|exp_TDVP - exp_TEBD|)
    axes[1].plot(df['Grid size'], df['Fidelity'], 'o-')
    axes[1].set_xlabel('Grid size')
    axes[1].set_ylabel('Absolute error in ⟨Xᵢ Xⱼ⟩')
    axes[1].set_yscale('log')
    axes[1].set_title('(b) TDVP–TEBD correlator error')
    axes[1].grid(True)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}.pdf')
    plt.savefig(f'{output_prefix}.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    PICKLE_DIR = '.'  # Replace with your path
    df = collect_bond_data(PICKLE_DIR)
    df.to_csv('cluster2d_summary.csv', index=False)
    plot_cluster2d_scaling(df)
