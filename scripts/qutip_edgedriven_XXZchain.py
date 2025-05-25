import qutip as qt
import numpy as np
import matplotlib.pyplot as plt


T = 80
dt = 0.01


L = 8      # Number of spins
delta = 1.0     # Anisotropy Δ
epsilon =   40        # coupling strength (ε)          


# Operators
sp = qt.sigmap()
sm = qt.sigmam()
sz = qt.sigmaz()
I = qt.qeye(2)

# Time vector
t = np.arange(0, T + dt, dt)

# Define Pauli matrices
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

# Build Hamiltonian
H = 0
for j in range(L - 1):
    sp_sm = qt.tensor([sp if n == j else sm if n == j + 1 else I for n in range(L)])
    sm_sp = qt.tensor([sm if n == j else sp if n == j + 1 else I for n in range(L)])
    sz_sz = qt.tensor([sz if n == j or n == j + 1 else I for n in range(L)])
    H += 2 * (sp_sm + sm_sp) + delta * sz_sz

c_ops = []

# L1 = sqrt(ε) * σ_1^+
c_ops.append(np.sqrt(epsilon) * qt.tensor([sp] + [I] * (L - 1)))

# L2 = sqrt(ε) * σ_n^-
c_ops.append(np.sqrt(epsilon) * qt.tensor([I] * (L - 1) + [sm]))


# Initial state
psi0 = qt.rand_ket(2**L) 
psi0.dims = [[2]*L, [1]]  # Matches the L-qubit tensor structure

# Define measurement operators
sz_list = [qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

# Exact Lindblad solution
result_lindblad = qt.mesolve(H, psi0, t, c_ops, sz_list, progress_bar=True)



# Exact steady-state ⟨σ^z_j⟩ from Prosen's 2011 PRL (strong driving, Δ = 1)
for i in range(1,L+1):
    print(i-1)
sz_exact = [np.cos(np.pi * (j-1) / (L - 1)) for j in range(1,L+1)]
print("Exact steady-state ⟨σ^z_j⟩:")
for i, sz_val in enumerate(sz_exact):
    print(f"⟨Z_{i}⟩ = {sz_val:.4f}")


final_expvals = [result_lindblad.expect[i][-1] for i in range(L)]
diffs = [final_expvals[i] - sz_exact[i] for i in range(L)]
print("Final difference (numerical - analytical):")
for i, d in enumerate(diffs):
    print(f"Site {i}: {d:.4e}")


plt.figure(figsize=(10, 6))
for i in range(L):
    plt.plot(t, result_lindblad.expect[i], label=f'⟨Z_{i}⟩ (numerical)')
    plt.axhline(y=sz_exact[i], linestyle='--', color='gray',
                label='⟨Z_j⟩ (exact steady state)' if i == 0 else None)
plt.title('Lindblad Dynamics vs Exact Steady State (Prosen 2011)')
plt.xlabel('Time')
plt.ylabel('Expectation Values ⟨σ^z_j⟩')
plt.legend()
plt.grid()
plt.show()
