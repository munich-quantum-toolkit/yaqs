
import numpy as np
from scipy.linalg import expm

def sanity_check():
    # Construct MPO tensors as in verify_process_tensor.py
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Zero = np.zeros((2, 2), dtype=complex)

    # w0: (1, 5, 2, 2) -> (left, right, out, in)
    w0 = np.zeros((1, 5, 2, 2), dtype=complex)
    w0[0, 0] = I
    w0[0, 1] = X
    w0[0, 2] = Y
    w0[0, 3] = Z
    w0[0, 4] = Zero

    # w1: (5, 1, 2, 2) -> (left, right, out, in)
    w1 = np.zeros((5, 1, 2, 2), dtype=complex)
    w1[0, 0] = Zero
    w1[1, 0] = X
    w1[2, 0] = Y
    w1[3, 0] = Z
    w1[4, 0] = I

    # Transpose to YAQS expected shape (out, in, left, right)
    # Original: (left, right, out, in)
    # Target: (out, in, left, right)
    w0 = w0.transpose(2, 3, 0, 1)
    w1 = w1.transpose(2, 3, 0, 1)

    print(f"w0 shape: {w0.shape}")
    print(f"w1 shape: {w1.shape}")

    # Contract to full Hamiltonian
    # w0(a, b, l, m) w1(c, d, m, r) -> H(ac, bd)
    # outL=a, inL=b, leftL=l, rightL=m
    # outR=c, inR=d, leftR=m, rightR=r
    # Contract over m (index 3 of w0, index 2 of w1)
    
    # H_tensor[a, b, c, d] (outL, inL, outR, inR)
    # Actually we want Matrix H[ac, bd]
    # Connect rightL=m with leftR=m
    
    H_tensor = np.tensordot(w0, w1, axes=([3], [2]))
    # Result shape: (a, b, l, c, d, r) -> (2, 2, 1, 2, 2, 1)
    # Remove l, r (dim 1)
    H_tensor = np.squeeze(H_tensor)
    # Shape (a, b, c, d) -> (outL, inL, outR, inR)
    
    # We want matrix H acting on Vector v[L, R]
    # H(outL, outR; inL, inR)
    # Reorder to (outL, outR, inL, inR)
    H_tensor = H_tensor.transpose(0, 2, 1, 3)
    
    H_mat = H_tensor.reshape(4, 4)
    
    print("\nHamiltonian Matrix:")
    print(np.round(H_mat.real, 2))

    # Check evolution of |01> (Index 1)
    # |00>, |01>, |10>, |11>
    v_In = np.zeros(4, dtype=complex)
    v_In[1] = 1.0 # |01>
    
    dt = 0.5
    U = expm(-1j * H_mat * dt)
    
    v_Out = U @ v_In
    
    print(f"\nTime t={dt}")
    print(f"Initial State |01>: {v_In}")
    print(f"Final State: {np.round(v_Out, 3)}")
    
    prob_01 = np.abs(v_Out[1])**2
    prob_10 = np.abs(v_Out[2])**2
    
    print(f"Prob |01>: {prob_01:.4f}")
    print(f"Prob |10>: {prob_10:.4f}")

    if np.abs(prob_01 - 1.0) < 1e-5:
        print("RESULT: Identity Behavior (No Interaction)")
    else:
        print("RESULT: Interaction Occurred")

if __name__ == "__main__":
    sanity_check()
