# Individual-gates main-text figure caption (draft)

\textbf{Single-gate accuracy and path refinement.}
Long-range gates between sites 2 and 11 of $N=12$ MPS are compared with
dense application of the same gate. (a) Median infidelity for $R_{XX}$,
$R_{YY}$, and $R_{ZZ}$ at $\chi_{\max}=8$ and $n_{\mathrm{sub}}=1$; bands
span three gates and three states and are not confidence intervals. The
$\theta^2$ line is a guide, not a fit. The identity control is reported
separately because $\theta=0$ cannot lie on the logarithmic axis; TEBD+SWAP
still executes the routing sequence in this diagnostic. (b) CNOT infidelity
versus $\chi_{\max}$ with an effectively zero SVD threshold and
$n_{\mathrm{sub}}=1$ for TDVP. Symbols show three-state medians and bands
their full ranges. Variational MPO is the locally converged alternating-sweep
endpoint fit; all 87 controls converged without fallback. Markers in (a,b)
are offset where needed to expose overlapping medians. Curves and bands
remain at the stated coordinates.
The curves include the $\chi_{\max}=16$ endpoints,
which lie below the displayed range and therefore continue out of frame.
(c,d) Fixed-cap $R_{XX}$ infidelity and convergence at
$\theta/(2\pi)=10^{-2}$; (e,f) the corresponding CNOT results, all at
$\chi_{\max}=8$. Curves and horizontal lines show three-state medians,
with full-range bands. Panels (d,f) show
$D(n)=\min_\phi\|\Psi_n-e^{i\phi}\Psi_{2n}\|_2$, where $n$ is the
number of substeps. Refinement stabilizes
the cap-constrained TDVP path without driving it to the exact gate endpoint.
The states are normalized to roundoff, so
$D(n)=\sqrt{2-2|\langle\Psi_n|\Psi_{2n}\rangle|}$; this is a
phase-aligned state-vector distance, not an infidelity.
