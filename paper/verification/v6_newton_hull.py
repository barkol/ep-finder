"""V6: the hypothesis of Proposition 2, beta'_j = o(|b|^{(j-p+m)/m}) for p-m<j<p, says that the point of
B = sum_k beta_k lam^{p-k} at k = p-j lies strictly below the hull edge from (0, log|b|) to (m, log|d|)
as b -> 0 (d fixed): the edge height at k is log(|b|^{1-k/m}|d|^{k/m}) and (j-p+m)/m = 1-k/m."""
import sympy as sp
j, p, m, b, d = sp.symbols('j p m b d', positive=True)
k = p - j
edge = (1 - k/m)*sp.log(b) + (k/m)*sp.log(d)
print("exponent (j-p+m)/m - (1-k/m) =", sp.simplify((j-p+m)/m - (1-k/m)))
print("edge height - log(|b|^{(j-p+m)/m} |d|^{k/m}) =", sp.simplify(sp.expand_log(edge - sp.log(b**((j-p+m)/m)*d**(k/m)), force=True)))
