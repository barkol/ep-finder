"""V5: four-mode network of equation (21) of the paper from the path expansion. Links (drive l -> k is entry M[k,l]):
3->1 (1), 1->0 (1), 3->2 (1), 2->0 (e^{i phi}), 1->2 (w), swept 0->3 (c). Zero detuning, uniform loss dropped.
Expect A = lam^4, B = (1+e^{i phi}) lam + w e^{i phi}."""
import sympy as sp
lam, c, w, p = sp.symbols('lambda c w phi')
M0 = sp.zeros(4, 4)
M0[1, 3] = 1; M0[0, 1] = 1; M0[2, 3] = 1; M0[0, 2] = sp.exp(sp.I*p); M0[2, 1] = w
X = lam*sp.eye(4) - M0
A = sp.expand(X.det()); B = sp.expand(X.adjugate()[0, 3])
E = sp.zeros(4, 4); E[3, 0] = 1
chi = sp.expand((lam*sp.eye(4) - M0 - c*E).det())
print("A =", A, "; B =", sp.collect(B, lam))
print("chi - [lam^4 - c((1+e^{i phi})lam + w e^{i phi})] =", sp.simplify(chi - (lam**4 - c*((1+sp.exp(sp.I*p))*lam + w*sp.exp(sp.I*p)))))
