import numpy as np
# Two-grid LFA factor, x-semicoarsening, omega=1/2 Jacobi, nu pre-smoothings.
# rho(theta) = |Shat(theta)^nu (1-beta) + Shat(theta*)^nu beta|, beta = |P1|^2 l1 / (|P1|^2 l1 + |P2|^2 l2)
def tg_factor(eps, calib, nu=2, n=600):
    txs = np.linspace(1e-5, np.pi/2, n)
    tys = np.linspace(0.0, np.pi, n)
    best = 0.0
    for tx in txs:
        sx = np.sin(tx/2)**2; cx = np.cos(tx/2)**2
        if calib == 1:
            p1 = np.cos(tx/2); p2 = 1j*np.sin(tx/2)      # piecewise constant
        else:
            p1 = cx; p2 = sx                              # linear (caliber-2)
        a1 = abs(p1)**2; a2 = abs(p2)**2
        for ty in tys:
            sy = np.sin(ty/2)**2
            l1 = 4*sx + 4*eps*sy
            l2 = 4*cx + 4*eps*sy
            S1 = 1 - l1/(4*(1+eps))     # omega=1/2 Jacobi symbol at theta
            S2 = 1 - l2/(4*(1+eps))     # at theta*
            Lc = a1*l1 + a2*l2
            if Lc <= 0: continue
            beta = a1*l1/Lc
            rho = abs(S1**nu*(1-beta) + S2**nu*beta)
            if rho > best: best = rho
    return best

print("eps      cal1(nu1) cal2(nu1) cal1(nu2) cal2(nu2) cal2(nu3)")
for eps in [1.0, 1e-1, 1e-2, 1e-4]:
    r = [tg_factor(eps,1,1), tg_factor(eps,2,1), tg_factor(eps,1,2), tg_factor(eps,2,2), tg_factor(eps,2,3)]
    print(f"{eps:<8.0e} " + " ".join(f"{x:8.4f}" for x in r))
