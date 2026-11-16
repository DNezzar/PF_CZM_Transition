#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pf_czm_implicit.py — PF (AT2) + CZM bilinéaire énergétique, HHT-alpha implicite, tangente consistante
======================================================================================================
Même cas test que pf_czm_bleyer.py (dont ce script réutilise le maillage et les sorties ; garder les
deux fichiers dans le même dossier).

Une seule énergie, dont tout dérive :

  (psi+/psi- : --split oed [défaut, décomposition orthogonale], miehe [spectrale] ou amor)
  E(u, d, D) = int_Omega [ g(d) psi+(eps) + psi-(eps) ] + Gc/(2 l0) int_Omega [ d^2 + l0^2 |grad d|^2 ]
             + int_Gamma [ g(phi) (1-D) 1/2 K |q|^2 + 1/2 K <-Delta_n>^2 + int_0^D Yc ]

  g(x) = (1-x)^2 ;  phi = (d+ + d-)/2 ;  q = (Delta_t, <Delta_n>+) ;  lambda = |q|
  Yc(D) = 1/2 K [Do Dc / (Dc - D (Dc - Do))]^2     (redonne la loi bilinéaire ; int_0^1 Yc = Gc_i)

  Critère cohésif énergétique : g(phi) 1/2 K lambda^2 <= Yc(D)   <=>   D = D_bilin((1-phi) lambda)
  Traction      : t  = dE/dq   = g(phi) (1-D) K q   (+ contact K <Delta_n>-)
  Force motrice : dE/dphi = -2 (1-phi) (1-D) 1/2 K lambda^2     (énergie stockée seulement)

Résolution : minimisation alternée  [u (D condensé, Newton, tangente consistante symétrique)]  <->
             [d (problème quadratique borné d_n <= d <= 1, pas de champ d'histoire)].
Temps      : HHT-alpha,  M a_{n+1} + (1-alpha) F_{n+1} + alpha F_n = 0,
             beta = (1+alpha)^2/4, gamma = 1/2 + alpha, pas adaptatif (division si Newton échoue).

  python pf_czm_implicit.py --check                 # vérifie tangente et symétrie du couplage
  python pf_czm_implicit.py --l0 4e-4 --h 1e-4      # calcul
"""
import argparse, time
from dataclasses import dataclass, asdict
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pf_czm_bleyer import Params, Model, NG, AX, AY

try:                                   # solveur direct rapide si disponible
    import pypardiso
    def lsolve(A, b): return pypardiso.spsolve(A.tocsr(), b)
except Exception:
    def lsolve(A, b): return spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A").solve(b)


@dataclass
class ParamsImp(Params):
    model: str = "AT2"; coupling: str = "energetique"
    split: str = "oed"             # "oed" (orthogonale, He / Nguyen et al.), "miehe" (spectrale), "amor"
    alpha_hht: float = 0.05        # amortissement numérique, 0 <= alpha <= 1/3
    dt_fac: float = 1.0            # dt = dt_fac * h / c_R
    newton_tol: float = 1e-6; newton_max: int = 12
    stag_tol: float = 1e-3; stag_max: int = 20
    cR: float = 906.


def pattern(edofs, n):
    """Structure CSR fixe + table d'assemblage : data = bincount(inv, valeurs élémentaires)."""
    m = edofs.shape[1]
    key = (np.repeat(edofs, m, 1).astype(np.int64) * n + np.tile(edofs, (1, m))).ravel()
    uniq, inv = np.unique(key, return_inverse=True)
    indptr = np.r_[0, np.cumsum(np.bincount(uniq // n, minlength=n))]
    return (uniq % n).astype(np.int32), indptr.astype(np.int32), inv, uniq.size


class ModelImp(Model):
    def __init__(s, p):
        super().__init__(p); p = s.p
        ed = np.empty((s.ne, 8), int); ed[:, 0::2] = 2 * s.conn; ed[:, 1::2] = 2 * s.conn + 1
        s.pu = pattern(ed, s.ndof); s.pd = pattern(s.dconn, s.nd)
        NN = np.array([np.outer(NG[g], NG[g]) for g in range(4)])            # (4,4,4)
        s.NN = NN
        s.Mc = s.asm(s.pd, np.einsum('e,gij->eij', s.w, NN), s.nd)           # masse consistante de d
        s.Ad0 = (s.Kg + s.cm * s.Mc + sp.diags(s.d_unused.astype(float))).tocsr()
        s.Bv = []; s.Bd = []; s.Bs = []
        for g in range(4):                                                   # tr(eps), exx-eyy, gamma_xy
            bx = AX[g] * s.ihx[:, None]; by = AY[g] * s.ihy[:, None]
            bv = np.zeros((s.ne, 8)); bv[:, 0::2] = bx; bv[:, 1::2] = by
            bd = np.zeros((s.ne, 8)); bd[:, 0::2] = bx; bd[:, 1::2] = -by
            bs = np.zeros((s.ne, 8)); bs[:, 0::2] = by; bs[:, 1::2] = bx
            s.Bv.append(bv); s.Bd.append(bd); s.Bs.append(bs)
        s.ed = ed
        s.B3 = [np.stack([s.Bv[g], s.Bd[g], s.Bs[g]], axis=1) for g in range(4)]         # (ne, 3, 8)
        s.Bt = [np.ascontiguousarray(B.transpose(0, 2, 1)) for B in s.B3]
        s.D = s.rD.copy()                                                    # endommagement cohésif convergé
        s.bcmask = np.zeros(s.ndof, bool); s.bcmask[s.bc_dof] = True
        s.dt0 = p.dt_fac * p.h / p.cR

    @staticmethod
    def asm(pat, ke, n):
        ind, ptr, inv, nnz = pat
        return sp.csr_matrix((np.bincount(inv, ke.ravel(), nnz), ind, ptr), shape=(n, n))

    # ------------------------------------------------------------------------- interface
    def czm_state(s, u, d, Dn):
        t, b = s.ip_t, s.ip_b
        Dt = u[2 * t] - u[2 * b]; Dn_ = u[2 * t + 1] - u[2 * b + 1]; qn = np.maximum(Dn_, 0)
        lam = np.sqrt(Dt ** 2 + qn ** 2)
        om = 1 - 0.5 * (d[s.dmap[t]] + d[s.dmap[b]]); le = om * lam          # ouverture effective
        Dtr = np.clip(s.Dc * (le - s.Do) / (np.maximum(le, 1e-300) * (s.Dc - s.Do)), 0, 1)
        D = np.maximum(Dn, Dtr)
        return Dt, Dn_, qn, lam, om, le, D

    def czm_force(s, u, d, Dn, tangent=False):
        """Forces cohésives (D condensé) et, si demandé, tangente consistante."""
        t, b, w, K = s.ip_t, s.ip_b, s.ip_w, s.K
        Dt, Dn_, qn, lam, om, le, D = s.czm_state(u, d, Dn)
        kk = om ** 2 * (1 - D) * K
        tt = kk * Dt; tn = kk * qn + K * np.minimum(Dn_, 0)
        f = np.zeros(s.ndof)
        np.add.at(f, 2 * t, w * tt); np.add.at(f, 2 * b, -w * tt)
        np.add.at(f, 2 * t + 1, w * tn); np.add.at(f, 2 * b + 1, -w * tn)
        if not tangent: return f, D
        load = (D > Dn) & (D < 1) & (lam > 0)
        Dp = np.where(load, s.Dc * s.Do / ((s.Dc - s.Do) * np.maximum(le, 1e-300) ** 2), 0.)   # dD/d(le)
        cq = om ** 3 * K * Dp / np.maximum(lam, 1e-300)                      # -dt/dq = cq q (x) q
        Ttt = (kk - cq * Dt ** 2) * w
        Tnn = np.where(Dn_ > 0, kk - cq * qn ** 2, K * (Dn_ < 0)) * w
        Ttn = -cq * Dt * qn * w
        tx, ty, bx, by = 2 * t, 2 * t + 1, 2 * b, 2 * b + 1
        R, C, V = [], [], []
        for (ra, rb), (ca, cb), T in (((tx, bx), (tx, bx), Ttt), ((ty, by), (ty, by), Tnn),
                                      ((tx, bx), (ty, by), Ttn), ((ty, by), (tx, bx), Ttn)):
            R += [ra, rb, ra, rb]; C += [ca, cb, cb, ca]; V += [T, T, -T, -T]
        Kc = sp.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(s.ndof, s.ndof)).tocsr()
        return f, D, Kc

    # ------------------------------------------------------------------------- volume
    def _spec(s, m, R, c, sn, sign, coef, a, b):
        """Énergie coef * sum <e_i>_±^2 avec e_1,2 = a m ± b R, en variables (m, x, y), R = |(x, y)|,
        m = (exx+eyy)/2, x = (exx-eyy)/2, y = gxy/2. Retourne énergie, gradient et hessienne en (m, x, y).
          Miehe : coef = mu, a = b = 1 (déformations principales)
          OED   : coef = 1/2, a = sqrt(2 kappa), b = sqrt(2 mu) (valeurs propres de C^{1/2} : eps)"""
        mt, Rt = a * m, b * R; e1, e2 = mt + Rt, mt - Rt
        if sign > 0: p1, p2, H1, H2 = np.maximum(e1, 0), np.maximum(e2, 0), e1 > 0, e2 > 0
        else:        p1, p2, H1, H2 = np.minimum(e1, 0), np.minimum(e2, 0), e1 < 0, e2 < 0
        A = 2 * coef * (H1 * 1. + H2); B = 2 * coef * (H1 * 1. - H2); fR = 2 * coef * (p1 - p2)
        q = np.where(R > 1e-14, fR / np.maximum(Rt, 1e-300), A)             # phi_R / R (limite A si R -> 0)
        return (coef * (p1 ** 2 + p2 ** 2), a * 2 * coef * (p1 + p2), b * fR * c, b * fR * sn,
                a * a * A, a * b * B * c, a * b * B * sn,
                b * b * (A * c ** 2 + q * sn ** 2), b * b * (A - q) * c * sn, b * b * (A * sn ** 2 + q * c ** 2))

    def strains(s, u):
        super().strains(u)                                                   # exx, eyy, gxy, trp, trn (+ Amor)
        if s.p.split == "amor": return
        m = 0.5 * (s.exx + s.eyy); x = 0.5 * (s.exx - s.eyy); y = 0.5 * s.gxy
        R = np.sqrt(x ** 2 + y ** 2); ok = R > 1e-14
        c = np.where(ok, x / np.maximum(R, 1e-300), 1.); sn = np.where(ok, y / np.maximum(R, 1e-300), 0.)
        if s.p.split == "oed":       # psi± = 1/2 |<C^{1/2} eps>_±|^2 ; pas de terme en trace séparé
            par = (0.5, np.sqrt(2 * s.kap), np.sqrt(2 * s.mu)); s.lam_v = 0.
        else:                        # Miehe : psi± = lam/2 <tr>_±^2 + mu sum <eps_i>_±^2
            par = (s.mu, 1., 1.); s.lam_v = s.lam
        s.sp = s._spec(m, R, c, sn, +1, *par); s.sm = s._spec(m, R, c, sn, -1, *par)
        s.psip = 0.5 * s.lam_v * s.trp ** 2 + s.sp[0]
        s.psin = 0.5 * s.lam_v * s.trn ** 2 + s.sm[0]

    def fint_bulk(s, d):
        if s.p.split == "amor": return super().fint_bulk(d)
        gd = s.gdeg(d); fe = np.zeros((s.ne, 8))
        for g in range(4):
            G = gd[g]; P, N = s.sp, s.sm
            sv = s.lam_v * (G * s.trp[g] + s.trn[g]) + 0.5 * (G * P[1][g] + N[1][g])
            fe += (s.w * sv)[:, None] * s.Bv[g] + (0.5 * s.w * (G * P[2][g] + N[2][g]))[:, None] * s.Bd[g] \
                + (0.5 * s.w * (G * P[3][g] + N[3][g]))[:, None] * s.Bs[g]
        return np.bincount(s.ed.ravel(), fe.ravel(), s.ndof)

    def ktan_bulk(s, d):
        """Tangente consistante du volume (d fixé) : Ke = sum_g B^T H B, B = [tr ; exx-eyy ; gxy], H 3x3 symétrique."""
        gd = s.gdeg(d); Ke = np.zeros((s.ne, 8, 8)); Hm = np.zeros((s.ne, 3, 3))
        for g in range(4):
            G = gd[g]; Hm[:] = 0.
            if s.p.split == "amor":
                Hm[:, 0, 0] = np.where(s.trp[g] > 0, G, 1.) * s.kap * s.w
                Hm[:, 1, 1] = Hm[:, 2, 2] = G * s.mu * s.w
            else:
                P, N = s.sp, s.sm; H = [0.25 * s.w * (G * P[k][g] + N[k][g]) for k in range(4, 10)]   # mm mx my xx xy yy
                Hm[:, 0, 0] = s.lam_v * np.where(s.trp[g] > 0, G, 1.) * s.w + H[0]
                Hm[:, 0, 1] = Hm[:, 1, 0] = H[1]; Hm[:, 0, 2] = Hm[:, 2, 0] = H[2]
                Hm[:, 1, 1] = H[3]; Hm[:, 1, 2] = Hm[:, 2, 1] = H[4]; Hm[:, 2, 2] = H[5]
            Ke += s.Bt[g] @ (Hm @ s.B3[g])
        return s.asm(s.pu, Ke, s.ndof)

    def F(s, u, d, Dn, tangent=False):
        s.strains(u); fb = s.fint_bulk(d)
        if not tangent:
            fc, D = s.czm_force(u, d, Dn); return fb + fc, D
        fc, D, Kc = s.czm_force(u, d, Dn, True)
        return fb + fc, D, s.ktan_bulk(d) + Kc

    # ------------------------------------------------------------------------- champ de phase
    def d_system(s, u, d, D):
        """Opérateur et second membre du problème (linéaire) en d, à u et D fixés."""
        s.strains(u)
        ke = np.einsum('ge,gij->eij', 2 * s.w * s.psip, s.NN)
        be = np.einsum('ge,gi->ei', 2 * s.w * s.psip, NG)
        A = s.Ad0 + s.asm(s.pd, ke, s.nd)
        b = np.bincount(s.dconn.ravel(), be.ravel(), s.nd)
        _, _, _, lam, _, _, _ = s.czm_state(u, d, D)
        c = (1 - D) * s.K * lam ** 2 * s.ip_w                                # 2 (1-D) psi0 w
        t, bb = s.dmap[s.ip_t], s.dmap[s.ip_b]
        A = A + sp.coo_matrix((np.tile(0.25 * c, 4), (np.r_[t, bb, t, bb], np.r_[t, bb, bb, t])), shape=A.shape).tocsr()
        np.add.at(b, t, 0.5 * c); np.add.at(b, bb, 0.5 * c)
        b[s.d_unused] = 0.
        return A, b

    def solve_d(s, u, d, D, lb, maxit=30):
        """min 1/2 d.A.d - b.d,  lb <= d <= 1  (ensemble actif primal-dual)."""
        A, b = s.d_system(u, d, D)
        x = lb.copy(); tol = 1e-10 * (np.abs(b).max() + 1e-300)
        free = (A @ x - b) < -tol
        for it in range(maxit):
            idx = np.flatnonzero(free)
            if idx.size == 0: break
            x = lb.copy()
            if idx.size == s.nd: x = lsolve(A, b)
            else:
                Ar = A[idx]; xf = lb.copy(); xf[idx] = 0.
                x[idx] = lsolve(Ar[:, idx], b[idx] - Ar @ xf)
            g = A @ x - b
            new = (free & (x > lb)) | (~free & (g < -tol))
            x = np.maximum(x, lb)
            if np.array_equal(new, free): break
            free = new
        return np.clip(x, lb, 1.)

    # ------------------------------------------------------------------------- énergies
    def energy_parts(s, u, d, D):
        s.strains(u); gd = s.gdeg(d)
        Dt, Dn_, qn, lam, om, le, _ = s.czm_state(u, d, D)
        e_int = np.sum(s.ip_w * (om ** 2 * (1 - D) * 0.5 * s.K * lam ** 2 + 0.5 * s.K * np.minimum(Dn_, 0) ** 2))
        e_el = np.sum(s.w * (gd * s.psip + s.psin)) + e_int
        e_pf = 0.5 * d @ ((s.Kg + s.cm * s.Mc) @ d)
        dis = 0.5 * s.K * s.Do ** 2 * s.Dc * D / (s.Dc - D * (s.Dc - s.Do))   # int_0^D Yc
        e_czm = np.sum((s.ip_w * dis)[~s.notch])
        return e_el, e_pf, e_czm

    def energies(s):
        e_el, e_pf, e_czm = s.energy_parts(s.u, s.d, s.D)
        return e_el, 0.5 * np.sum(s.M * s.v ** 2), e_pf, e_czm

    # ------------------------------------------------------------------------- un pas HHT-alpha
    def step(s, dt):
        p = s.p; al = p.alpha_hht; be = (1 + al) ** 2 / 4; ga = 0.5 + al
        up = s.u + dt * s.v + dt ** 2 * (0.5 - be) * s.a                     # prédicteur
        u = up.copy(); u[s.bc_dof] = s.bc_val; d = s.d.copy()
        cm = s.M / (be * dt ** 2); pen = sp.diags(s.bcmask * 1e30)
        nit = 0
        for k in range(p.stag_max):
            for it in range(p.newton_max):                                   # --- Newton sur u (d fixé)
                Fi, D, Kt = s.F(u, d, s.D, True)
                R = cm * (u - up) + (1 - al) * Fi + al * s.Fn; R[s.bc_dof] = 0
                ref = np.linalg.norm(((1 - al) * Fi + al * s.Fn)[~s.bcmask]) + 1e-12
                if np.linalg.norm(R) < p.newton_tol * ref: break
                u -= lsolve(sp.diags(cm) + (1 - al) * Kt + pen, R); nit += 1
            else: return False, nit
            dn = s.solve_d(u, d, D, s.d)                                     # --- d (u, D fixés)
            err = np.abs(dn - d).max(); d = dn
            if err < p.stag_tol: break
        Fi, D = s.F(u, d, s.D)
        a = (u - up) / (be * dt ** 2); a[s.bc_dof] = 0
        s.v = s.v + dt * ((1 - ga) * s.a + ga * a); s.v[s.bc_dof] = 0
        s.u, s.a, s.d, s.D, s.Fn = u, a, d, D, Fi
        s.m_int = np.where(s.notch, 0., (1 - 0.5 * (d[s.dmap[s.ip_t]] + d[s.dmap[s.ip_b]])) ** 2 * (1 - D))
        s.n_stag = k + 1
        return True, nit

    def run(s, verbose=True):
        p = s.p; t0 = time.time()
        s.prestrain()                                                        # statique élastique, d = 0
        s.Fn, _ = s.F(s.u, s.d, s.D)
        s.a = -s.Fn / s.M; s.a[s.bc_dof] = 0
        t = 0.; dt = s.dt0; hist = []; snaps = []; tsnap = list(np.linspace(0, p.t_end, p.n_snap + 1)[1:]); n = 0
        if verbose:
            print(f"{s.ne} éléments, {s.ndof} ddl u | dt0 = {dt:.2e} s | alpha = {p.alpha_hht} | sig_i = {p.sig_i/1e6:.1f} MPa,"
                  f" Gc_i = {p.Gc_i:.0f}, K = {s.K:.2e} | statique {time.time()-t0:.1f} s")
        while t < p.t_end - 1e-15:
            sv = (s.u, s.v, s.a, s.d, s.D, s.Fn)
            ok, nit = s.step(dt)
            if not ok:
                s.u, s.v, s.a, s.d, s.D, s.Fn = sv; dt /= 2
                if dt < s.dt0 / 64: raise RuntimeError("Newton ne converge pas")
                continue
            t += dt; n += 1
            hist.append((t, *s.tips(), *s.energies()))
            if tsnap and t >= tsnap[0] - 1e-15:
                tsnap.pop(0); snaps.append((t, s.d.copy(), s.m_int.copy())); h_ = hist[-1]
                if verbose:
                    print(f"  t = {t*1e6:6.2f} µs | pas {n} (dt {dt:.1e}, {s.n_stag} it. alternées, {nit} Newton) | pointe CZM {h_[1]*1e3:6.2f}"
                          f" PF {h_[2]*1e3:6.2f} mm | E_pf {h_[5]:.3e} E_czm {h_[6]:.3e} | {time.time()-t0:.0f} s")
            dt = min(s.dt0, dt * 1.5)
        s.hist = np.array(hist); s.snaps = snaps
        return s.hist

    # ------------------------------------------------------------------------- vérifications
    def check(s, seed=0):
        """Sur un état perturbé avec chargement cohésif : (1) tangente K_uu vs différences finies,
        (2) symétrie de K_uu, (3) symétrie du couplage K_ud = K_du^T, (4) forces = gradient de l'énergie."""
        rng = np.random.default_rng(seed); s.prestrain()
        u = s.u.copy(); u[1::2] *= 1.0
        u[2 * s.ip_t + 1] += rng.uniform(0, 1.2 * s.Dc, s.ip_t.size)         # ouvertures dans toute la loi
        u[2 * s.ip_t] += rng.uniform(-0.3, 0.3, s.ip_t.size) * s.Dc
        u += 1e-7 * rng.standard_normal(s.ndof)
        d = rng.uniform(0, 0.6, s.nd); Dn = s.D.copy()
        du = rng.standard_normal(s.ndof) * 1e-6; dd = rng.standard_normal(s.nd) * 1e-2
        F0, D, Kt = s.F(u, d, Dn, True)
        print(f"points cohésifs : {np.mean((D > Dn) & (D < 1)):.0%} en chargement, {np.mean(D >= 1):.0%} rompus")
        for eps in (1e-2, 1e-3, 1e-4):
            fd = (s.F(u + eps * du, d, Dn)[0] - s.F(u - eps * du, d, Dn)[0]) / (2 * eps)
            print(f"(1) eps = {eps:.0e} : |K du - DF| / |K du| = {np.linalg.norm(Kt @ du - fd) / np.linalg.norm(Kt @ du):.2e}")
        print(f"(2) |K - K^T| / |K| = {spla.norm(Kt - Kt.T) / spla.norm(Kt):.2e}")
        def Fd(u_, d_):                                                      # résidu en d, D condensé
            D_ = s.czm_state(u_, d_, Dn)[6]; A, b = s.d_system(u_, d_, D_); return A @ d_ - b
        e1, e2 = 1e-3, 1e-3
        a1 = du @ (s.F(u, d + e2 * dd, Dn)[0] - s.F(u, d - e2 * dd, Dn)[0]) / (2 * e2)
        a2 = dd @ (Fd(u + e1 * du, d) - Fd(u - e1 * du, d)) / (2 * e1)
        print(f"(3) du.K_ud.dd = {a1:.6e} ; dd.K_du.du = {a2:.6e} ; écart relatif {abs(a1 - a2) / abs(a1):.2e}")
        def E(u_, d_):
            D_ = s.czm_state(u_, d_, Dn)[6]; return sum(s.energy_parts(u_, d_, D_))
        gu = (E(u + e1 * du, d) - E(u - e1 * du, d)) / (2 * e1); gdd = (E(u, d + e2 * dd) - E(u, d - e2 * dd)) / (2 * e2)
        print(f"(4) dE/du.du = {gu:.6e} vs F.du = {F0 @ du:.6e} ; dE/dd.dd = {gdd:.6e} vs Fd.dd = {Fd(u, d) @ dd:.6e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in asdict(ParamsImp()).items():
        if isinstance(v, bool): ap.add_argument(f"--{k}", action="store_true")
        else: ap.add_argument(f"--{k}", type=(str if isinstance(v, str) else float), default=v)
    ap.add_argument("--out", default="pf_czm_imp"); ap.add_argument("--check", action="store_true")
    a = vars(ap.parse_args()); out = a.pop("out"); chk = a.pop("check")
    for k in ("n_out", "n_snap", "newton_max", "stag_max"): a[k] = int(a[k])
    mdl = ModelImp(ParamsImp(**a))
    if chk: mdl.check()
    else:
        mdl.run(); mdl.save(out); mdl.plot(out, cR=mdl.p.cR); print("->", out + ".png", out + ".npz")
