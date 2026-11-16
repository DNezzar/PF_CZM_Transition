#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pf_czm_bleyer.py — Couplage champ de phase (PF) / zone cohésive (CZM) en dynamique explicite
============================================================================================
Cas test : plaque PMMA pré-déformée de Bleyer et al. (2017), mode I.
  32 mm x 16 mm, pré-entaille 4 mm, contraintes planes, +/- dU imposé en haut / bas.

Volume   : modèle d'endommagement à gradient (AT1 par défaut, comme Bleyer ; AT2 en option),
           décomposition volumique / déviatorique (Amor), irréversibilité par bornes d >= d_old.
Interface: éléments cohésifs d'épaisseur nulle sur la ligne médiane y = 0, de la pointe
           d'entaille jusqu'à x_czm_end (toute la longueur par défaut).
Couplage : "quintanas" -> loi bilinéaire (Turon), intégrité m = min(m_Delta, m_phi),
                          m_phi = (1 - r_phi)^2, + force motrice d'interface sur le PF.
           "paggi"     -> loi linéaire avec coupure en traction, ouverture critique
                          g_c(d) = (1-d) g_c0 + d g_c1 et k(d) = k0 (g_c0/g_c)^2 (G_c^i constant).
           "none"      -> CZM et PF indépendants.
Temps    : état initial statique (pré-déformation), puis Newmark explicite (différences centrées),
           masse condensée ; problème d'endommagement résolu à chaque pas (ensemble actif primal-dual).
Maillage : grille Q4 structurée, x uniforme, y raffiné dans une bande autour de y = 0.

Dépendances : numpy, scipy, matplotlib.     Unités : SI.
Exemples :
  python pf_czm_bleyer.py                         # CZM sur toute la longueur
  python pf_czm_bleyer.py --x_czm_end 16e-3       # CZM jusqu'à mi-longueur, PF seul ensuite
  python pf_czm_bleyer.py --dU 0.045e-3 --Gc_i 600 --coupling paggi
"""
import argparse, time
from dataclasses import dataclass, asdict
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ----------------------------------------------------------------------------- paramètres
@dataclass
class Params:
    # géométrie / chargement (Bleyer 2017, §3.1)
    L: float = 32e-3; H: float = 16e-3; a0: float = 4e-3; dU: float = 0.035e-3
    # PMMA
    E: float = 3.09e9; nu: float = 0.35; rho: float = 1180.; Gc: float = 300.; l0: float = 1e-4
    model: str = "AT1"            # "AT1" (Bleyer) ou "AT2"
    kres: float = 1e-8            # rigidité résiduelle
    # maillage
    h: float = None               # taille de maille fine (défaut l0/4)
    y_band: float = 1.5e-3        # demi-hauteur de la bande fine (None -> uniforme)
    growth: float = 1.15; hy_max_fac: float = 10.
    # interface cohésive
    x_czm_end: float = None       # fin de la CZM (défaut L : toute la longueur)
    Gc_i: float = None            # ténacité d'interface (défaut Gc)
    sig_i: float = None           # résistance cohésive (défaut : contrainte critique AT1)
    K_fac: float = 2.             # pénalité K = K_fac * E / h
    coupling: str = "quintanas"   # "quintanas" | "paggi" | "none"
    phi_min: float = 0.2; phi_max: float = 0.8   # bornes du couplage (Quintanas)
    paggi_ratio: float = 3.       # g_c1 / g_c0 (Paggi)
    tie_d: bool = False           # True : d continu à travers l'interface
    # temps
    t_end: float = 40e-6; cfl: float = 0.6; n_out: int = 20; n_snap: int = 5

    def finalize(s):
        s.h = s.h or s.l0 / 4
        s.x_czm_end = s.L if s.x_czm_end is None else s.x_czm_end
        s.Gc_i = s.Gc if s.Gc_i is None else s.Gc_i
        s.sig_i = np.sqrt(3 * s.Gc * s.E / (8 * s.l0)) if s.sig_i is None else s.sig_i
        return s


# ----------------------------------------------------------------------------- Q4 : points de Gauss
_g = 1 / np.sqrt(3)
XI = np.array([-1, 1, 1, -1.]); ETA = np.array([-1, -1, 1, 1.])
GP = [(-_g, -_g), (_g, -_g), (_g, _g), (-_g, _g)]
NG = np.array([0.25 * (1 + XI * x) * (1 + ETA * e) for x, e in GP])   # N_a(g)
AX = np.array([0.5 * XI * (1 + ETA * e) for x, e in GP])              # dN/dx * hx
AY = np.array([0.5 * ETA * (1 + XI * x) for x, e in GP])              # dN/dy * hy


class Model:
    # ------------------------------------------------------------------------- maillage
    def __init__(s, p: Params):
        s.p = p = p.finalize()
        nx = int(round(p.L / p.h)); x = np.linspace(0, p.L, nx + 1)
        if p.y_band is None or p.y_band >= p.H / 2:
            yp = np.linspace(0, p.H / 2, int(round(p.H / 2 / p.h)) + 1)
        else:                                            # bande fine + croissance géométrique
            yp = list(np.linspace(0, p.y_band, int(round(p.y_band / p.h)) + 1)); hy = p.h
            while yp[-1] < p.H / 2 - 1e-12:
                hy = min(hy * p.growth, p.hy_max_fac * p.h); yp.append(yp[-1] + hy)
            yp = np.array(yp); yp[-1] = p.H / 2
            if yp[-1] - yp[-2] < 0.5 * (yp[-2] - yp[-3]): yp = np.delete(yp, -2)
        y = np.concatenate([-yp[:0:-1], yp]); ny = len(y) - 1; j0 = ny // 2
        X, Y = np.meshgrid(x, y); n0 = X.size
        nid = np.arange(n0).reshape(ny + 1, nx + 1)
        # nœuds dédoublés sur y = 0 (entaille + interface cohésive)
        tol = 1e-9
        idup = np.flatnonzero((x < p.x_czm_end - tol) | (p.x_czm_end >= p.L - tol))
        top = nid[j0].copy(); top[idup] = n0 + np.arange(idup.size)
        s.xy = np.vstack([np.c_[X.ravel(), Y.ravel()], np.c_[x[idup], np.zeros(idup.size)]])
        s.nn = s.xy.shape[0]; s.ndof = 2 * s.nn
        I, J = np.meshgrid(np.arange(nx), np.arange(ny))
        I, J = I.ravel(), J.ravel()
        lo = np.where(J == j0, top[I], nid[J, I]); lo1 = np.where(J == j0, top[I + 1], nid[J, I + 1])
        s.conn = np.c_[lo, lo1, nid[J + 1, I + 1], nid[J + 1, I]]
        s.ne = s.conn.shape[0]; s.nx, s.ny, s.x, s.y, s.j0 = nx, ny, x, y, j0
        s.hx = (x[1:] - x[:-1])[I]; s.hy = (y[1:] - y[:-1])[J]
        s.w = s.hx * s.hy / 4; s.ihx = 1 / s.hx; s.ihy = 1 / s.hy
        # numérotation de d (liée ou non à travers l'interface)
        s.dmap = np.arange(s.nn)
        if p.tie_d: s.dmap[top[idup]] = nid[j0][idup]
        s.nd = s.nn; s.dconn = s.dmap[s.conn]
        # points d'intégration d'interface (Newton-Cotes : aux nœuds), poids hx/2
        seg = np.arange(nx); seg = seg[x[seg] < p.x_czm_end - tol]
        s.ip_t = np.r_[top[seg], top[seg + 1]]; s.ip_b = np.r_[nid[j0][seg], nid[j0][seg + 1]]
        s.ip_w = np.r_[(x[seg + 1] - x[seg]) / 2] * 1.0; s.ip_w = np.r_[s.ip_w, s.ip_w]
        s.ip_x = np.r_[x[seg], x[seg + 1]]
        notch = np.r_[x[seg + 1] <= p.a0 + tol, x[seg + 1] <= p.a0 + tol]
        s.rD = np.where(notch, 1., 0.)          # seuil d'endommagement cohésif (1 = rompu)
        s.rP = np.zeros(s.ip_x.size)            # seuil lié au champ de phase
        s.notch = notch
        # conditions aux limites
        s.top_n = nid[ny]; s.bot_n = nid[0]
        s.bc_dof = np.r_[2 * s.top_n + 1, 2 * s.bot_n + 1]
        s.bc_val = np.r_[np.full(nx + 1, p.dU), np.full(nx + 1, -p.dU)]
        # matériau (contraintes planes)
        s.mu = p.E / (2 * (1 + p.nu)); s.lam = p.E * p.nu / (1 - p.nu ** 2); s.kap = s.lam + s.mu
        s.cd = np.sqrt((s.lam + 2 * s.mu) / p.rho)
        # masse condensée
        s.m = np.bincount(s.conn.ravel(), np.repeat(p.rho * s.hx * s.hy / 4, 4), s.nn)
        s.M = np.repeat(s.m, 2)
        # opérateurs du problème d'endommagement
        s.vol = np.bincount(s.dconn.ravel(), np.repeat(s.hx * s.hy / 4, 4), s.nd)
        Kd = np.zeros((s.ne, 4, 4))
        for g in range(4):
            bx = AX[g][None, :] * s.ihx[:, None]; by = AY[g][None, :] * s.ihy[:, None]
            Kd += s.w[:, None, None] * (bx[:, :, None] * bx[:, None, :] + by[:, :, None] * by[:, None, :])
        r = np.repeat(s.dconn, 4, axis=1).ravel(); c = np.tile(s.dconn, (1, 4)).ravel()
        s.Kg = sp.csr_matrix((Kd.ravel(), (r, c)), shape=(s.nd, s.nd))
        if p.model == "AT1": s.cg, s.cw, s.cm = 3 * p.Gc * p.l0 / 4, 3 * p.Gc / (8 * p.l0), 0.
        else:                s.cg, s.cw, s.cm = p.Gc * p.l0, 0., p.Gc / p.l0
        s.Kg = (s.cg * s.Kg).tocsr()
        used = np.zeros(s.nd, bool); used[s.dconn.ravel()] = True; s.d_unused = ~used
        # loi cohésive
        s.K = p.K_fac * p.E / p.h
        s.Dc = 2 * p.Gc_i / p.sig_i                    # ouverture critique (bilinéaire)
        s.Do = p.sig_i / s.K                           # ouverture d'amorçage
        s.g0 = 2 * p.Gc_i / p.sig_i; s.k0 = p.sig_i / s.g0; s.g1 = p.paggi_ratio * s.g0   # Paggi
        kmax = s.K if p.coupling != "paggi" else max(s.K, s.k0)
        s.dt = p.cfl / np.sqrt(2 * s.cd ** 2 / p.h ** 2 + kmax / (p.rho * p.h))
        s.u = np.zeros(s.ndof); s.v = np.zeros(s.ndof); s.a = np.zeros(s.ndof); s.d = np.zeros(s.nd)

    # ------------------------------------------------------------------------- volume
    def strains(s, u):
        """Déformations et énergies psi+/psi- aux points de Gauss."""
        ux, uy = u[0::2][s.conn], u[1::2][s.conn]
        s.exx = np.array([(ux @ AX[g]) * s.ihx for g in range(4)])
        s.eyy = np.array([(uy @ AY[g]) * s.ihy for g in range(4)])
        s.gxy = np.array([(ux @ AY[g]) * s.ihy + (uy @ AX[g]) * s.ihx for g in range(4)])
        tr = s.exx + s.eyy
        s.trp, s.trn = np.maximum(tr, 0), np.minimum(tr, 0)
        s.exd, s.eyd = s.exx - tr / 2, s.eyy - tr / 2
        s.psip = 0.5 * s.kap * s.trp ** 2 + s.mu * (s.exd ** 2 + s.eyd ** 2 + 0.5 * s.gxy ** 2)
        s.psin = 0.5 * s.kap * s.trn ** 2

    def gdeg(s, d):
        de = d[s.dconn]
        return np.array([(1 - de @ NG[g]) ** 2 + s.p.kres for g in range(4)])

    def fint_bulk(s, d):
        """Forces internes volumiques : sigma = g(d) * dpsi+/deps + dpsi-/deps."""
        gd = s.gdeg(d)
        sxx = gd * (s.kap * s.trp + 2 * s.mu * s.exd) + s.kap * s.trn
        syy = gd * (s.kap * s.trp + 2 * s.mu * s.eyd) + s.kap * s.trn
        sxy = gd * s.mu * s.gxy
        fx = np.zeros((s.ne, 4)); fy = np.zeros((s.ne, 4))
        for g in range(4):
            bx = AX[g][None, :] * (s.w * s.ihx)[:, None]; by = AY[g][None, :] * (s.w * s.ihy)[:, None]
            fx += sxx[g][:, None] * bx + sxy[g][:, None] * by
            fy += syy[g][:, None] * by + sxy[g][:, None] * bx
        f = np.zeros(s.ndof); c = s.conn.ravel()
        f[0::2] = np.bincount(c, fx.ravel(), s.nn); f[1::2] = np.bincount(c, fy.ravel(), s.nn)
        return f

    # ------------------------------------------------------------------------- interface
    def czm(s, u, d, update=True):
        """Tractions cohésives couplées au PF. Retourne f_int, et (diag, second membre)
        de la contribution d'interface au problème d'endommagement."""
        p = s.p; t, b, w = s.ip_t, s.ip_b, s.ip_w
        Dt = u[2 * t] - u[2 * b]; Dn = u[2 * t + 1] - u[2 * b + 1]
        Dnp = np.maximum(Dn, 0); lam2 = Dnp ** 2 + Dt ** 2; lam = np.sqrt(lam2)
        dt_, db_ = d[s.dmap[t]], d[s.dmap[b]]; phi = 0.5 * (dt_ + db_)
        Dd = np.zeros(s.nd); bd = np.zeros(s.nd)
        if p.coupling == "paggi":
            gc = (1 - phi) * s.g0 + phi * s.g1; k = s.k0 * (s.g0 / gc) ** 2
            rD = np.maximum(s.rD, (lam >= gc).astype(float))            # coupure en traction
            m = 1 - rD
            tn = m * k * Dnp + max(s.K, s.k0) * np.minimum(Dn, 0); tt = m * k * Dt
            dG = -m * s.k0 * lam2 * s.g0 ** 2 * (s.g1 - s.g0) / gc ** 3   # dG_i/dphi (<0)
            np.add.at(bd, s.dmap[t], -0.5 * w * dG); np.add.at(bd, s.dmap[b], -0.5 * w * dG)
            rP = s.rP; s.e_czm_el = np.sum(w * 0.5 * m * k * lam2)
        else:
            rD = np.maximum(s.rD, np.clip((lam - s.Do) / (s.Dc - s.Do), 0, 1))
            mD = 1 - rD * s.Dc / (rD * s.Dc + (1 - rD) * s.Do)
            psi = 0.5 * s.K * lam2
            if p.coupling == "quintanas":
                rP = np.maximum(s.rP, np.clip((phi - p.phi_min) / (p.phi_max - p.phi_min), 0, 1))
                m = np.minimum(mD, (1 - rP) ** 2)
                # énergie d'interface mD*psi*(1-r_phi)^2 : terme quadratique en phi, actif si 0<r_phi<1
                c = np.where((rP > 0) & (rP < 1), 2 * mD * psi / (p.phi_max - p.phi_min) ** 2, 0.) * w
                np.add.at(Dd, s.dmap[t], 0.25 * c); np.add.at(Dd, s.dmap[b], 0.25 * c)
                np.add.at(bd, s.dmap[t], 0.5 * c * (p.phi_max - 0.5 * db_))
                np.add.at(bd, s.dmap[b], 0.5 * c * (p.phi_max - 0.5 * dt_))
            else: rP, m = s.rP, mD
            tn = m * s.K * Dnp + s.K * np.minimum(Dn, 0); tt = m * s.K * Dt
            s.e_czm_el = np.sum(w * m * psi)
        if update: s.rD, s.rP = rD, rP
        s.m_int = np.where(s.notch, 0., m)
        f = np.zeros(s.ndof)
        np.add.at(f, 2 * t, w * tt); np.add.at(f, 2 * b, -w * tt)
        np.add.at(f, 2 * t + 1, w * tn); np.add.at(f, 2 * b + 1, -w * tn)
        return f, Dd, bd

    # ------------------------------------------------------------------------- endommagement
    def solve_d(s, Dd, bd, maxit=30):
        """min 1/2 d.A.d - b.d  avec  d_old <= d <= 1  (ensemble actif primal-dual)."""
        h = np.zeros(s.nd); c = s.dconn.ravel()
        for g in range(4):
            h += np.bincount(c, (2 * s.w * s.psip[g])[:, None].repeat(4, 1).ravel() * np.tile(NG[g], s.ne), s.nd)
        D = h + s.cm * s.vol + Dd; D[s.d_unused] = 1.
        b = h - s.cw * s.vol + bd; b[s.d_unused] = 0.
        lb = s.d; d = lb.copy(); tol = 1e-12 * (np.abs(b).max() + 1e-300)
        free = (s.Kg @ d + D * d - b) < -tol
        for it in range(maxit):
            idx = np.flatnonzero(free)
            if idx.size == 0: break
            dfix = lb.copy(); dfix[idx] = 0.
            Kr = s.Kg[idx]
            A = (Kr[:, idx] + sp.diags(D[idx])).tocsc()
            d = lb.copy(); d[idx] = spla.spsolve(A, b[idx] - Kr @ dfix) if idx.size > 1 else (b[idx] - Kr @ dfix) / A[0, 0]
            g = s.Kg @ d + D * d - b
            new = (free & (d > lb)) | (~free & (g < -tol))
            d = np.maximum(d, lb)
            if np.array_equal(new, free): break
            free = new
        return np.clip(d, lb, 1.)

    # ------------------------------------------------------------------------- statique initiale
    def prestrain(s):
        """Équilibre élastique linéaire (d = 0, interface intacte de raideur K) sous +/- dU."""
        C = np.array([[s.lam + 2 * s.mu, s.lam, 0], [s.lam, s.lam + 2 * s.mu, 0], [0, 0, s.mu]])
        Ke = np.zeros((s.ne, 8, 8))
        for g in range(4):
            B = np.zeros((s.ne, 3, 8))
            B[:, 0, 0::2] = AX[g] * s.ihx[:, None]; B[:, 1, 1::2] = AY[g] * s.ihy[:, None]
            B[:, 2, 0::2] = AY[g] * s.ihy[:, None]; B[:, 2, 1::2] = AX[g] * s.ihx[:, None]
            Ke += s.w[:, None, None] * np.einsum('eki,kl,elj->eij', B, C, B)
        ed = np.empty((s.ne, 8), int); ed[:, 0::2] = 2 * s.conn; ed[:, 1::2] = 2 * s.conn + 1
        K = sp.coo_matrix((Ke.ravel(), (np.repeat(ed, 8, 1).ravel(), np.tile(ed, (1, 8)).ravel())),
                          shape=(s.ndof, s.ndof)).tocsr(); del Ke
        act = ~s.notch; t, b = s.ip_t[act], s.ip_b[act]
        kk = (s.K if s.p.coupling != "paggi" else s.k0) * s.ip_w[act]
        for o in (0, 1):
            r = np.r_[2 * t + o, 2 * b + o, 2 * t + o, 2 * b + o]; c = np.r_[2 * t + o, 2 * b + o, 2 * b + o, 2 * t + o]
            K = K + sp.coo_matrix((np.r_[kk, kk, -kk, -kk], (r, c)), shape=K.shape).tocsr()
        u = np.zeros(s.ndof); u[s.bc_dof] = s.bc_val
        fix = np.r_[s.bc_dof, 2 * s.top_n[s.nx // 2], 2 * s.bot_n[s.nx // 2]]   # + blocage du mode rigide en x
        free = np.setdiff1d(np.arange(s.ndof), fix)
        u[free] = spla.spsolve(K[free][:, free].tocsc(), -(K[free] @ u))
        s.u = u

    # ------------------------------------------------------------------------- énergies / pointe
    def energies(s):
        gd = s.gdeg(s.d)
        e_el = np.sum(s.w * (gd * s.psip + s.psin)) + s.e_czm_el
        e_kin = 0.5 * np.sum(s.M * s.v ** 2)
        e_pf = s.cw * s.vol @ s.d + 0.5 * s.cm * s.vol @ s.d ** 2 + 0.5 * s.d @ (s.Kg @ s.d)
        e_czm = s.p.Gc_i * np.sum(s.ip_w[~s.notch] * s.rD[~s.notch])
        return e_el, e_kin, e_pf, e_czm

    def tips(s):
        brk = (s.m_int < 1e-3) | s.notch
        a_czm = s.ip_x[brk].max() if brk.any() else s.p.a0
        nd_ = np.flatnonzero(s.d >= 0.9)
        a_pf = s.xy[nd_, 0].max() if nd_.size else np.nan
        return a_czm, a_pf

    # ------------------------------------------------------------------------- boucle en temps
    def run(s, verbose=True):
        p = s.p; t0 = time.time()
        s.prestrain(); s.strains(s.u)
        fc, Dd, bd = s.czm(s.u, s.d)
        f = s.fint_bulk(s.d) + fc
        s.a = -f / s.M; s.a[s.bc_dof] = 0
        nstep = int(np.ceil(p.t_end / s.dt)); dt = s.dt
        hist = []; snaps = []; snap_at = set(np.linspace(0, nstep, p.n_snap + 1).astype(int)[1:])
        if verbose:
            print(f"{s.ne} éléments, {s.ndof} ddl u, {s.ip_x.size} PI cohésifs | dt = {dt:.2e} s, {nstep} pas"
                  f" | sig_i = {p.sig_i/1e6:.1f} MPa, Gc_i = {p.Gc_i:.0f} J/m2, K = {s.K:.2e}"
                  f" | statique : {time.time()-t0:.1f} s")
        for n in range(1, nstep + 1):
            s.v += 0.5 * dt * s.a                       # prédiction v_{n+1/2}
            s.u += dt * s.v                             # u_{n+1}
            s.u[s.bc_dof] = s.bc_val
            s.strains(s.u)
            _, Dd, bd = s.czm(s.u, s.d, update=False)   # force motrice d'interface (d retardé)
            s.d = s.solve_d(Dd, bd)                     # d_{n+1}
            fc, _, _ = s.czm(s.u, s.d)                  # tractions avec d_{n+1}
            f = s.fint_bulk(s.d) + fc
            s.a = -f / s.M; s.a[s.bc_dof] = 0
            s.v += 0.5 * dt * s.a; s.v[s.bc_dof] = 0    # correction v_{n+1}
            if n % p.n_out == 0 or n == nstep:
                hist.append((n * dt, *s.tips(), *s.energies()))
                if not np.isfinite(hist[-1][3]) or hist[-1][4] > 1e3 * max(hist[0][3], 1e-12):
                    raise RuntimeError("instabilité : réduire cfl")
            if n in snap_at:
                snaps.append((n * dt, s.d.copy(), s.m_int.copy()))
                if verbose:
                    h_ = hist[-1]
                    print(f"  t = {n*dt*1e6:6.2f} µs | pointe CZM {h_[1]*1e3:6.2f} mm, PF {h_[2]*1e3:6.2f} mm"
                          f" | E_pf {h_[5]:.3e}  E_czm {h_[6]:.3e} J/m | {time.time()-t0:.0f} s")
        s.hist = np.array(hist); s.snaps = snaps
        return s.hist

    # ------------------------------------------------------------------------- sorties
    def dgrid(s, d):
        """Champ d sur la grille (lèvre inférieure sur y = 0) pour le tracé."""
        return d[s.dmap[:(s.nx + 1) * (s.ny + 1)]].reshape(s.ny + 1, s.nx + 1)

    def save(s, name):
        np.savez_compressed(name + ".npz", hist=s.hist, x=s.x, y=s.y, ip_x=s.ip_x,
                            snap_t=[q[0] for q in s.snaps], snap_d=[s.dgrid(q[1]) for q in s.snaps],
                            snap_m=[q[2] for q in s.snaps], params=str(asdict(s.p)))

    def plot(s, name, cR=906.):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        p = s.p; h = s.hist; t = h[:, 0] * 1e6; ns = len(s.snaps)
        fig = plt.figure(figsize=(11, 2.1 * ns + 6.5)); gs = fig.add_gridspec(ns + 2, 2, height_ratios=[1] * ns + [1.4, 1.4])
        o = np.argsort(s.ip_x)
        for k, (ts, d, m) in enumerate(s.snaps):
            ax = fig.add_subplot(gs[k, :])
            im = ax.pcolormesh(s.x * 1e3, s.y * 1e3, s.dgrid(d), cmap="magma_r", vmin=0, vmax=1, shading="gouraud", rasterized=True)
            brk = (m[o] < 1e-3) & ~s.notch[o]; sof = (m[o] < 0.999) & ~brk & ~s.notch[o]
            ax.plot(s.ip_x[o][sof] * 1e3, 0 * s.ip_x[o][sof], ".", c="tab:orange", ms=2, label="CZM en adoucissement")
            ax.plot(s.ip_x[o][brk] * 1e3, 0 * s.ip_x[o][brk], ".", c="tab:cyan", ms=2, label="CZM rompue")
            ax.plot([0, p.a0 * 1e3], [0, 0], "k-", lw=1.5)
            if p.x_czm_end < p.L: ax.axvline(p.x_czm_end * 1e3, c="gray", ls="--", lw=0.8)
            ax.set_ylim(-4, 4); ax.set_aspect("equal"); ax.set_ylabel("y (mm)")
            ax.set_title(f"t = {ts*1e6:.1f} µs", fontsize=9, loc="left")
            if k == 0: ax.legend(loc="upper right", fontsize=7, markerscale=4, ncol=2)
        fig.colorbar(im, ax=fig.axes, fraction=0.015, pad=0.01, label="d")
        ax = fig.add_subplot(gs[ns, 0])
        ax.plot(t, h[:, 1] * 1e3, label="pointe CZM"); ax.plot(t, h[:, 2] * 1e3, label="pointe PF (d ≥ 0.9)")
        if p.x_czm_end < p.L: ax.axhline(p.x_czm_end * 1e3, c="gray", ls="--", lw=0.8)
        ax.set_xlabel("t (µs)"); ax.set_ylabel("x pointe (mm)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        ax = fig.add_subplot(gs[ns, 1])
        a = np.fmax(h[:, 1], np.nan_to_num(h[:, 2], nan=0.)); k = max(3, len(t) // 40)
        vel = (a[2 * k:] - a[:-2 * k]) / (h[2 * k:, 0] - h[:-2 * k, 0])
        ax.plot(t[k:-k], vel / cR); ax.axhline(0.6, c="gray", ls=":", lw=0.8)
        ax.set_xlabel("t (µs)"); ax.set_ylabel("v / c_R"); ax.set_ylim(0, 1.2); ax.grid(alpha=.3)
        ax = fig.add_subplot(gs[ns + 1, 0])
        for j, lab in zip((3, 4, 5, 6), ("élastique", "cinétique", "rupture PF", "dissipée CZM")): ax.plot(t, h[:, j], label=lab)
        ax.plot(t, h[:, 3:7].sum(1), "k--", lw=0.8, label="total")
        ax.set_xlabel("t (µs)"); ax.set_ylabel("énergie (J/m)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        ax = fig.add_subplot(gs[ns + 1, 1])
        ed = h[:, 5] + h[:, 6]; da = a[2 * k:] - a[:-2 * k]
        G = np.where(da > 2 * p.h, (ed[2 * k:] - ed[:-2 * k]) / np.where(da > 0, da, 1), np.nan)
        ax.plot(a[k:-k] * 1e3, G / p.Gc)
        ax.set_xlabel("x pointe (mm)"); ax.set_ylabel("Γ / G_c  (dissipation par unité d'avancée)"); ax.grid(alpha=.3)
        fig.suptitle(f"PF ({p.model}) + CZM ({p.coupling}) — ΔU = {p.dU*1e3:.3f} mm, l0 = {p.l0*1e3:.2f} mm, h = {p.h*1e3:.3f} mm, "
                     f"σ_i = {p.sig_i/1e6:.0f} MPa, G_c^i/G_c = {p.Gc_i/p.Gc:.2f}", fontsize=10)
        fig.savefig(name + ".png", dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in asdict(Params()).items():
        if isinstance(v, bool): ap.add_argument(f"--{k}", action="store_true")
        else: ap.add_argument(f"--{k}", type=(str if isinstance(v, str) else float), default=v)
    ap.add_argument("--out", default="pf_czm")
    a = vars(ap.parse_args()); out = a.pop("out")
    for k in ("n_out", "n_snap"): a[k] = int(a[k])
    mdl = Model(Params(**a)); mdl.run(); mdl.save(out); mdl.plot(out)
    print("->", out + ".png", out + ".npz")
