# PF_CZM_Transition

Couplage champ de phase (PF) / zone cohésive (CZM) en dynamique, sur le cas test de la plaque
pré-déformée de Bleyer et al. (2017) en mode I. Objectif : étudier la transition d'une fissure
portée par la CZM vers une fissure portée par le champ de phase.

## Scripts

| Fichier | Contenu |
|---|---|
| `pf_czm_bleyer.py` | AT1 (ou AT2), décomposition d'Amor, Newmark explicite. Couplages `quintanas`, `paggi`, `none`. Fournit aussi le maillage et les sorties. |
| `pf_czm_implicit.py` | AT2, loi cohésive bilinéaire sous forme énergétique, HHT-alpha implicite, Newton avec tangente consistante, minimisation alternée. Décompositions `oed` (défaut), `miehe`, `amor`. Importe `pf_czm_bleyer.py`. |

Dépendances : `numpy`, `scipy`, `matplotlib` ; `pypardiso` optionnel (solveur plus rapide).

```bash
python pf_czm_implicit.py --check                      # tangente, symétries, forces = gradient de l'énergie
python pf_czm_implicit.py --l0 2e-4 --h 1e-4 --sig_i 18e6 --Gc_i 150 --dU 0.040e-3 --t_end 40e-6
python pf_czm_implicit.py --l0 2e-4 --h 1e-4 --sig_i 15e6 --Gc_i 150 --x_czm_end 16e-3 --t_end 45e-6
python pf_czm_bleyer.py --l0 2e-4 --h 5e-5 --t_end 45e-6    # explicite AT1, couplage Quintanas
```

## Loi de couplage (script implicite)

Énergie d'interface, avec `phi` la moyenne de `d` sur les deux lèvres, `D` l'endommagement cohésif,
`lambda` l'ouverture :

    Psi_i = (1 - phi)^2 (1 - D) 1/2 K lambda^2 + 1/2 K <-Delta_n>^2 + int_0^D Yc

- traction : `t = (1 - phi)^2 (1 - D) K q`
- `D` : loi bilinéaire lue sur l'ouverture effective `(1 - phi) lambda`
- force motrice du PF : `-2 (1 - phi) (1 - D) 1/2 K lambda^2`

## Résultats obtenus (maillages grossiers, voir `figures/`)

- À ténacités égales (`Gc_i = Gc`), l'énergie est indifférente entre CZM et PF : la dissipation se
  partage et le PF finit par prendre la main. Le rapport `Gc_i / Gc` pilote qui mène.
- Avec `Gc_i = Gc/2` et `sigma_precontrainte < sig_i < sigma_AT2`, la CZM mène (87 % de la dissipation).
- Interface arrêtée à mi-longueur : transition nette vers le PF en bout d'interface.
- Interface sur toute la longueur, `dU` = 0,035 et 0,040 mm : pas de transition, vitesse limite
  0,83 c_R, dissipation totale 1,2 à 1,4 Gc (seuil 2 Gc non atteint), lobes PF diffus hors axe.

## Limites connues

- Maillages de démonstration grossiers (`h = l0/2` à `l0/4`, `l0` = 0,2 à 0,4 mm), sans étude de convergence.
- AT2 : endommagement diffus dès la pré-déformation, ce qui borne le chargement admissible.
- Énergie totale : dérive de -2 à -6 % selon les cas (HHT-alpha).
- Modes I et II de même raideur, sans critère de Benzeggagh-Kenane.
- `note/note_pf_czm.tex` décrit la version avec décomposition d'Amor ; l'OED et les derniers
  résultats n'y figurent pas encore.

## Références

- J. Bleyer, C. Roux-Langlois, J.-F. Molinari, Int. J. Fract., 2017.
- M. Paggi, J. Reinoso, Comput. Methods Appl. Mech. Engrg. 321, 2017.
- A. Quintanas-Corominas, A. Turon, J. Reinoso et al., Comput. Methods Appl. Mech. Engrg. 358, 2020.
- T.-T. Nguyen, J. Yvonnet, D. Waldmann, Q.-C. He, Int. J. Numer. Methods Eng. 121, 2020.
