# Galactic dynamics

`astro_polars.galactic` turns observables into Galactocentric positions and
velocities; `astro_polars.dynamics` adds the `df.galactic` namespace, whose
`calc` method derives kinematic and dynamical quantities (angular momenta,
energies, actions, orbit extent) with their dependencies resolved
automatically. Energies and actions use [AGAMA](https://github.com/GalacticDynamics-Oxford/Agama).

```python
import polars as pl
import astro_polars  # registers df.sky, df.galactic, ...
from astro_polars.surveys import gaia

lf = pl.scan_parquet("gaia_dr3_rvs/*.parquet")
lf = gaia.process(lf, correct_rv=True, zero_point="L21")   # l, b, pml, pmb, X..vZ, R, vphi, ...
lf = lf.filter(gaia.dodd23_quality())
lf = lf.galactic.calc(["E", "Lz", "Lp", "JR", "Jz", "Jphi", "circ", "ecc"])
df = lf.collect()
```

`calc` starts from whatever is already there: Galactocentric `X, Y, Z, vX,
vY, vZ`, or the observables `ra, dec, pmra, pmdec, radial_velocity` with
`distance` (kpc) or `parallax` (mas, inverted). Map other column names with
`columns={"radial_velocity": "rv_corrected"}`. Columns that already exist are
reused, internal helper columns (names starting with `_`) are dropped, and
the potential is loaded only if a requested quantity needs it.

## Frame and sign conventions

- Galactocentric frame as astropy's `Galactocentric`: right-handed, Sun at
  `X ~ -R0` (and `Z = z0`), `Y` towards `l = 90 deg`, `Z` to the North
  Galactic Pole.
- Default solar parameters (`galactic.SolarParameters`): `R0 = 8.2 kpc`,
  `z0 = 0`, `vcirc = 232.8 km/s` (McMillan 2017), `(U, V, W) = (11.1,
  12.24, 7.25) km/s` (Schoenrich et al. 2010). Pass `solar=` to any function
  or change the default with `galactic.set_default_solar(R0=8.122)`.
- `vphi` (alias `vT`) is positive along Galactic rotation; `L = -(r x v)`,
  so `Lz = R vphi > 0` for disc orbits.
- `Jphi`, `thetaphi` and `Omegaphi` use the same orientation, so `Jphi = Lz`
  in axisymmetric potentials (AGAMA's raw `Jphi` has the opposite sign in this
  frame).
- `phi = atan2(Y, X)` in degrees; the Sun is at 180.
- Units: kpc, km/s, mas/yr; energies in (km/s)^2, actions in kpc km/s,
  frequencies in km/s/kpc.

## Quantities

`astro_polars.dynamics.available()` returns the full table. Main entries:

| Group | Names |
|---|---|
| Observed | `distance`, `l`, `b`, `pml`, `pmb`, `v_l`, `v_b`, `vtan`, `distance_modulus` |
| Phase space | `X`, `Y`, `Z`, `vX`, `vY`, `vZ` |
| Cylindrical, spherical | `R`, `phi`, `vR`, `vphi`, `r`, `vr`, `vtheta`, `vt`, `v_toomre` |
| Angular momentum | `Lx`, `Ly`, `Lz`, `L`, `Lp` |
| Energy (potential) | `Ekin`, `Epot`, `E` |
| Orbit extent (potential) | `peri`, `apo`, `ecc`, `RcircE`, `RcircL`, `Tcirc`, `Lcirc`, `circ` |
| Actions (potential) | `JR`, `Jz`, `Jphi`, `Jtot`, `Jdiamond`, `Jphi_Jtot`, `sqrt_JR`, `sqrt_Jz`, `sqrt_Jphi`, `sqrt_Lz` |
| Angles, frequencies (potential) | `thetaR`, `thetaz`, `thetaphi`, `OmegaR`, `Omegaz`, `Omegaphi` |

Names from polars_astro are accepted and added under that name too:
`U` (= `Epot`), `ke` (`Ekin`), `En` (`E`), `vT` (`vphi`), `Lcirc_E`
(`Lcirc`), `AR, Az, Aphi` (angles), `FreqR, Freqz, Freqphi` (frequencies),
`distance_mod_mag` (`distance_modulus`).

`peri` and `apo` come from AGAMA's `Rperiapo(E, L)`: exact in spherical
potentials and an approximation for orbits far from the plane in flattened
ones. When angles or frequencies are requested, actions, angles and
frequencies come from a single call to the action finder.

## Potentials

```python
from astro_polars import dynamics

lf.galactic.calc("E", potential="McMillan17")       # a potential shipped with AGAMA
lf.galactic.calc("E", potential="my_model.ini")     # any AGAMA .ini file
dynamics.set_default_potential("MWPotential2014")   # default for later calls
model = dynamics.GalaxyModel("Cautun20", interp=True)  # interpolated (faster) actions
lf.galactic.calc(["JR", "Jz"], potential=model)
```

The default is `Cautun20` (Cautun et al. 2020). Names are looked up in
`$AGAMA_DATA`, `~/.config/agama` and AGAMA's `data` folder. Loading a
potential from a file sets AGAMA's units to kpc, km/s and solar masses.
Models are cached, so the action finder is built once per potential.

AGAMA runs in parallel (OpenMP) inside each Polars batch; quantities that
need it work on lazy frames and on the streaming engine.

## Reflex motion and GSR velocities

```python
lf.galactic.reflex_correct()                # pmra_gsr, pmdec_gsr
lf.galactic.reflex_correct(distance=10.0)   # one distance for every star, e.g. a stream
lf.galactic.vgsr()                          # line-of-sight velocity in the GSR
lf.galactic.to_observed()                   # Galactocentric -> ra, dec, distance, PMs, RV
```
