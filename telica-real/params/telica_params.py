"""Telica BHL physical parameters: the single source for baseline, friction and controller path.

Rule (TR-003): the Telica datasheet where it has the value, Garcia 2013 Table I otherwise.
Force scale (TR-004): the model is in physical newtons with datasheet masses; the plant input is
F = I_logged * Kt * s_F per stage axis [X1, X2, Y]. Reading (A) s_F = [3.469, 3.469, 3.202]
(D-112) is the default representation; reading (B) s_F = 1 with 3.5x lighter effective masses is
kept as a labelled alternative. Only param / s_F is identifiable from the data.

    python telica-real/params/telica_params.py     prints the PARAMS.md table

Every other telica-real module imports numbers from here; the vendored `gantry_ss.py` and
`gantry_dynamic/controller.py` do too (G1).
"""
from dataclasses import dataclass

DS = 'Telica datasheet ASME-YGNN-08-0750-0800W3 v1.0'
GA = 'garcia2013 Table I (PDF pp. 12-13)'


@dataclass(frozen=True)
class Param:
    name: str
    value: float
    unit: str
    source: str
    identifiable: str
    note: str = ''


# Identifiability labels (reduced block D-190/D-191: ten combinations; plus the force scale).
_FS = 'only as value/s_F (force scale)'
ROWS = (
    # ==== masses and inertias =============================================================
    Param('mh', 19.0, 'kg', f'{DS} p.2 "Moving mass" Y',
          f'yes (combination mh), {_FS}', 'Y carriage; the datasheet lists Z (1.7 kg) separately'),
    Param('m_X', 91.0, 'kg', f'{DS} p.2 "Moving mass" X, per gantry beam',
          f'yes as m1+m2+mb+mh, {_FS}', 'whole X-moving assembly incl. Y carriage'),
    Param('m1', 72.0 * 10.2 / 43.7, 'kg', f'm_X - mh split by {GA} proportions 10.2:10.7:22.8',
          'no (only m1+m2+mb and m1-m2)', 'HEURISTIC split (D-112)'),
    Param('m2', 72.0 * 10.7 / 43.7, 'kg', f'm_X - mh split by {GA} proportions',
          'no (only m1+m2+mb and m1-m2)', 'HEURISTIC split (D-112)'),
    Param('mb', 72.0 * 22.8 / 43.7, 'kg', f'm_X - mh split by {GA} proportions',
          'no (only m1+m2+mb)', 'HEURISTIC split (D-112)'),
    Param('Jb', 1.0, 'kg m^2', GA, 'no (only J_eff = Jb+Jh+(m1+m2)Lb^2/4)', 'different machine'),
    Param('Jh', 0.05, 'kg m^2', GA, 'no (only J_eff)', 'different machine'),
    # ==== viscous friction and joint dynamics ==============================================
    Param('cg1', 136.0, 'N/(m/s)', f'{DS} p.2 "Dynamic friction (maximal value)" X, per motor',
          f'yes, {_FS}', 'maximal spec value: upper bound / init'),
    Param('cg2', 136.0, 'N/(m/s)', f'{DS} p.2 "Dynamic friction (maximal value)" X, per motor',
          f'yes, {_FS}', 'maximal spec value: upper bound / init'),
    Param('cy', 98.0, 'N/(m/s)', f'{DS} p.2 "Dynamic friction (maximal value)" Y',
          f'yes, {_FS}', 'maximal spec value: upper bound / init'),
    Param('cb1', 9.0, 'Nm/(rad/s)', GA, 'no (only cb1+cb2)', 'different machine'),
    Param('cb2', 9.0, 'Nm/(rad/s)', GA, 'no (only cb1+cb2)', 'different machine'),
    Param('kb1', 1987.5, 'Nm/rad', GA, 'no (only kb1+kb2)', 'different machine'),
    Param('kb2', 1987.5, 'Nm/rad', GA, 'no (only kb1+kb2)', 'different machine'),
    # ==== geometry ==========================================================================
    Param('Lb', 0.725, 'm', GA, 'no (frame constant of P; fixed)', 'different machine'),
    Param('d', 0.1, 'm', GA, 'yes (combination d)', 'different machine'),
    # ==== Coulomb friction =================================================================
    Param('cc1', 43.0, 'N', f'{DS} p.2 "Static friction (maximal value)" X, per motor',
          f'yes if velocity reverses on the rail, {_FS}',
          'static maximal: upper bound for kinetic cc (D-116 init)'),
    Param('cc2', 43.0, 'N', f'{DS} p.2 "Static friction (maximal value)" X, per motor',
          f'yes if velocity reverses on the rail, {_FS}', 'as cc1'),
    Param('ccy', 49.0, 'N', f'{DS} p.2 "Static friction (maximal value)" Y',
          f'yes if velocity reverses on the rail, {_FS}', 'as cc1'),
    Param('V_BRK', 2.25e-3, 'm/s', 'D-209: lee2020feeddrive Table 5 p.2839 (THK SSR20XW guide)',
          'no (not identifiable from constant-velocity-free data)', 'a transfer; Karnopp band'),
    Param('V0_TANH', 1e-3, 'm/s', 'D-116 HEURISTIC (Makkar & Dixon 2005 form)',
          'n/a (surrogate width)', 'differentiable surrogate only'),
    # ==== actuation and sampling ============================================================
    Param('Kt_X', 109.0, 'N/Arms', f'{DS} p.3 Kt; = Telica 1.mat MotorForceConst X',
          'no (degenerate with s_F and masses)', 'per motor; one motor per logged X channel'),
    Param('Kt_Y', 77.6, 'N/Arms', f'{DS} p.3 Kt; = Telica 1.mat MotorForceConst Y',
          'no (degenerate with s_F and masses)', ''),
    Param('s_F_X', 3.469, '-', 'D-112 HEURISTIC, reading (A): 91.0/26.229',
          'no (degenerate with masses)', 'reading (B): 1.0 with m_X,eff = 26.2 kg (TR-004)'),
    Param('s_F_Y', 3.202, '-', 'D-112 HEURISTIC, reading (A): 19.0/5.933',
          'no (degenerate with masses)', 'reading (B): 1.0 with mh,eff = 5.9 kg (TR-004)'),
    Param('Ts', 5e-5, 's', 'Telica 1.mat SamplingTime; = controller zpk Ts (D-073)',
          'n/a', 'logs are 20 kHz native'),
)
PARAMS = {p.name: p for p in ROWS}

# ==== module-level floats (what the code imports) ==========================================
mh, m1, m2, mb = (PARAMS[k].value for k in ('mh', 'm1', 'm2', 'mb'))
Jb, Jh = PARAMS['Jb'].value, PARAMS['Jh'].value
cg1, cg2, cy = PARAMS['cg1'].value, PARAMS['cg2'].value, PARAMS['cy'].value
cb1, cb2 = PARAMS['cb1'].value, PARAMS['cb2'].value
kb1, kb2 = PARAMS['kb1'].value, PARAMS['kb2'].value
Lb, d = PARAMS['Lb'].value, PARAMS['d'].value
CC = (PARAMS['cc1'].value, PARAMS['cc2'].value, PARAMS['ccy'].value)     # stage [X1, X2, Y] N
V_BRK = PARAMS['V_BRK'].value
V0_TANH = PARAMS['V0_TANH'].value
KT = (PARAMS['Kt_X'].value, PARAMS['Kt_X'].value, PARAMS['Kt_Y'].value)  # stage [X1, X2, Y]
TS = PARAMS['Ts'].value
FS = 1.0 / TS

# Force scale readings (TR-004). The model uses reading A unless told otherwise.
S_FORCE = {'A': (PARAMS['s_F_X'].value, PARAMS['s_F_X'].value, PARAMS['s_F_Y'].value),
           'B': (1.0, 1.0, 1.0)}
FORCE_SCALE_READING = 'A'


def a_to_n(reading=FORCE_SCALE_READING):
    """Logged current [A] -> stage force [N], per stage axis [X1, X2, Y]."""
    s = S_FORCE[reading]
    return tuple(k * f for k, f in zip(KT, s))


# Order of `_Trainable_Gantry_State_Block.PARAM_NAMES`.
RAW14_NAMES = ('kb1', 'kb2', 'cg1', 'cg2', 'cy', 'cb1', 'cb2',
               'mh', 'm1', 'm2', 'mb', 'Jb', 'Jh', 'd')


def raw14():
    """The fourteen raw scalars in block order, as floats."""
    return tuple(PARAMS[n].value for n in RAW14_NAMES)


def combos10():
    """The ten identifiable combinations (reduced block order), as a dict of floats."""
    return {
        'kb_sum': kb1 + kb2, 'cg1': cg1, 'cg2': cg2, 'cy': cy, 'cb_sum': cb1 + cb2, 'mh': mh,
        'm_total': m1 + m2 + mb, 'm_diff': m1 - m2,
        'J_eff': Jb + Jh + (m1 + m2) * Lb ** 2 / 4, 'd': d,
    }


def markdown_table():
    lines = ['| Parameter | Value | Unit | Source | Identifiable | Note |', '|-|-|-|-|-|-|']
    for p in ROWS:
        lines.append(f'| {p.name} | {p.value:.6g} | {p.unit} | {p.source} | '
                     f'{p.identifiable} | {p.note} |')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(markdown_table())
    print()
    print('ten combinations:', {k: round(v, 6) for k, v in combos10().items()})
    print('A->N reading A:', a_to_n('A'), ' reading B:', a_to_n('B'))
