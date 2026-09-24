# Telica BHL parameters (G1)

Generated from `params/telica_params.py` (`python telica-real/params/telica_params.py`); edit the
module, not this table. Rule TR-003 (datasheet where available, Garcia 2013 Table I otherwise);
force scale TR-004. Identifiability: the reduced block's ten combinations (kb_sum, cg1, cg2, cy,
cb_sum, mh, m_total = m1+m2+mb, m_diff = m1-m2, J_eff = Jb+Jh+(m1+m2)Lb^2/4, d) plus the force
scale: every force-like parameter enters the data only as value / s_F.

| Parameter | Value | Unit | Source | Identifiable | Note |
|-|-|-|-|-|-|
| mh | 19 | kg | Telica datasheet ASME-YGNN-08-0750-0800W3 v1.0 p.2 "Moving mass" Y | yes (combination mh), only as value/s_F (force scale) | Y carriage; the datasheet lists Z (1.7 kg) separately |
| m_X | 91 | kg | Telica datasheet ASME-YGNN-08-0750-0800W3 v1.0 p.2 "Moving mass" X, per gantry beam | yes as m1+m2+mb+mh, only as value/s_F (force scale) | whole X-moving assembly incl. Y carriage |
| m1 | 16.8055 | kg | m_X - mh split by garcia2013 Table I (PDF pp. 12-13) proportions 10.2:10.7:22.8 | no (only m1+m2+mb and m1-m2) | HEURISTIC split (D-112) |
| m2 | 17.6293 | kg | m_X - mh split by garcia2013 Table I (PDF pp. 12-13) proportions | no (only m1+m2+mb and m1-m2) | HEURISTIC split (D-112) |
| mb | 37.5652 | kg | m_X - mh split by garcia2013 Table I (PDF pp. 12-13) proportions | no (only m1+m2+mb) | HEURISTIC split (D-112) |
| Jb | 1 | kg m^2 | garcia2013 Table I (PDF pp. 12-13) | no (only J_eff = Jb+Jh+(m1+m2)Lb^2/4) | different machine |
| Jh | 0.05 | kg m^2 | garcia2013 Table I (PDF pp. 12-13) | no (only J_eff) | different machine |
| cg1 | 136 | N/(m/s) | Telica datasheet ASME-YGNN-08-0750-0800W3 v1.0 p.2 "Dynamic friction (maximal value)" X, per motor | yes, only as value/s_F (force scale) | maximal spec value: upper bound / init |
| cg2 | 136 | N/(m/s) | Telica datasheet ASME-YGNN-08-0750-0800W3 v1.0 p.2 "Dynamic friction (maximal value)" X, per motor | yes, only as value/s_F (force scale) | maximal spec value: upper bound / init |
| cy | 98 | N/(m/s) | Telica datasheet ASME-YGNN-08-0750-0800W3 v1.0 p.2 "Dynamic friction (maximal value)" Y | yes, only as value/s_F (force scale) | maximal spec value: upper bound / init |
| cb1 | 9 | Nm/(rad/s) | garcia2013 Table I (PDF pp. 12-13) | no (only cb1+cb2) | different machine |
| cb2 | 9 | Nm/(rad/s) | garcia2013 Table I (PDF pp. 12-13) | no (only cb1+cb2) | different machine |
| kb1 | 1987.5 | Nm/rad | garcia2013 Table I (PDF pp. 12-13) | no (only kb1+kb2) | different machine |
| kb2 | 1987.5 | Nm/rad | garcia2013 Table I (PDF pp. 12-13) | no (only kb1+kb2) | different machine |
| Lb | 0.725 | m | garcia2013 Table I (PDF pp. 12-13) | no (frame constant of P; fixed) | different machine |
| d | 0.1 | m | garcia2013 Table I (PDF pp. 12-13) | yes (combination d) | different machine |
| cc1 | 43 | N | Telica datasheet ASME-YGNN-08-0750-0800W3 v1.0 p.2 "Static friction (maximal value)" X, per motor | yes if velocity reverses on the rail, only as value/s_F (force scale) | static maximal: upper bound for kinetic cc (D-116 init) |
| cc2 | 43 | N | Telica datasheet ASME-YGNN-08-0750-0800W3 v1.0 p.2 "Static friction (maximal value)" X, per motor | yes if velocity reverses on the rail, only as value/s_F (force scale) | as cc1 |
| ccy | 49 | N | Telica datasheet ASME-YGNN-08-0750-0800W3 v1.0 p.2 "Static friction (maximal value)" Y | yes if velocity reverses on the rail, only as value/s_F (force scale) | as cc1 |
| V_BRK | 0.00225 | m/s | D-209: lee2020feeddrive Table 5 p.2839 (THK SSR20XW guide) | no (not identifiable from constant-velocity-free data) | a transfer; Karnopp band |
| V0_TANH | 0.001 | m/s | D-116 HEURISTIC (Makkar & Dixon 2005 form) | n/a (surrogate width) | differentiable surrogate only |
| Kt_X | 109 | N/Arms | Telica datasheet ASME-YGNN-08-0750-0800W3 v1.0 p.3 Kt; = Telica 1.mat MotorForceConst X | no (degenerate with s_F and masses) | per motor; one motor per logged X channel |
| Kt_Y | 77.6 | N/Arms | Telica datasheet ASME-YGNN-08-0750-0800W3 v1.0 p.3 Kt; = Telica 1.mat MotorForceConst Y | no (degenerate with s_F and masses) |  |
| s_F_X | 3.469 | - | D-112 HEURISTIC, reading (A): 91.0/26.229 | no (degenerate with masses) | reading (B): 1.0 with m_X,eff = 26.2 kg (TR-004) |
| s_F_Y | 3.202 | - | D-112 HEURISTIC, reading (A): 19.0/5.933 | no (degenerate with masses) | reading (B): 1.0 with mh,eff = 5.9 kg (TR-004) |
| Ts | 5e-05 | s | Telica 1.mat SamplingTime; = controller zpk Ts (D-073) | n/a | logs are 20 kHz native |

## Force scale: the two consistent readings (TR-004)
| Reading | Logged A -> N | Masses | Status |
|-|-|-|-|
| (A) default | Kt * [3.469, 3.469, 3.202] = [378.1, 378.1, 248.5] N/A | datasheet: X-moving 91.0 kg, mh 19.0 kg | D-112 HEURISTIC; factor 2 attributed to a forcer pair, ~sqrt(3) unexplained |
| (B) | Kt = [109, 109, 77.6] N/A | effective X 26.2 kg, Y 5.9 kg (3.5x below datasheet) | contradicts the datasheet masses; not exactly representable with s_X != s_Y |

The data fix only the ratios. The datasheet friction maxima (43 / 49 N, absolute newtons) are a
one-sided handle: recovered kinetic cc above them in (A) units contradicts (A).

Derived at the defaults: ten combinations kb_sum 3975, cg1 136, cg2 136, cy 98, cb_sum 18, mh 19,
m_total 72, m_diff -0.824, J_eff 5.575, d 0.1.
