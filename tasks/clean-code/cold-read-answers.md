# Cold read, answers (D-284 amendment 11, step 3)

Source: `cold-read-merged.md` with Dirk's ruling on the single-reader items ("Accept all", 2026-10-31). Item
ids are the readers' own (`co` Codex, `cl` Claude); the second column is the line in `final-step5a`, the
answer location is in the current `Clean/`. Comments and docstrings only: `clean-up-verification/comment_ast_check.py`
prints `AST_IDENTICAL 24/24`.

| Verdict (per reader item) | Count |
|-|-|
| Q answered | 287 |
| Q rejected | 59 |
| N deleted | 28 |
| N kept | 431 |

Comment lines per file, before (`final-step5a`) and after:

```
IDENTICAL gantry_dynamic/__init__.py  comments 2 -> 2, docstring lines 1 -> 1
IDENTICAL gantry_dynamic/config.py  comments 28 -> 65, docstring lines 23 -> 25
IDENTICAL gantry_dynamic/controller.py  comments 17 -> 20, docstring lines 87 -> 91
IDENTICAL gantry_dynamic/data.py  comments 15 -> 18, docstring lines 59 -> 70
IDENTICAL gantry_dynamic/evaluation.py  comments 9 -> 14, docstring lines 71 -> 75
IDENTICAL gantry_dynamic/model.py  comments 34 -> 47, docstring lines 46 -> 50
IDENTICAL gantry_dynamic/obc_gantry.py  comments 14 -> 14, docstring lines 95 -> 105
IDENTICAL gantry_dynamic/training.py  comments 26 -> 39, docstring lines 21 -> 37
IDENTICAL gantry_interconnect_dynamic.py  comments 14 -> 15, docstring lines 9 -> 10
IDENTICAL model_augmentation/__init__.py  comments 0 -> 0, docstring lines 3 -> 3
IDENTICAL model_augmentation/fit_systems/__init__.py  comments 0 -> 0, docstring lines 0 -> 0
IDENTICAL model_augmentation/fit_systems/blocks.py  comments 60 -> 64, docstring lines 250 -> 256
IDENTICAL model_augmentation/fit_systems/closed_loop.py  comments 28 -> 30, docstring lines 109 -> 112
IDENTICAL model_augmentation/fit_systems/interconnect.py  comments 58 -> 68, docstring lines 106 -> 115
IDENTICAL model_augmentation/fit_systems/lbfgs_polish.py  comments 20 -> 23, docstring lines 32 -> 37
IDENTICAL model_augmentation/fit_systems/obc.py  comments 15 -> 15, docstring lines 73 -> 77
IDENTICAL model_augmentation/fit_systems/pre_encoder.py  comments 17 -> 17, docstring lines 27 -> 30
IDENTICAL model_augmentation/fit_systems/ps2_opening.py  comments 42 -> 46, docstring lines 88 -> 93
IDENTICAL model_augmentation/systems/__init__.py  comments 0 -> 0, docstring lines 0 -> 0
IDENTICAL model_augmentation/systems/gantry_ss.py  comments 8 -> 18, docstring lines 28 -> 29
IDENTICAL model_augmentation/utils/__init__.py  comments 0 -> 0, docstring lines 0 -> 0
IDENTICAL model_augmentation/utils/deepSI_corrections.py  comments 5 -> 5, docstring lines 27 -> 30
IDENTICAL model_augmentation/utils/torch_nets.py  comments 7 -> 7, docstring lines 35 -> 37
IDENTICAL model_augmentation/utils/utils.py  comments 3 -> 3, docstring lines 21 -> 21
```

Residue for a later round (not a comment item): `least_squares_start` returns W and b, which its only caller
ignores (cl Q060, `gantry_dynamic/model.py:225` in `final-step5a`).

Step 5 note: comment lines above it moved the MM16 line of `gantry_dynamic/data.py` from line 218 to line 231;
its text is unchanged, so the `sed` of the server job needs the new line number.

```
cl Q001 | gantry_interconnect_dynamic.py:28 | answered at gantry_interconnect_dynamic.py:28
cl Q002 | gantry_interconnect_dynamic.py:40 | answered at gantry_dynamic/training.py:331 (agreed, G04)
cl Q003 | gantry_interconnect_dynamic.py:119 | answered at gantry_dynamic/controller.py:209
cl Q004 | gantry_interconnect_dynamic.py:129 | answered at gantry_dynamic/training.py:124 (agreed, G01)
cl Q005 | gantry_interconnect_dynamic.py:130 | answered at gantry_dynamic/training.py:289
cl Q006 | gantry_interconnect_dynamic.py:133 | answered at gantry_dynamic/training.py:292 (agreed, G02)
cl Q007 | gantry_interconnect_dynamic.py:138 | answered at gantry_interconnect_dynamic.py:135
cl Q008 | gantry_interconnect_dynamic.py:144 | answered at gantry_dynamic/config.py:86 (agreed, G05)
cl Q009 | gantry_interconnect_dynamic.py:155 | answered at gantry_interconnect_dynamic.py:154 (agreed, G03)
cl Q010 | gantry_dynamic/config.py:16 | answered at gantry_dynamic/config.py:16
cl Q011 | gantry_dynamic/config.py:20 | answered at gantry_dynamic/config.py:24
cl Q012 | gantry_dynamic/config.py:22 | answered at gantry_dynamic/config.py:22
cl Q013 | gantry_dynamic/config.py:25 | answered at gantry_dynamic/config.py:28 (agreed, G06)
cl Q014 | gantry_dynamic/config.py:27 | answered at gantry_dynamic/config.py:35
cl Q015 | gantry_dynamic/config.py:30 | answered at gantry_dynamic/config.py:78 (agreed, G07)
cl Q016 | gantry_dynamic/config.py:30 | answered at gantry_dynamic/config.py:73 (agreed, G08)
cl Q017 | gantry_dynamic/config.py:30 | answered at gantry_dynamic/config.py:76 (agreed, G09)
cl Q018 | gantry_dynamic/config.py:67 | answered at gantry_dynamic/data.py:24 (agreed, G10)
cl Q019 | gantry_dynamic/config.py:71 | answered at gantry_dynamic/config.py:89 (agreed, G11)
cl Q020 | gantry_dynamic/config.py:73 | answered at gantry_dynamic/config.py:91 (agreed, G12)
cl Q021 | gantry_dynamic/config.py:74 | answered at gantry_dynamic/config.py:93 (agreed, G13)
cl Q022 | gantry_dynamic/config.py:76 | answered at gantry_dynamic/config.py:96 (agreed, G14)
cl Q023 | gantry_dynamic/config.py:79 | answered at gantry_dynamic/config.py:99 (agreed, G15)
cl Q024 | gantry_dynamic/config.py:81 | answered at gantry_dynamic/config.py:101 (agreed, G16)
cl Q025 | gantry_dynamic/config.py:91 | answered at gantry_dynamic/config.py:118 (agreed, G18)
cl Q026 | gantry_dynamic/config.py:92 | answered at gantry_dynamic/config.py:118 (agreed, G18)
cl Q027 | gantry_dynamic/config.py:99 | answered at gantry_dynamic/config.py:128 (agreed, G19)
cl Q028 | gantry_dynamic/config.py:101 | answered at gantry_dynamic/config.py:129 (agreed, G20)
cl Q029 | gantry_dynamic/config.py:105 | answered at gantry_dynamic/config.py:139 (agreed, G21)
cl Q030 | gantry_dynamic/config.py:109 | answered at gantry_dynamic/config.py:141 (agreed, G68)
cl Q031 | gantry_dynamic/config.py:114 | answered at gantry_dynamic/config.py:148 (agreed, G22)
cl Q032 | gantry_dynamic/config.py:145 | answered at gantry_dynamic/config.py:180
cl Q033 | gantry_dynamic/config.py:166 | answered at gantry_dynamic/config.py:208 (agreed, G23)
cl Q034 | gantry_dynamic/controller.py:20 | answered at gantry_dynamic/controller.py:21
cl Q035 | gantry_dynamic/controller.py:20 | answered at gantry_dynamic/controller.py:3
cl Q036 | gantry_dynamic/controller.py:43 | answered at gantry_dynamic/controller.py:46 (agreed, G25)
cl Q037 | gantry_dynamic/controller.py:125 | answered at gantry_dynamic/controller.py:157 (agreed, G26)
cl Q038 | gantry_dynamic/data.py:3 | answered at gantry_dynamic/config.py:118 (agreed, G18)
cl Q039 | gantry_dynamic/data.py:23 | answered at gantry_dynamic/data.py:24 (agreed, G10)
cl Q040 | gantry_dynamic/data.py:25 | answered at gantry_dynamic/data.py:26 (agreed, G27)
cl Q041 | gantry_dynamic/data.py:36 | answered at gantry_dynamic/data.py:39 (agreed, G28)
cl Q042 | gantry_dynamic/data.py:58 | answered at gantry_dynamic/data.py:60 (agreed, G29)
cl Q043 | gantry_dynamic/data.py:77 | answered at model_augmentation/fit_systems/blocks.py:86 (agreed, G30)
cl Q044 | gantry_dynamic/data.py:112 | answered at gantry_dynamic/data.py:123 (agreed, G31)
cl Q045 | gantry_dynamic/data.py:120 | answered at gantry_dynamic/data.py:122
cl Q046 | gantry_dynamic/data.py:130 | answered at gantry_dynamic/data.py:104 (agreed, G32)
cl Q047 | gantry_dynamic/data.py:224 | answered at gantry_dynamic/data.py:236 (agreed, G33)
cl Q048 | gantry_dynamic/evaluation.py:66 | answered at gantry_dynamic/evaluation.py:69 (agreed, G34)
cl Q049 | gantry_dynamic/evaluation.py:245 | answered at gantry_dynamic/config.py:106 (agreed, G17)
cl Q050 | gantry_dynamic/model.py:31 | answered at gantry_dynamic/model.py:32
cl Q051 | gantry_dynamic/model.py:56 | answered at gantry_dynamic/model.py:52
cl Q052 | gantry_dynamic/model.py:78 | answered at gantry_dynamic/model.py:82 (agreed, G35)
cl Q053 | gantry_dynamic/model.py:90 | answered at gantry_dynamic/model.py:117 (agreed, G36)
cl Q054 | gantry_dynamic/model.py:117 | answered at gantry_dynamic/model.py:117 (agreed, G36)
cl Q055 | gantry_dynamic/model.py:157 | answered at gantry_dynamic/model.py:166 (agreed, G37)
cl Q056 | gantry_dynamic/model.py:159 | answered at model_augmentation/fit_systems/blocks.py:11 (agreed, G38)
cl Q057 | gantry_dynamic/model.py:173 | answered at gantry_dynamic/model.py:185 (agreed, G39)
cl Q058 | gantry_dynamic/model.py:185 | answered at gantry_dynamic/model.py:199
cl Q059 | gantry_dynamic/model.py:194 | answered at gantry_dynamic/model.py:210 (agreed, G40)
cl Q060 | gantry_dynamic/model.py:225 | rejected: the unused return is a code question (write-only output); flag as residue, not a comment
cl Q061 | gantry_dynamic/model.py:258 | answered at gantry_dynamic/model.py:272 (agreed, G41)
cl Q062 | gantry_dynamic/obc_gantry.py:4 | answered at gantry_dynamic/obc_gantry.py:9 (agreed, G42)
cl Q063 | gantry_dynamic/obc_gantry.py:35 | answered at gantry_dynamic/obc_gantry.py:40 (agreed, G43)
cl Q064 | gantry_dynamic/obc_gantry.py:59 | answered at gantry_dynamic/obc_gantry.py:67 (agreed, G44)
cl Q065 | gantry_dynamic/obc_gantry.py:59 | answered at gantry_dynamic/obc_gantry.py:67 (agreed, G44)
cl Q066 | gantry_dynamic/obc_gantry.py:86 | answered at gantry_dynamic/obc_gantry.py:49 (agreed, G45)
cl Q067 | gantry_dynamic/obc_gantry.py:119 | answered at gantry_dynamic/obc_gantry.py:150 (agreed, G46)
cl Q068 | gantry_dynamic/obc_gantry.py:258 | answered at gantry_dynamic/obc_gantry.py:266 (agreed, G47)
cl Q069 | gantry_dynamic/training.py:59 | answered at gantry_dynamic/training.py:250 (agreed, G49)
cl Q070 | gantry_dynamic/training.py:87 | answered at model_augmentation/fit_systems/blocks.py:500 (agreed, G50)
cl Q071 | gantry_dynamic/training.py:203 | answered at gantry_dynamic/training.py:214 (agreed, G51)
cl Q072 | gantry_dynamic/training.py:238 | answered at model_augmentation/fit_systems/ps2_opening.py:79
cl Q073 | gantry_dynamic/training.py:302 | answered at gantry_dynamic/training.py:325 (agreed, G52)
cl Q074 | model_augmentation/fit_systems/blocks.py:4 | answered at model_augmentation/fit_systems/blocks.py:6
cl Q075 | model_augmentation/fit_systems/blocks.py:234 | answered at model_augmentation/fit_systems/blocks.py:240 (agreed, G53)
cl Q076 | model_augmentation/fit_systems/blocks.py:365 | answered at model_augmentation/systems/gantry_ss.py:13 (agreed, G24)
cl Q077 | model_augmentation/fit_systems/blocks.py:439 | answered at model_augmentation/fit_systems/blocks.py:11 (agreed, G38)
cl Q078 | model_augmentation/fit_systems/blocks.py:491 | answered at model_augmentation/fit_systems/blocks.py:500 (agreed, G50)
cl Q079 | model_augmentation/fit_systems/closed_loop.py:124 | answered at model_augmentation/fit_systems/closed_loop.py:126 (agreed, G54)
cl Q080 | model_augmentation/fit_systems/closed_loop.py:204 | answered at model_augmentation/fit_systems/closed_loop.py:206 (agreed, G55)
cl Q081 | model_augmentation/fit_systems/closed_loop.py:254 | answered at model_augmentation/fit_systems/closed_loop.py:261 (agreed, G56)
cl Q082 | model_augmentation/fit_systems/interconnect.py:37 | answered at model_augmentation/fit_systems/interconnect.py:37 (agreed, G57)
cl Q083 | model_augmentation/fit_systems/interconnect.py:124 | answered at model_augmentation/fit_systems/interconnect.py:127 (agreed, G58)
cl Q084 | model_augmentation/fit_systems/interconnect.py:220 | answered at model_augmentation/fit_systems/interconnect.py:227 (agreed, G59)
cl Q085 | model_augmentation/fit_systems/interconnect.py:348 | answered at model_augmentation/fit_systems/interconnect.py:360 (agreed, G60)
cl Q086 | model_augmentation/fit_systems/interconnect.py:460 | answered at model_augmentation/fit_systems/interconnect.py:334 (agreed, G61)
cl Q087 | model_augmentation/fit_systems/lbfgs_polish.py:159 | answered at model_augmentation/fit_systems/lbfgs_polish.py:165 (agreed, G62)
cl Q088 | model_augmentation/fit_systems/lbfgs_polish.py:248 | answered at model_augmentation/fit_systems/lbfgs_polish.py:253
cl Q089 | model_augmentation/fit_systems/obc.py:178 | answered at model_augmentation/fit_systems/interconnect.py:623 (agreed, G48)
cl Q090 | model_augmentation/fit_systems/pre_encoder.py:36 | answered at model_augmentation/fit_systems/pre_encoder.py:37 (agreed, G63)
cl Q091 | model_augmentation/fit_systems/ps2_opening.py:37 | answered at model_augmentation/fit_systems/ps2_opening.py:38
cl Q092 | model_augmentation/fit_systems/ps2_opening.py:154 | answered at model_augmentation/fit_systems/ps2_opening.py:162 (agreed, G64)
cl Q093 | model_augmentation/systems/gantry_ss.py:1 | answered at model_augmentation/systems/gantry_ss.py:8 (agreed, G65)
cl Q094 | model_augmentation/utils/deepSI_corrections.py:14 | answered at model_augmentation/utils/deepSI_corrections.py:15 (agreed, G66)
cl Q095 | model_augmentation/utils/deepSI_corrections.py:18 | answered at model_augmentation/utils/deepSI_corrections.py:29 (agreed, G67)
co Q001 | gantry_interconnect_dynamic.py:129 | answered at gantry_dynamic/training.py:124 (agreed, G01)
co Q002 | gantry_interconnect_dynamic.py:133 | answered at gantry_dynamic/training.py:325 (agreed, G52)
co Q003 | gantry_interconnect_dynamic.py:134 | answered at gantry_dynamic/training.py:292 (agreed, G02)
co Q004 | gantry_interconnect_dynamic.py:151 | answered at gantry_interconnect_dynamic.py:40
co Q005 | gantry_interconnect_dynamic.py:156 | answered at gantry_interconnect_dynamic.py:154 (agreed, G03)
co Q006 | gantry_interconnect_dynamic.py:158 | answered at gantry_interconnect_dynamic.py:158
co Q007 | gantry_dynamic/config.py:23 | answered at gantry_dynamic/config.py:28 (agreed, G06)
co Q008 | gantry_dynamic/config.py:55 | answered at gantry_dynamic/config.py:28 (agreed, G06)
co Q009 | gantry_dynamic/config.py:63 | answered at gantry_dynamic/config.py:70
co Q010 | gantry_dynamic/config.py:64 | answered at gantry_dynamic/config.py:73 (agreed, G08)
co Q011 | gantry_dynamic/config.py:65 | answered at gantry_dynamic/config.py:76 (agreed, G09)
co Q012 | gantry_dynamic/config.py:66 | answered at gantry_dynamic/config.py:78 (agreed, G07)
co Q013 | gantry_dynamic/config.py:70 | answered at gantry_dynamic/config.py:86 (agreed, G05)
co Q014 | gantry_dynamic/config.py:71 | answered at gantry_dynamic/config.py:89 (agreed, G11)
co Q015 | gantry_dynamic/config.py:73 | answered at gantry_dynamic/config.py:91 (agreed, G12)
co Q016 | gantry_dynamic/config.py:74 | answered at gantry_dynamic/config.py:93 (agreed, G13)
co Q017 | gantry_dynamic/config.py:76 | answered at gantry_dynamic/config.py:96 (agreed, G14)
co Q018 | gantry_dynamic/config.py:79 | answered at gantry_dynamic/config.py:99 (agreed, G15)
co Q019 | gantry_dynamic/config.py:81 | answered at gantry_dynamic/config.py:101 (agreed, G16)
co Q020 | gantry_dynamic/config.py:82 | answered at gantry_dynamic/config.py:106 (agreed, G17)
co Q021 | gantry_dynamic/config.py:90 | answered at gantry_dynamic/config.py:115
co Q022 | gantry_dynamic/config.py:91 | answered at gantry_dynamic/config.py:118 (agreed, G18)
co Q023 | gantry_dynamic/config.py:97 | answered at gantry_dynamic/config.py:125
co Q024 | gantry_dynamic/config.py:99 | answered at gantry_dynamic/config.py:128 (agreed, G19)
co Q025 | gantry_dynamic/config.py:100 | answered at gantry_dynamic/config.py:129
co Q026 | gantry_dynamic/config.py:101 | answered at gantry_dynamic/config.py:129 (agreed, G20)
co Q027 | gantry_dynamic/config.py:104 | answered at gantry_dynamic/config.py:135
co Q028 | gantry_dynamic/config.py:105 | answered at gantry_dynamic/config.py:139 (agreed, G21)
co Q029 | gantry_dynamic/config.py:114 | answered at gantry_dynamic/config.py:148 (agreed, G22)
co Q030 | gantry_dynamic/config.py:166 | answered at gantry_dynamic/config.py:208 (agreed, G23)
co Q031 | gantry_dynamic/controller.py:21 | answered at model_augmentation/systems/gantry_ss.py:13 (agreed, G24)
co Q032 | gantry_dynamic/controller.py:22 | answered at model_augmentation/systems/gantry_ss.py:13 (agreed, G24)
co Q033 | gantry_dynamic/controller.py:23 | answered at model_augmentation/systems/gantry_ss.py:13 (agreed, G24)
co Q034 | gantry_dynamic/controller.py:29 | answered at gantry_dynamic/config.py:118 (agreed, G18)
co Q035 | gantry_dynamic/controller.py:36 | answered at gantry_dynamic/controller.py:46 (agreed, G25)
co Q036 | gantry_dynamic/controller.py:77 | rejected: the THEORY tag names the source of the pole placement
co Q037 | gantry_dynamic/controller.py:119 | answered at gantry_dynamic/controller.py:157 (agreed, G26)
co Q038 | gantry_dynamic/controller.py:144 | answered at gantry_dynamic/controller.py:150
co Q039 | gantry_dynamic/controller.py:151 | answered at gantry_dynamic/controller.py:157 (agreed, G26)
co Q040 | gantry_dynamic/controller.py:168 | answered at gantry_dynamic/controller.py:182
co Q041 | gantry_dynamic/controller.py:200 | answered at gantry_dynamic/controller.py:206
co Q042 | gantry_dynamic/controller.py:234 | answered at gantry_dynamic/controller.py:240
co Q043 | gantry_dynamic/controller.py:237 | rejected: the W_r formula (controller.py:236) and V (interconnect.py:619) state it
co Q044 | gantry_dynamic/data.py:23 | answered at gantry_dynamic/data.py:24 (agreed, G10)
co Q045 | gantry_dynamic/data.py:25 | answered at gantry_dynamic/data.py:26 (agreed, G27)
co Q046 | gantry_dynamic/data.py:30 | answered at gantry_dynamic/data.py:33
co Q047 | gantry_dynamic/data.py:36 | answered at gantry_dynamic/data.py:39 (agreed, G28)
co Q048 | gantry_dynamic/data.py:45 | answered at gantry_dynamic/data.py:47
co Q049 | gantry_dynamic/data.py:57 | answered at gantry_dynamic/data.py:60 (agreed, G29)
co Q050 | gantry_dynamic/data.py:76 | answered at model_augmentation/fit_systems/blocks.py:86 (agreed, G30)
co Q051 | gantry_dynamic/data.py:93 | answered at gantry_dynamic/data.py:104 (agreed, G32)
co Q052 | gantry_dynamic/data.py:108 | answered at gantry_dynamic/data.py:153
co Q053 | gantry_dynamic/data.py:112 | answered at gantry_dynamic/data.py:123 (agreed, G31)
co Q054 | gantry_dynamic/data.py:127 | rejected: internal helper with one caller
co Q055 | gantry_dynamic/data.py:140 | rejected: hypothetical use (records from several folders)
co Q056 | gantry_dynamic/data.py:157 | answered at gantry_dynamic/data.py:169
co Q057 | gantry_dynamic/data.py:203 | answered at gantry_dynamic/config.py:118 (agreed, G18)
co Q058 | gantry_dynamic/data.py:223 | answered at gantry_dynamic/data.py:236 (agreed, G33)
co Q059 | gantry_dynamic/data.py:230 | rejected: pooled statistics are the plain form; no reason to state
co Q060 | gantry_dynamic/data.py:231 | answered at gantry_dynamic/data.py:245 (the existing line, unchanged)
co Q061 | gantry_dynamic/evaluation.py:43 | answered at gantry_dynamic/evaluation.py:43
co Q062 | gantry_dynamic/evaluation.py:55 | answered at gantry_dynamic/evaluation.py:44
co Q063 | gantry_dynamic/evaluation.py:73 | answered at gantry_dynamic/evaluation.py:69 (agreed, G34)
co Q064 | gantry_dynamic/evaluation.py:98 | rejected: interface contract of an internal helper
co Q065 | gantry_dynamic/evaluation.py:101 | answered at model_augmentation/fit_systems/closed_loop.py:261 (agreed, G56)
co Q066 | gantry_dynamic/evaluation.py:103 | answered at gantry_dynamic/evaluation.py:106
co Q067 | gantry_dynamic/evaluation.py:181 | answered at gantry_dynamic/evaluation.py:185
co Q068 | gantry_dynamic/evaluation.py:205 | rejected: edge case of a resumed run
co Q069 | gantry_dynamic/evaluation.py:245 | answered at gantry_dynamic/config.py:106 (agreed, G17)
co Q070 | gantry_dynamic/evaluation.py:272 | answered at gantry_dynamic/evaluation.py:277
co Q071 | gantry_dynamic/evaluation.py:278 | answered at gantry_dynamic/evaluation.py:285
co Q072 | gantry_dynamic/evaluation.py:305 | answered at gantry_dynamic/config.py:208 (agreed, G23)
co Q073 | gantry_dynamic/model.py:28 | answered at gantry_dynamic/model.py:37
co Q074 | gantry_dynamic/model.py:32 | answered at gantry_dynamic/model.py:33
co Q075 | gantry_dynamic/model.py:65 | answered at gantry_dynamic/config.py:118 (agreed, G18)
co Q076 | gantry_dynamic/model.py:70 | rejected: activation choice is a thesis design choice
co Q077 | gantry_dynamic/model.py:78 | answered at gantry_dynamic/model.py:82 (agreed, G35)
co Q078 | gantry_dynamic/model.py:95 | answered at model_augmentation/fit_systems/pre_encoder.py:87
co Q079 | gantry_dynamic/model.py:111 | answered at gantry_dynamic/model.py:117 (agreed, G36)
co Q080 | gantry_dynamic/model.py:120 | answered at gantry_dynamic/model.py:127
co Q081 | gantry_dynamic/model.py:126 | rejected: interface contract of an internal helper
co Q082 | gantry_dynamic/model.py:157 | answered at gantry_dynamic/model.py:166 (agreed, G37)
co Q083 | gantry_dynamic/model.py:173 | answered at gantry_dynamic/model.py:185 (agreed, G39)
co Q084 | gantry_dynamic/model.py:181 | answered at model_augmentation/fit_systems/blocks.py:11 (agreed, G38)
co Q085 | gantry_dynamic/model.py:194 | answered at gantry_dynamic/model.py:210 (agreed, G40)
co Q086 | gantry_dynamic/model.py:196 | rejected: design choice; the code shows it
co Q087 | gantry_dynamic/model.py:206 | answered at gantry_dynamic/model.py:223
co Q088 | gantry_dynamic/model.py:213 | rejected: observability argument belongs to the thesis
co Q089 | gantry_dynamic/model.py:258 | answered at gantry_dynamic/model.py:272 (agreed, G41)
co Q090 | gantry_dynamic/model.py:274 | answered at model_augmentation/fit_systems/pre_encoder.py:37
co Q091 | gantry_dynamic/obc_gantry.py:5 | answered at gantry_dynamic/obc_gantry.py:9 (agreed, G42)
co Q092 | gantry_dynamic/obc_gantry.py:7 | answered at gantry_dynamic/obc_gantry.py:9 (agreed, G42)
co Q093 | gantry_dynamic/obc_gantry.py:36 | answered at gantry_dynamic/obc_gantry.py:40 (agreed, G43)
co Q094 | gantry_dynamic/obc_gantry.py:59 | answered at gantry_dynamic/obc_gantry.py:67 (agreed, G44)
co Q095 | gantry_dynamic/obc_gantry.py:60 | answered at gantry_dynamic/obc_gantry.py:67 (agreed, G44)
co Q096 | gantry_dynamic/obc_gantry.py:86 | answered at gantry_dynamic/obc_gantry.py:49 (agreed, G45)
co Q097 | gantry_dynamic/obc_gantry.py:87 | rejected: no method reason; the only reason is identical results to the thesis runs, which is history (see agreed noise obc_gantry.py:84)
co Q098 | gantry_dynamic/obc_gantry.py:112 | answered at gantry_dynamic/obc_gantry.py:114
co Q099 | gantry_dynamic/obc_gantry.py:138 | answered at model_augmentation/fit_systems/interconnect.py:623 (agreed, G48)
co Q100 | gantry_dynamic/obc_gantry.py:142 | answered at gantry_dynamic/obc_gantry.py:150 (agreed, G46)
co Q101 | gantry_dynamic/obc_gantry.py:165 | answered at gantry_dynamic/obc_gantry.py:67 (agreed, G44)
co Q102 | gantry_dynamic/obc_gantry.py:187 | rejected: hypothetical use
co Q103 | gantry_dynamic/obc_gantry.py:200 | answered at gantry_dynamic/obc_gantry.py:209
co Q104 | gantry_dynamic/obc_gantry.py:210 | answered at gantry_dynamic/obc_gantry.py:210
co Q105 | gantry_dynamic/obc_gantry.py:217 | rejected: hypothetical use
co Q106 | gantry_dynamic/obc_gantry.py:244 | rejected: interface contract of an internal helper
co Q107 | gantry_dynamic/obc_gantry.py:258 | answered at gantry_dynamic/obc_gantry.py:266 (agreed, G47)
co Q108 | gantry_dynamic/training.py:31 | answered at gantry_dynamic/training.py:33
co Q109 | gantry_dynamic/training.py:61 | answered at gantry_dynamic/training.py:325 (agreed, G52)
co Q110 | gantry_dynamic/training.py:63 | answered at gantry_dynamic/training.py:250 (agreed, G49)
co Q111 | gantry_dynamic/training.py:70 | rejected: interface contract
co Q112 | gantry_dynamic/training.py:87 | answered at model_augmentation/fit_systems/blocks.py:500 (agreed, G50)
co Q113 | gantry_dynamic/training.py:118 | answered at gantry_dynamic/config.py:106 (agreed, G17)
co Q114 | gantry_dynamic/training.py:144 | answered at gantry_dynamic/training.py:124 (agreed, G01)
co Q115 | gantry_dynamic/training.py:164 | answered at gantry_dynamic/training.py:172
co Q116 | gantry_dynamic/training.py:165 | answered at gantry_dynamic/training.py:173
co Q117 | gantry_dynamic/training.py:198 | rejected: selection design choice
co Q118 | gantry_dynamic/training.py:203 | answered at gantry_dynamic/training.py:214 (agreed, G51)
co Q119 | gantry_dynamic/training.py:220 | answered at gantry_dynamic/training.py:230
co Q120 | gantry_dynamic/training.py:229 | answered at gantry_dynamic/training.py:241
co Q121 | gantry_dynamic/training.py:238 | answered at gantry_dynamic/training.py:250 (agreed, G49)
co Q122 | gantry_dynamic/training.py:256 | answered at gantry_dynamic/config.py:96 (agreed, G14)
co Q123 | gantry_dynamic/training.py:264 | rejected: edge case; ps2_summary records it
co Q124 | gantry_dynamic/training.py:301 | answered at gantry_dynamic/training.py:325 (agreed, G52)
co Q125 | gantry_dynamic/training.py:306 | answered at gantry_dynamic/training.py:331 (agreed, G04)
co Q126 | gantry_dynamic/training.py:310 | rejected: answered by the code
co Q127 | gantry_dynamic/training.py:333 | answered at gantry_dynamic/training.py:360
co Q128 | gantry_dynamic/training.py:346 | answered at gantry_dynamic/training.py:369
co Q129 | model_augmentation/fit_systems/blocks.py:9 | rejected: units follow from the parameter units (agreed group)
co Q130 | model_augmentation/fit_systems/blocks.py:25 | rejected: Jan's base-class contract
co Q131 | model_augmentation/fit_systems/blocks.py:85 | answered at model_augmentation/fit_systems/blocks.py:86 (agreed, G30)
co Q132 | model_augmentation/fit_systems/blocks.py:119 | answered at model_augmentation/fit_systems/blocks.py:123
co Q133 | model_augmentation/fit_systems/blocks.py:137 | answered at model_augmentation/fit_systems/blocks.py:141
co Q134 | model_augmentation/fit_systems/blocks.py:140 | answered at gantry_dynamic/config.py:139 (agreed, G21)
co Q135 | model_augmentation/fit_systems/blocks.py:151 | answered at gantry_dynamic/config.py:118 (agreed, G18)
co Q136 | model_augmentation/fit_systems/blocks.py:193 | answered at model_augmentation/fit_systems/blocks.py:11 (agreed, G38)
co Q137 | model_augmentation/fit_systems/blocks.py:216 | answered at model_augmentation/fit_systems/blocks.py:221
co Q138 | model_augmentation/fit_systems/blocks.py:234 | answered at model_augmentation/fit_systems/blocks.py:240 (agreed, G53)
co Q139 | model_augmentation/fit_systems/blocks.py:246 | answered at model_augmentation/fit_systems/blocks.py:256
co Q140 | model_augmentation/fit_systems/blocks.py:365 | answered at model_augmentation/systems/gantry_ss.py:13 (agreed, G24)
co Q141 | model_augmentation/fit_systems/blocks.py:369 | rejected: hypothetical inputs
co Q142 | model_augmentation/fit_systems/blocks.py:477 | rejected: identifiability argument belongs to the thesis
co Q143 | model_augmentation/fit_systems/blocks.py:489 | answered at model_augmentation/fit_systems/blocks.py:498
co Q144 | model_augmentation/fit_systems/blocks.py:491 | answered at model_augmentation/fit_systems/blocks.py:500 (agreed, G50)
co Q145 | model_augmentation/fit_systems/blocks.py:496 | rejected: hypothetical inputs
co Q146 | model_augmentation/fit_systems/blocks.py:549 | rejected: hypothetical inputs
co Q147 | model_augmentation/fit_systems/closed_loop.py:5 | answered at model_augmentation/fit_systems/closed_loop.py:261 (agreed, G56)
co Q148 | model_augmentation/fit_systems/closed_loop.py:25 | rejected: built by controller.py; ordering is the agreed ordering group
co Q149 | model_augmentation/fit_systems/closed_loop.py:85 | answered at model_augmentation/fit_systems/blocks.py:11 (agreed, G38)
co Q150 | model_augmentation/fit_systems/closed_loop.py:91 | answered at model_augmentation/fit_systems/blocks.py:86 (agreed, G30)
co Q151 | model_augmentation/fit_systems/closed_loop.py:111 | answered at model_augmentation/fit_systems/closed_loop.py:112
co Q152 | model_augmentation/fit_systems/closed_loop.py:124 | answered at model_augmentation/fit_systems/closed_loop.py:126 (agreed, G54)
co Q153 | model_augmentation/fit_systems/closed_loop.py:148 | rejected: interface contract
co Q154 | model_augmentation/fit_systems/closed_loop.py:202 | answered at model_augmentation/fit_systems/interconnect.py:384
co Q155 | model_augmentation/fit_systems/closed_loop.py:204 | answered at model_augmentation/fit_systems/closed_loop.py:206 (agreed, G55)
co Q156 | model_augmentation/fit_systems/closed_loop.py:221 | rejected: check design, answered by the code
co Q157 | model_augmentation/fit_systems/closed_loop.py:239 | rejected: selection criterion is a thesis choice
co Q158 | model_augmentation/fit_systems/interconnect.py:37 | answered at model_augmentation/fit_systems/interconnect.py:37 (agreed, G57)
co Q159 | model_augmentation/fit_systems/interconnect.py:55 | answered at model_augmentation/fit_systems/blocks.py:11 (agreed, G38)
co Q160 | model_augmentation/fit_systems/interconnect.py:75 | rejected: Jan's design
co Q161 | model_augmentation/fit_systems/interconnect.py:96 | rejected: interface contract (Jan's)
co Q162 | model_augmentation/fit_systems/interconnect.py:118 | rejected: interface contract (Jan's)
co Q163 | model_augmentation/fit_systems/interconnect.py:124 | answered at model_augmentation/fit_systems/interconnect.py:127 (agreed, G58)
co Q164 | model_augmentation/fit_systems/interconnect.py:136 | answered at model_augmentation/fit_systems/interconnect.py:127 (agreed, G58)
co Q165 | model_augmentation/fit_systems/interconnect.py:142 | answered at model_augmentation/fit_systems/interconnect.py:127 (agreed, G58)
co Q166 | model_augmentation/fit_systems/interconnect.py:148 | rejected: Jan's graph; one wiring in build_model
co Q167 | model_augmentation/fit_systems/interconnect.py:164 | rejected: hypothetical: the model has one physics block
co Q168 | model_augmentation/fit_systems/interconnect.py:196 | answered at model_augmentation/fit_systems/blocks.py:86
co Q169 | model_augmentation/fit_systems/interconnect.py:220 | answered at model_augmentation/fit_systems/interconnect.py:227 (agreed, G59)
co Q170 | model_augmentation/fit_systems/interconnect.py:235 | rejected: Jan's internals
co Q171 | model_augmentation/fit_systems/interconnect.py:256 | rejected: Jan's API; our one call is the agreed connect-signals group
co Q172 | model_augmentation/fit_systems/interconnect.py:256 | rejected: Jan's API
co Q173 | model_augmentation/fit_systems/interconnect.py:256 | rejected: Jan's API
co Q174 | model_augmentation/fit_systems/interconnect.py:268 | rejected: Jan's API
co Q175 | model_augmentation/fit_systems/interconnect.py:324 | answered at model_augmentation/fit_systems/interconnect.py:384
co Q176 | model_augmentation/fit_systems/interconnect.py:333 | answered at model_augmentation/fit_systems/interconnect.py:334 (agreed, G61)
co Q177 | model_augmentation/fit_systems/interconnect.py:348 | answered at model_augmentation/fit_systems/interconnect.py:360 (agreed, G60)
co Q178 | model_augmentation/fit_systems/interconnect.py:360 | answered at gantry_dynamic/config.py:141 (agreed, G68)
co Q179 | model_augmentation/fit_systems/interconnect.py:365 | rejected: hypothetical use
co Q180 | model_augmentation/fit_systems/interconnect.py:370 | answered at model_augmentation/fit_systems/interconnect.py:384
co Q181 | model_augmentation/fit_systems/interconnect.py:442 | answered at model_augmentation/fit_systems/interconnect.py:336
co Q182 | model_augmentation/fit_systems/interconnect.py:452 | answered at model_augmentation/fit_systems/interconnect.py:466
co Q183 | model_augmentation/fit_systems/interconnect.py:478 | answered at gantry_dynamic/training.py:250
co Q184 | model_augmentation/fit_systems/interconnect.py:516 | answered at model_augmentation/fit_systems/interconnect.py:519
co Q185 | model_augmentation/fit_systems/interconnect.py:528 | answered at model_augmentation/fit_systems/interconnect.py:543
co Q186 | model_augmentation/fit_systems/interconnect.py:550 | answered at model_augmentation/fit_systems/interconnect.py:567
co Q187 | model_augmentation/fit_systems/interconnect.py:576 | answered at gantry_dynamic/obc_gantry.py:150 (agreed, G46)
co Q188 | model_augmentation/fit_systems/interconnect.py:606 | answered at model_augmentation/fit_systems/interconnect.py:623 (agreed, G48)
co Q189 | model_augmentation/fit_systems/interconnect.py:608 | rejected: gradient-path analysis is thesis depth
co Q190 | model_augmentation/fit_systems/lbfgs_polish.py:19 | answered at gantry_dynamic/config.py:99 (agreed, G15)
co Q191 | model_augmentation/fit_systems/lbfgs_polish.py:48 | answered at gantry_dynamic/config.py:101 (agreed, G16)
co Q192 | model_augmentation/fit_systems/lbfgs_polish.py:75 | answered at model_augmentation/fit_systems/lbfgs_polish.py:78
co Q193 | model_augmentation/fit_systems/lbfgs_polish.py:119 | rejected: the comment at lbfgs_polish.py:212 gives the reason
co Q194 | model_augmentation/fit_systems/lbfgs_polish.py:147 | answered at model_augmentation/fit_systems/lbfgs_polish.py:151
co Q195 | model_augmentation/fit_systems/lbfgs_polish.py:158 | answered at gantry_dynamic/config.py:141 (agreed, G68)
co Q196 | model_augmentation/fit_systems/lbfgs_polish.py:170 | answered at model_augmentation/fit_systems/lbfgs_polish.py:165 (agreed, G62)
co Q197 | model_augmentation/fit_systems/lbfgs_polish.py:255 | answered at gantry_dynamic/config.py:99 (agreed, G15)
co Q198 | model_augmentation/fit_systems/obc.py:2 | answered at gantry_dynamic/obc_gantry.py:9 (agreed, G42)
co Q199 | model_augmentation/fit_systems/obc.py:21 | rejected: hypothetical; the class is general
co Q200 | model_augmentation/fit_systems/obc.py:28 | answered at gantry_dynamic/obc_gantry.py:114
co Q201 | model_augmentation/fit_systems/obc.py:91 | rejected: interface contract
co Q202 | model_augmentation/fit_systems/obc.py:107 | rejected: gradient-path analysis, as Q189
co Q203 | model_augmentation/fit_systems/obc.py:129 | answered at model_augmentation/fit_systems/obc.py:131 (the existing line, unchanged)
co Q204 | model_augmentation/fit_systems/obc.py:155 | rejected: answered by the code
co Q205 | model_augmentation/fit_systems/obc.py:187 | rejected: interface contract
co Q206 | model_augmentation/fit_systems/pre_encoder.py:41 | answered at gantry_dynamic/config.py:129 (agreed, G20)
co Q207 | model_augmentation/fit_systems/pre_encoder.py:65 | answered at model_augmentation/fit_systems/pre_encoder.py:37 (agreed, G63)
co Q208 | model_augmentation/fit_systems/pre_encoder.py:67 | answered at model_augmentation/fit_systems/pre_encoder.py:37 (agreed, G63)
co Q209 | model_augmentation/fit_systems/pre_encoder.py:74 | rejected: architecture choice of the thesis
co Q210 | model_augmentation/fit_systems/pre_encoder.py:76 | answered at model_augmentation/fit_systems/pre_encoder.py:37 (agreed, G63)
co Q211 | model_augmentation/fit_systems/pre_encoder.py:84 | answered at gantry_dynamic/config.py:129 (agreed, G20)
co Q212 | model_augmentation/fit_systems/pre_encoder.py:89 | answered at model_augmentation/fit_systems/pre_encoder.py:90
co Q213 | model_augmentation/fit_systems/pre_encoder.py:102 | answered at model_augmentation/fit_systems/pre_encoder.py:104
co Q214 | model_augmentation/fit_systems/ps2_opening.py:5 | answered at model_augmentation/fit_systems/ps2_opening.py:162 (agreed, G64)
co Q215 | model_augmentation/fit_systems/ps2_opening.py:22 | answered at model_augmentation/fit_systems/ps2_opening.py:22
co Q216 | model_augmentation/fit_systems/ps2_opening.py:24 | answered at model_augmentation/fit_systems/ps2_opening.py:25
co Q217 | model_augmentation/fit_systems/ps2_opening.py:26 | rejected: hypothetical settings
co Q218 | model_augmentation/fit_systems/ps2_opening.py:42 | answered at model_augmentation/fit_systems/ps2_opening.py:45
co Q219 | model_augmentation/fit_systems/ps2_opening.py:53 | answered at model_augmentation/fit_systems/ps2_opening.py:58
co Q220 | model_augmentation/fit_systems/ps2_opening.py:54 | answered at model_augmentation/fit_systems/ps2_opening.py:59
co Q221 | model_augmentation/fit_systems/ps2_opening.py:73 | answered at gantry_dynamic/config.py:79
co Q222 | model_augmentation/fit_systems/ps2_opening.py:89 | answered at model_augmentation/fit_systems/ps2_opening.py:148
co Q223 | model_augmentation/fit_systems/ps2_opening.py:103 | rejected: PS design choice
co Q224 | model_augmentation/fit_systems/ps2_opening.py:149 | rejected: PS design choice (HEURISTIC)
co Q225 | model_augmentation/fit_systems/ps2_opening.py:181 | answered at model_augmentation/fit_systems/ps2_opening.py:180
co Q226 | model_augmentation/fit_systems/ps2_opening.py:204 | rejected: PS design choice
co Q227 | model_augmentation/fit_systems/ps2_opening.py:243 | rejected: PS design choice
co Q228 | model_augmentation/fit_systems/ps2_opening.py:279 | rejected: PS design choice
co Q229 | model_augmentation/fit_systems/ps2_opening.py:288 | rejected: PS design choice
co Q230 | model_augmentation/fit_systems/ps2_opening.py:301 | rejected: the formula comment at line 301 states it
co Q231 | model_augmentation/fit_systems/ps2_opening.py:305 | rejected: PS design choice
co Q232 | model_augmentation/systems/gantry_ss.py:1 | answered at gantry_dynamic/data.py:24 (agreed, G10)
co Q233 | model_augmentation/systems/gantry_ss.py:8 | answered at model_augmentation/systems/gantry_ss.py:8 (agreed, G65)
co Q234 | model_augmentation/systems/gantry_ss.py:11 | answered at model_augmentation/systems/gantry_ss.py:13 (agreed, G24)
co Q235 | model_augmentation/systems/gantry_ss.py:16 | answered at model_augmentation/systems/gantry_ss.py:13 (agreed, G24)
co Q236 | model_augmentation/systems/gantry_ss.py:19 | answered at model_augmentation/systems/gantry_ss.py:13 (agreed, G24)
co Q237 | model_augmentation/systems/gantry_ss.py:23 | answered at model_augmentation/systems/gantry_ss.py:13 (agreed, G24)
co Q238 | model_augmentation/systems/gantry_ss.py:26 | answered at model_augmentation/systems/gantry_ss.py:13 (agreed, G24)
co Q239 | model_augmentation/systems/gantry_ss.py:29 | answered at model_augmentation/systems/gantry_ss.py:13 (agreed, G24)
co Q240 | model_augmentation/systems/gantry_ss.py:32 | answered at gantry_dynamic/config.py:118 (agreed, G18)
co Q241 | model_augmentation/systems/gantry_ss.py:46 | rejected: hypothetical inputs
co Q242 | model_augmentation/systems/gantry_ss.py:59 | answered at gantry_dynamic/controller.py:46 (agreed, G25)
co Q243 | model_augmentation/systems/gantry_ss.py:86 | answered at model_augmentation/fit_systems/blocks.py:240 (agreed, G53)
co Q244 | model_augmentation/systems/gantry_ss.py:95 | answered at model_augmentation/systems/gantry_ss.py:105
co Q245 | model_augmentation/systems/gantry_ss.py:98 | answered at model_augmentation/systems/gantry_ss.py:108
co Q246 | model_augmentation/utils/deepSI_corrections.py:14 | answered at model_augmentation/utils/deepSI_corrections.py:15 (agreed, G66)
co Q247 | model_augmentation/utils/deepSI_corrections.py:18 | answered at model_augmentation/fit_systems/interconnect.py:360 (agreed, G60)
co Q248 | model_augmentation/utils/deepSI_corrections.py:19 | answered at model_augmentation/utils/deepSI_corrections.py:29 (agreed, G67)
co Q249 | model_augmentation/utils/torch_nets.py:16 | answered at model_augmentation/utils/torch_nets.py:16
co Q250 | model_augmentation/utils/torch_nets.py:44 | answered at model_augmentation/fit_systems/pre_encoder.py:37 (agreed, G63)
co Q251 | model_augmentation/utils/torch_nets.py:72 | answered at model_augmentation/utils/torch_nets.py:73
cl N001 | gantry_interconnect_dynamic.py:22 | deleted (agreed)
cl N002 | gantry_interconnect_dynamic.py:80 | deleted (agreed)
cl N003 | gantry_interconnect_dynamic.py:112 | deleted (agreed)
cl N004 | gantry_dynamic/config.py:72 | kept: label, shortened to its source tag
cl N005 | gantry_dynamic/config.py:75 | kept: label, shortened to its source tag
cl N006 | gantry_dynamic/config.py:96 | kept: label, shortened to its source tag
cl N007 | gantry_dynamic/config.py:98 | kept: label, shortened to its source tag
cl N008 | gantry_dynamic/config.py:102 | kept: label, shortened to its source tag
cl N009 | gantry_dynamic/config.py:106 | kept: label, shortened to its source tag
cl N010 | gantry_dynamic/config.py:159 | kept: docstring content required by amendment 7
cl N011 | gantry_dynamic/config.py:174 | kept: label, shortened to its source tag
cl N012 | gantry_dynamic/controller.py:67 | kept: formula comment above its step (amendment 4)
cl N013 | gantry_dynamic/controller.py:83 | kept: formula comment above its step (amendment 4)
cl N014 | gantry_dynamic/controller.py:108 | kept: label, shortened to its source tag
cl N015 | gantry_dynamic/controller.py:236 | kept: formula comment above its step (amendment 4)
cl N016 | gantry_dynamic/data.py:123 | kept: formula comment above its step (amendment 4)
cl N017 | gantry_dynamic/data.py:162 | kept: formula comment above its step (amendment 4)
cl N018 | gantry_dynamic/data.py:221 | kept: formula comment above its step (amendment 4)
cl N019 | gantry_dynamic/data.py:230 | kept: formula comment above its step (amendment 4)
cl N020 | gantry_dynamic/data.py:239 | kept: numbered phase header (amendment 7)
cl N021 | gantry_dynamic/model.py:235 | kept: formula comment above its step (amendment 4)
cl N022 | gantry_dynamic/obc_gantry.py:84 | deleted the history clause (agreed)
cl N023 | gantry_dynamic/obc_gantry.py:128 | kept: formula comment above its step (amendment 4)
cl N024 | gantry_dynamic/obc_gantry.py:178 | kept: formula comment above its step (amendment 4)
cl N025 | gantry_dynamic/obc_gantry.py:209 | deleted the pointer clause; formula kept (agreed)
cl N026 | gantry_dynamic/training.py:317 | deleted (agreed)
cl N027 | model_augmentation/fit_systems/blocks.py:12 | kept: CHANGED / @added / origin marker or Jan attribution
cl N028 | model_augmentation/fit_systems/blocks.py:17 | kept: CHANGED / @added / origin marker or Jan attribution
cl N029 | model_augmentation/fit_systems/blocks.py:84 | kept: CHANGED / @added / origin marker or Jan attribution
cl N030 | model_augmentation/fit_systems/blocks.py:110 | kept: CHANGED / @added / origin marker or Jan attribution
cl N031 | model_augmentation/fit_systems/blocks.py:121 | kept: CHANGED / @added / origin marker or Jan attribution
cl N032 | model_augmentation/fit_systems/blocks.py:148 | deleted (agreed)
cl N033 | model_augmentation/fit_systems/blocks.py:176 | kept: formula comment above its step (amendment 4)
cl N034 | model_augmentation/fit_systems/blocks.py:200 | kept: numbered phase header (amendment 7)
cl N035 | model_augmentation/fit_systems/blocks.py:211 | kept: formula comment above its step (amendment 4)
cl N036 | model_augmentation/fit_systems/blocks.py:242 | kept: numbered phase header (amendment 7)
cl N037 | model_augmentation/fit_systems/blocks.py:266 | kept: numbered phase header (amendment 7)
cl N038 | model_augmentation/fit_systems/blocks.py:272 | kept: numbered phase header (amendment 7)
cl N039 | model_augmentation/fit_systems/blocks.py:288 | kept: formula comment above its step (amendment 4)
cl N040 | model_augmentation/fit_systems/blocks.py:292 | kept: formula comment above its step (amendment 4)
cl N041 | model_augmentation/fit_systems/blocks.py:303 | kept: formula comment above its step (amendment 4)
cl N042 | model_augmentation/fit_systems/blocks.py:309 | kept: formula comment above its step (amendment 4)
cl N043 | model_augmentation/fit_systems/blocks.py:326 | kept: numbered phase header (amendment 7)
cl N044 | model_augmentation/fit_systems/blocks.py:346 | kept: formula comment above its step (amendment 4)
cl N045 | model_augmentation/fit_systems/blocks.py:356 | kept: formula comment above its step (amendment 4)
cl N046 | model_augmentation/fit_systems/blocks.py:395 | kept: formula comment above its step (amendment 4)
cl N047 | model_augmentation/fit_systems/blocks.py:401 | deleted (agreed)
cl N048 | model_augmentation/fit_systems/blocks.py:421 | kept: formula comment above its step (amendment 4)
cl N049 | model_augmentation/fit_systems/blocks.py:427 | kept: formula comment above its step (amendment 4)
cl N050 | model_augmentation/fit_systems/blocks.py:600 | kept: formula comment above its step (amendment 4)
cl N051 | model_augmentation/fit_systems/blocks.py:602 | kept: formula comment above its step (amendment 4)
cl N052 | model_augmentation/fit_systems/blocks.py:624 | kept: formula comment above its step (amendment 4)
cl N053 | model_augmentation/fit_systems/blocks.py:630 | kept: formula comment above its step (amendment 4)
cl N054 | model_augmentation/fit_systems/closed_loop.py:69 | kept: formula comment above its step (amendment 4)
cl N055 | model_augmentation/fit_systems/closed_loop.py:93 | kept: formula comment above its step (amendment 4)
cl N056 | model_augmentation/fit_systems/closed_loop.py:95 | kept: formula comment above its step (amendment 4)
cl N057 | model_augmentation/fit_systems/closed_loop.py:97 | kept: formula comment above its step (amendment 4)
cl N058 | model_augmentation/fit_systems/closed_loop.py:311 | kept: formula comment above its step (amendment 4)
cl N059 | model_augmentation/fit_systems/interconnect.py:2 | kept: CHANGED / @added / origin marker or Jan attribution
cl N060 | model_augmentation/fit_systems/interconnect.py:9 | kept: CHANGED / @added / origin marker or Jan attribution
cl N061 | model_augmentation/fit_systems/interconnect.py:18 | kept: CHANGED / @added / origin marker or Jan attribution
cl N062 | model_augmentation/fit_systems/interconnect.py:332 | deleted (agreed)
cl N063 | model_augmentation/fit_systems/interconnect.py:383 | kept: CHANGED / @added / origin marker or Jan attribution
cl N064 | model_augmentation/fit_systems/interconnect.py:428 | kept: CHANGED / @added / origin marker or Jan attribution
cl N065 | model_augmentation/fit_systems/interconnect.py:447 | kept: CHANGED / @added / origin marker or Jan attribution
cl N066 | model_augmentation/fit_systems/interconnect.py:537 | deleted (agreed)
cl N067 | model_augmentation/fit_systems/interconnect.py:619 | kept: formula comment above its step (amendment 4)
cl N068 | model_augmentation/fit_systems/obc.py:34 | kept: numbered phase header (amendment 7)
cl N069 | model_augmentation/fit_systems/obc.py:39 | kept: formula comment above its step (amendment 4)
cl N070 | model_augmentation/fit_systems/obc.py:65 | kept: formula comment above its step (amendment 4)
cl N071 | model_augmentation/fit_systems/obc.py:113 | kept: formula comment above its step (amendment 4)
cl N072 | model_augmentation/fit_systems/obc.py:121 | kept: formula comment above its step (amendment 4)
cl N073 | model_augmentation/fit_systems/pre_encoder.py:6 | kept: CHANGED / @added / origin marker or Jan attribution
cl N074 | model_augmentation/fit_systems/pre_encoder.py:12 | kept: CHANGED / @added / origin marker or Jan attribution
cl N075 | model_augmentation/fit_systems/pre_encoder.py:55 | deleted (agreed)
cl N076 | model_augmentation/fit_systems/pre_encoder.py:94 | kept: formula comment above its step (amendment 4)
cl N077 | model_augmentation/fit_systems/pre_encoder.py:102 | kept: formula comment above its step (amendment 4)
cl N078 | model_augmentation/fit_systems/ps2_opening.py:214 | kept: formula comment above its step (amendment 4)
cl N079 | model_augmentation/fit_systems/ps2_opening.py:216 | kept: formula comment above its step (amendment 4)
cl N080 | model_augmentation/fit_systems/ps2_opening.py:238 | kept: formula comment above its step (amendment 4)
cl N081 | model_augmentation/fit_systems/ps2_opening.py:240 | kept: formula comment above its step (amendment 4)
cl N082 | model_augmentation/systems/gantry_ss.py:2 | deleted
cl N083 | model_augmentation/systems/gantry_ss.py:10 | kept: label, shortened to its source tag
cl N084 | model_augmentation/systems/gantry_ss.py:60 | kept: formula comment above its step (amendment 4)
cl N085 | model_augmentation/systems/gantry_ss.py:105 | kept: formula comment above its step (amendment 4)
cl N086 | model_augmentation/systems/gantry_ss.py:112 | kept: formula comment above its step (amendment 4)
cl N087 | model_augmentation/utils/deepSI_corrections.py:5 | kept: CHANGED / @added / origin marker or Jan attribution
cl N088 | model_augmentation/utils/deepSI_corrections.py:69 | kept: CHANGED / @added / origin marker or Jan attribution
cl N089 | model_augmentation/utils/deepSI_corrections.py:75 | kept: formula comment above its step (amendment 4)
cl N090 | model_augmentation/utils/torch_nets.py:6 | kept: CHANGED / @added / origin marker or Jan attribution
cl N091 | model_augmentation/utils/torch_nets.py:11 | kept: CHANGED / @added / origin marker or Jan attribution
cl N092 | model_augmentation/utils/torch_nets.py:37 | kept: formula comment above its step (amendment 4)
cl N093 | model_augmentation/utils/torch_nets.py:61 | kept: formula comment above its step (amendment 4)
cl N094 | model_augmentation/utils/utils.py:2 | kept: CHANGED / @added / origin marker or Jan attribution
cl N095 | model_augmentation/utils/utils.py:15 | kept: CHANGED / @added / origin marker or Jan attribution
co N001 | gantry_interconnect_dynamic.py:22 | deleted (agreed)
co N002 | gantry_interconnect_dynamic.py:35 | deleted
co N003 | gantry_interconnect_dynamic.py:78 | kept: numbered phase header (amendment 7)
co N004 | gantry_interconnect_dynamic.py:80 | deleted (agreed)
co N005 | gantry_interconnect_dynamic.py:99 | kept: numbered phase header (amendment 7)
co N006 | gantry_interconnect_dynamic.py:111 | kept: numbered phase header (amendment 7)
co N007 | gantry_interconnect_dynamic.py:112 | deleted (agreed)
co N008 | gantry_interconnect_dynamic.py:122 | kept: numbered phase header (amendment 7)
co N009 | gantry_interconnect_dynamic.py:132 | kept: numbered phase header (amendment 7)
co N010 | gantry_interconnect_dynamic.py:136 | kept: numbered phase header (amendment 7)
co N011 | gantry_interconnect_dynamic.py:143 | kept: docstring content required by amendment 7
co N012 | gantry_dynamic/config.py:1 | kept: docstring content required by amendment 7
co N013 | gantry_dynamic/config.py:41 | kept: says the dict maps fields to config.json keys and who reads it
co N014 | gantry_dynamic/config.py:62 | kept: docstring content required by amendment 7
co N015 | gantry_dynamic/config.py:68 | kept: section header; says where the values change
co N016 | gantry_dynamic/config.py:89 | kept: section header; says which values config.json records
co N017 | gantry_dynamic/config.py:127 | kept: docstring content required by amendment 7
co N018 | gantry_dynamic/config.py:159 | kept: docstring content required by amendment 7
co N019 | gantry_dynamic/config.py:174 | kept: label, shortened to its source tag
co N020 | gantry_dynamic/config.py:203 | kept: docstring content required by amendment 7
co N021 | gantry_dynamic/controller.py:67 | kept: formula comment above its step (amendment 4)
co N022 | gantry_dynamic/controller.py:83 | kept: formula comment above its step (amendment 4)
co N023 | gantry_dynamic/controller.py:101 | kept: numbered phase header (amendment 7)
co N024 | gantry_dynamic/controller.py:107 | kept: numbered phase header (amendment 7)
co N025 | gantry_dynamic/controller.py:108 | kept: label, shortened to its source tag
co N026 | gantry_dynamic/controller.py:110 | kept: numbered phase header (amendment 7)
co N027 | gantry_dynamic/controller.py:208 | kept: numbered phase header (amendment 7)
co N028 | gantry_dynamic/controller.py:224 | kept: numbered phase header (amendment 7)
co N029 | gantry_dynamic/controller.py:233 | kept: numbered phase header (amendment 7)
co N030 | gantry_dynamic/controller.py:236 | kept: formula comment above its step (amendment 4)
co N031 | gantry_dynamic/data.py:22 | kept: not history, says where the folders sit (Claude Q039 extends it)
co N032 | gantry_dynamic/data.py:24 | deleted; replaced by the test-I/test-E answer
co N033 | gantry_dynamic/data.py:26 | kept: the name SOS does not say anti-aliasing filter
co N034 | gantry_dynamic/data.py:123 | kept: formula comment above its step (amendment 4)
co N035 | gantry_dynamic/data.py:162 | kept: formula comment above its step (amendment 4)
co N036 | gantry_dynamic/data.py:171 | kept: docstring content required by amendment 7
co N037 | gantry_dynamic/data.py:178 | kept: docstring content required by amendment 7
co N038 | gantry_dynamic/data.py:214 | kept: numbered phase header (amendment 7)
co N039 | gantry_dynamic/data.py:221 | kept: formula comment above its step (amendment 4)
co N040 | gantry_dynamic/data.py:223 | kept: formula comment above its step (amendment 4)
co N041 | gantry_dynamic/data.py:229 | kept: numbered phase header (amendment 7)
co N042 | gantry_dynamic/data.py:230 | kept: formula comment above its step (amendment 4)
co N043 | gantry_dynamic/data.py:239 | kept: numbered phase header (amendment 7)
co N044 | gantry_dynamic/evaluation.py:36 | kept: TEX does not say what it is for
co N045 | gantry_dynamic/evaluation.py:66 | kept: docstring content required by amendment 7
co N046 | gantry_dynamic/evaluation.py:236 | kept: numbered phase header (amendment 7)
co N047 | gantry_dynamic/evaluation.py:266 | kept: numbered phase header (amendment 7)
co N048 | gantry_dynamic/evaluation.py:303 | kept: numbered phase header (amendment 7)
co N049 | gantry_dynamic/evaluation.py:313 | kept: numbered phase header (amendment 7)
co N050 | gantry_dynamic/model.py:45 | kept: numbered phase header (amendment 7)
co N051 | gantry_dynamic/model.py:48 | kept: formula comment above its step (amendment 4)
co N052 | gantry_dynamic/model.py:49 | kept: formula comment above its step (amendment 4)
co N053 | gantry_dynamic/model.py:63 | kept: numbered phase header (amendment 7)
co N054 | gantry_dynamic/model.py:74 | kept: numbered phase header (amendment 7)
co N055 | gantry_dynamic/model.py:79 | kept: the routing of expansion_matrix is not visible in the call
co N056 | gantry_dynamic/model.py:87 | kept: numbered phase header (amendment 7)
co N057 | gantry_dynamic/model.py:105 | kept: numbered phase header (amendment 7)
co N058 | gantry_dynamic/model.py:106 | kept: the reason for refusing the CPU fallback
co N059 | gantry_dynamic/model.py:161 | kept: numbered phase header (amendment 7)
co N060 | gantry_dynamic/model.py:200 | kept: numbered phase header (amendment 7)
co N061 | gantry_dynamic/model.py:211 | kept: docstring content required by amendment 7
co N062 | gantry_dynamic/model.py:227 | kept: numbered phase header (amendment 7)
co N063 | gantry_dynamic/model.py:240 | kept: says what the driver setting is for
co N064 | gantry_dynamic/model.py:241 | kept: formula comment above its step (amendment 4)
co N065 | gantry_dynamic/model.py:248 | kept: numbered phase header (amendment 7)
co N066 | gantry_dynamic/model.py:260 | kept: numbered phase header (amendment 7)
co N067 | gantry_dynamic/model.py:261 | kept: numbered phase header (amendment 7)
co N068 | gantry_dynamic/obc_gantry.py:29 | kept: says what P_INV_T64 is
co N069 | gantry_dynamic/obc_gantry.py:77 | kept: numbered phase header (amendment 7)
co N070 | gantry_dynamic/obc_gantry.py:80 | kept: float64 and resampling are not visible in loadmat
co N071 | gantry_dynamic/obc_gantry.py:83 | kept: formula comment above its step (amendment 4)
co N072 | gantry_dynamic/obc_gantry.py:84 | deleted the history clause (agreed)
co N073 | gantry_dynamic/obc_gantry.py:91 | kept: numbered phase header (amendment 7)
co N074 | gantry_dynamic/obc_gantry.py:124 | kept: the row-to-column invariant the projection relies on
co N075 | gantry_dynamic/obc_gantry.py:128 | kept: formula comment above its step (amendment 4)
co N076 | gantry_dynamic/obc_gantry.py:134 | kept: formula comment above its step (amendment 4)
co N077 | gantry_dynamic/obc_gantry.py:175 | kept: formula comment above its step (amendment 4)
co N078 | gantry_dynamic/obc_gantry.py:178 | kept: formula comment above its step (amendment 4)
co N079 | gantry_dynamic/obc_gantry.py:183 | kept: docstring content required by amendment 7
co N080 | gantry_dynamic/obc_gantry.py:204 | kept: docstring content required by amendment 7
co N081 | gantry_dynamic/obc_gantry.py:209 | deleted the pointer clause; formula kept (agreed)
co N082 | gantry_dynamic/obc_gantry.py:228 | kept: says what the cache holds and its key
co N083 | gantry_dynamic/obc_gantry.py:232 | kept: docstring content required by amendment 7
co N084 | gantry_dynamic/training.py:19 | kept: docstring content required by amendment 7
co N085 | gantry_dynamic/training.py:86 | kept: docstring content required by amendment 7
co N086 | gantry_dynamic/training.py:109 | kept: docstring content required by amendment 7
co N087 | gantry_dynamic/training.py:154 | kept: docstring content required by amendment 7
co N088 | gantry_dynamic/training.py:171 | kept: docstring content required by amendment 7
co N089 | gantry_dynamic/training.py:183 | kept: docstring content required by amendment 7
co N090 | gantry_dynamic/training.py:248 | deleted
co N091 | gantry_dynamic/training.py:271 | kept: numbered phase header (amendment 7)
co N092 | gantry_dynamic/training.py:295 | kept: numbered phase header (amendment 7)
co N093 | gantry_dynamic/training.py:304 | kept: numbered phase header (amendment 7)
co N094 | gantry_dynamic/training.py:314 | kept: numbered phase header (amendment 7)
co N095 | gantry_dynamic/training.py:317 | deleted (agreed)
co N096 | gantry_dynamic/training.py:325 | kept: numbered phase header (amendment 7)
co N097 | model_augmentation/__init__.py:1 | kept: CHANGED / @added / origin marker or Jan attribution
co N098 | model_augmentation/__init__.py:2 | kept: CHANGED / @added / origin marker or Jan attribution
co N099 | model_augmentation/__init__.py:3 | kept: CHANGED / @added / origin marker or Jan attribution
co N100 | model_augmentation/fit_systems/blocks.py:1 | kept: docstring content required by amendment 7
co N101 | model_augmentation/fit_systems/blocks.py:12 | kept: CHANGED / @added / origin marker or Jan attribution
co N102 | model_augmentation/fit_systems/blocks.py:13 | kept: CHANGED / @added / origin marker or Jan attribution
co N103 | model_augmentation/fit_systems/blocks.py:17 | kept: CHANGED / @added / origin marker or Jan attribution
co N104 | model_augmentation/fit_systems/blocks.py:23 | kept: Jan's own docstring
co N105 | model_augmentation/fit_systems/blocks.py:26 | kept: docstring content required by amendment 7
co N106 | model_augmentation/fit_systems/blocks.py:36 | kept: docstring content required by amendment 7
co N107 | model_augmentation/fit_systems/blocks.py:52 | kept: CHANGED / @added / origin marker or Jan attribution
co N108 | model_augmentation/fit_systems/blocks.py:61 | kept: CHANGED / @added / origin marker or Jan attribution
co N109 | model_augmentation/fit_systems/blocks.py:62 | kept: CHANGED / @added / origin marker or Jan attribution
co N110 | model_augmentation/fit_systems/blocks.py:66 | kept: docstring content required by amendment 7
co N111 | model_augmentation/fit_systems/blocks.py:84 | kept: CHANGED / @added / origin marker or Jan attribution
co N112 | model_augmentation/fit_systems/blocks.py:86 | kept: docstring content required by amendment 7
co N113 | model_augmentation/fit_systems/blocks.py:100 | kept: docstring content required by amendment 7
co N114 | model_augmentation/fit_systems/blocks.py:110 | kept: CHANGED / @added / origin marker or Jan attribution
co N115 | model_augmentation/fit_systems/blocks.py:118 | kept: docstring content required by amendment 7
co N116 | model_augmentation/fit_systems/blocks.py:120 | kept: docstring content required by amendment 7
co N117 | model_augmentation/fit_systems/blocks.py:121 | kept: CHANGED / @added / origin marker or Jan attribution
co N118 | model_augmentation/fit_systems/blocks.py:131 | kept: docstring content required by amendment 7
co N119 | model_augmentation/fit_systems/blocks.py:148 | deleted (agreed)
co N120 | model_augmentation/fit_systems/blocks.py:154 | kept: numbered phase header (amendment 7)
co N121 | model_augmentation/fit_systems/blocks.py:162 | kept: docstring content required by amendment 7
co N122 | model_augmentation/fit_systems/blocks.py:176 | kept: formula comment above its step (amendment 4)
co N123 | model_augmentation/fit_systems/blocks.py:185 | kept: docstring content required by amendment 7
co N124 | model_augmentation/fit_systems/blocks.py:200 | kept: numbered phase header (amendment 7)
co N125 | model_augmentation/fit_systems/blocks.py:207 | kept: numbered phase header (amendment 7)
co N126 | model_augmentation/fit_systems/blocks.py:208 | kept: formula comment above its step (amendment 4)
co N127 | model_augmentation/fit_systems/blocks.py:211 | kept: formula comment above its step (amendment 4)
co N128 | model_augmentation/fit_systems/blocks.py:229 | kept: numbered phase header (amendment 7)
co N129 | model_augmentation/fit_systems/blocks.py:231 | kept: formula comment above its step (amendment 4)
co N130 | model_augmentation/fit_systems/blocks.py:235 | deleted
co N131 | model_augmentation/fit_systems/blocks.py:242 | kept: numbered phase header (amendment 7)
co N132 | model_augmentation/fit_systems/blocks.py:247 | kept: docstring content required by amendment 7
co N133 | model_augmentation/fit_systems/blocks.py:266 | kept: numbered phase header (amendment 7)
co N134 | model_augmentation/fit_systems/blocks.py:272 | kept: numbered phase header (amendment 7)
co N135 | model_augmentation/fit_systems/blocks.py:277 | kept: numbered phase header (amendment 7)
co N136 | model_augmentation/fit_systems/blocks.py:278 | kept: formula comment above its step (amendment 4)
co N137 | model_augmentation/fit_systems/blocks.py:279 | kept: formula comment above its step (amendment 4)
co N138 | model_augmentation/fit_systems/blocks.py:288 | kept: formula comment above its step (amendment 4)
co N139 | model_augmentation/fit_systems/blocks.py:292 | kept: formula comment above its step (amendment 4)
co N140 | model_augmentation/fit_systems/blocks.py:295 | kept: formula comment above its step (amendment 4)
co N141 | model_augmentation/fit_systems/blocks.py:303 | kept: formula comment above its step (amendment 4)
co N142 | model_augmentation/fit_systems/blocks.py:307 | kept: numbered phase header (amendment 7)
co N143 | model_augmentation/fit_systems/blocks.py:309 | kept: formula comment above its step (amendment 4)
co N144 | model_augmentation/fit_systems/blocks.py:313 | kept: formula comment above its step (amendment 4)
co N145 | model_augmentation/fit_systems/blocks.py:314 | kept: formula comment above its step (amendment 4)
co N146 | model_augmentation/fit_systems/blocks.py:326 | kept: numbered phase header (amendment 7)
co N147 | model_augmentation/fit_systems/blocks.py:332 | kept: docstring content required by amendment 7
co N148 | model_augmentation/fit_systems/blocks.py:346 | kept: formula comment above its step (amendment 4)
co N149 | model_augmentation/fit_systems/blocks.py:356 | kept: formula comment above its step (amendment 4)
co N150 | model_augmentation/fit_systems/blocks.py:395 | kept: formula comment above its step (amendment 4)
co N151 | model_augmentation/fit_systems/blocks.py:401 | deleted (agreed)
co N152 | model_augmentation/fit_systems/blocks.py:421 | kept: formula comment above its step (amendment 4)
co N153 | model_augmentation/fit_systems/blocks.py:427 | kept: formula comment above its step (amendment 4)
co N154 | model_augmentation/fit_systems/blocks.py:450 | kept: formula comment above its step (amendment 4)
co N155 | model_augmentation/fit_systems/blocks.py:458 | kept: docstring content required by amendment 7
co N156 | model_augmentation/fit_systems/blocks.py:494 | kept: docstring content required by amendment 7
co N157 | model_augmentation/fit_systems/blocks.py:504 | kept: numbered phase header (amendment 7)
co N158 | model_augmentation/fit_systems/blocks.py:511 | kept: numbered phase header (amendment 7)
co N159 | model_augmentation/fit_systems/blocks.py:525 | kept: docstring content required by amendment 7
co N160 | model_augmentation/fit_systems/blocks.py:562 | kept: formula comment above its step (amendment 4)
co N161 | model_augmentation/fit_systems/blocks.py:567 | kept: formula comment above its step (amendment 4)
co N162 | model_augmentation/fit_systems/blocks.py:568 | kept: formula comment above its step (amendment 4)
co N163 | model_augmentation/fit_systems/blocks.py:575 | kept: docstring content required by amendment 7
co N164 | model_augmentation/fit_systems/blocks.py:589 | kept: docstring content required by amendment 7
co N165 | model_augmentation/fit_systems/blocks.py:600 | kept: formula comment above its step (amendment 4)
co N166 | model_augmentation/fit_systems/blocks.py:602 | kept: formula comment above its step (amendment 4)
co N167 | model_augmentation/fit_systems/blocks.py:608 | kept: docstring content required by amendment 7
co N168 | model_augmentation/fit_systems/blocks.py:624 | kept: formula comment above its step (amendment 4)
co N169 | model_augmentation/fit_systems/blocks.py:630 | kept: formula comment above its step (amendment 4)
co N170 | model_augmentation/fit_systems/blocks.py:636 | kept: docstring content required by amendment 7
co N171 | model_augmentation/fit_systems/blocks.py:646 | kept: docstring content required by amendment 7
co N172 | model_augmentation/fit_systems/blocks.py:660 | kept: docstring content required by amendment 7
co N173 | model_augmentation/fit_systems/closed_loop.py:26 | kept: docstring content required by amendment 7
co N174 | model_augmentation/fit_systems/closed_loop.py:46 | kept: THEORY/HEURISTIC label (shortened where its setting was answered)
co N175 | model_augmentation/fit_systems/closed_loop.py:47 | kept: formula comment above its step (amendment 4)
co N176 | model_augmentation/fit_systems/closed_loop.py:69 | kept: formula comment above its step (amendment 4)
co N177 | model_augmentation/fit_systems/closed_loop.py:90 | kept: docstring content required by amendment 7
co N178 | model_augmentation/fit_systems/closed_loop.py:93 | kept: formula comment above its step (amendment 4)
co N179 | model_augmentation/fit_systems/closed_loop.py:95 | kept: formula comment above its step (amendment 4)
co N180 | model_augmentation/fit_systems/closed_loop.py:97 | kept: formula comment above its step (amendment 4)
co N181 | model_augmentation/fit_systems/closed_loop.py:145 | kept: docstring content required by amendment 7
co N182 | model_augmentation/fit_systems/closed_loop.py:151 | kept: says what None means
co N183 | model_augmentation/fit_systems/closed_loop.py:166 | kept: docstring content required by amendment 7
co N184 | model_augmentation/fit_systems/closed_loop.py:187 | kept: the reason for the condition
co N185 | model_augmentation/fit_systems/closed_loop.py:216 | kept: the reason no check follows
co N186 | model_augmentation/fit_systems/closed_loop.py:250 | kept: shape and unit
co N187 | model_augmentation/fit_systems/closed_loop.py:283 | kept: numbered phase header (amendment 7)
co N188 | model_augmentation/fit_systems/closed_loop.py:293 | kept: when the list is filled
co N189 | model_augmentation/fit_systems/closed_loop.py:295 | kept: says what step_kw is (Claude Q056 asks it)
co N190 | model_augmentation/fit_systems/closed_loop.py:297 | kept: numbered phase header (amendment 7)
co N191 | model_augmentation/fit_systems/closed_loop.py:302 | kept: formula comment above its step (amendment 4)
co N192 | model_augmentation/fit_systems/closed_loop.py:311 | kept: formula comment above its step (amendment 4)
co N193 | model_augmentation/fit_systems/interconnect.py:1 | kept: docstring content required by amendment 7
co N194 | model_augmentation/fit_systems/interconnect.py:2 | kept: CHANGED / @added / origin marker or Jan attribution
co N195 | model_augmentation/fit_systems/interconnect.py:3 | kept: CHANGED / @added / origin marker or Jan attribution
co N196 | model_augmentation/fit_systems/interconnect.py:9 | kept: CHANGED / @added / origin marker or Jan attribution
co N197 | model_augmentation/fit_systems/interconnect.py:10 | kept: CHANGED / @added / origin marker or Jan attribution
co N198 | model_augmentation/fit_systems/interconnect.py:18 | kept: CHANGED / @added / origin marker or Jan attribution
co N199 | model_augmentation/fit_systems/interconnect.py:19 | kept: CHANGED / @added / origin marker or Jan attribution
co N200 | model_augmentation/fit_systems/interconnect.py:27 | kept: docstring content required by amendment 7
co N201 | model_augmentation/fit_systems/interconnect.py:43 | kept: CHANGED / @added / origin marker or Jan attribution
co N202 | model_augmentation/fit_systems/interconnect.py:56 | kept: docstring content required by amendment 7
co N203 | model_augmentation/fit_systems/interconnect.py:79 | kept: CHANGED / @added / origin marker or Jan attribution
co N204 | model_augmentation/fit_systems/interconnect.py:100 | kept: short, says what None means
co N205 | model_augmentation/fit_systems/interconnect.py:112 | kept: why this block gets the matrices
co N206 | model_augmentation/fit_systems/interconnect.py:121 | kept: docstring content required by amendment 7
co N207 | model_augmentation/fit_systems/interconnect.py:160 | kept: CHANGED / @added / origin marker or Jan attribution
co N208 | model_augmentation/fit_systems/interconnect.py:162 | kept: CHANGED / @added / origin marker or Jan attribution
co N209 | model_augmentation/fit_systems/interconnect.py:166 | kept: CHANGED / @added / origin marker or Jan attribution
co N210 | model_augmentation/fit_systems/interconnect.py:269 | kept: docstring content required by amendment 7
co N211 | model_augmentation/fit_systems/interconnect.py:284 | kept: docstring content required by amendment 7
co N212 | model_augmentation/fit_systems/interconnect.py:298 | kept: docstring content required by amendment 7
co N213 | model_augmentation/fit_systems/interconnect.py:327 | kept: docstring content required by amendment 7
co N214 | model_augmentation/fit_systems/interconnect.py:332 | deleted (agreed)
co N215 | model_augmentation/fit_systems/interconnect.py:354 | kept: CHANGED / @added / origin marker or Jan attribution
co N216 | model_augmentation/fit_systems/interconnect.py:358 | kept: docstring content required by amendment 7
co N217 | model_augmentation/fit_systems/interconnect.py:366 | kept: docstring content required by amendment 7
co N218 | model_augmentation/fit_systems/interconnect.py:371 | kept: docstring content required by amendment 7
co N219 | model_augmentation/fit_systems/interconnect.py:377 | kept: docstring content required by amendment 7
co N220 | model_augmentation/fit_systems/interconnect.py:383 | kept: CHANGED / @added / origin marker or Jan attribution
co N221 | model_augmentation/fit_systems/interconnect.py:384 | kept: CHANGED / @added / origin marker or Jan attribution
co N222 | model_augmentation/fit_systems/interconnect.py:405 | kept: docstring content required by amendment 7
co N223 | model_augmentation/fit_systems/interconnect.py:407 | kept: the reason no measure is passed
co N224 | model_augmentation/fit_systems/interconnect.py:419 | kept: the device contract of the dump
co N225 | model_augmentation/fit_systems/interconnect.py:427 | kept: numbered phase header (amendment 7)
co N226 | model_augmentation/fit_systems/interconnect.py:428 | kept: CHANGED / @added / origin marker or Jan attribution
co N227 | model_augmentation/fit_systems/interconnect.py:447 | kept: CHANGED / @added / origin marker or Jan attribution
co N228 | model_augmentation/fit_systems/interconnect.py:456 | kept: numbered phase header (amendment 7)
co N229 | model_augmentation/fit_systems/interconnect.py:463 | kept: CHANGED / @added / origin marker or Jan attribution
co N230 | model_augmentation/fit_systems/interconnect.py:467 | kept: docstring content required by amendment 7
co N231 | model_augmentation/fit_systems/interconnect.py:502 | kept: numbered phase header (amendment 7)
co N232 | model_augmentation/fit_systems/interconnect.py:533 | kept: docstring content required by amendment 7
co N233 | model_augmentation/fit_systems/interconnect.py:537 | deleted (agreed)
co N234 | model_augmentation/fit_systems/interconnect.py:548 | kept: docstring content required by amendment 7
co N235 | model_augmentation/fit_systems/interconnect.py:570 | kept: docstring content required by amendment 7
co N236 | model_augmentation/fit_systems/interconnect.py:588 | kept: docstring content required by amendment 7
co N237 | model_augmentation/fit_systems/interconnect.py:600 | kept: numbered phase header (amendment 7)
co N238 | model_augmentation/fit_systems/interconnect.py:602 | kept: numbered phase header (amendment 7)
co N239 | model_augmentation/fit_systems/interconnect.py:612 | kept: numbered phase header (amendment 7)
co N240 | model_augmentation/fit_systems/interconnect.py:619 | kept: formula comment above its step (amendment 4)
co N241 | model_augmentation/fit_systems/interconnect.py:642 | kept: docstring content required by amendment 7
co N242 | model_augmentation/fit_systems/lbfgs_polish.py:17 | kept: docstring content required by amendment 7
co N243 | model_augmentation/fit_systems/lbfgs_polish.py:42 | kept: docstring content required by amendment 7
co N244 | model_augmentation/fit_systems/lbfgs_polish.py:53 | kept: names the check that guards this call
co N245 | model_augmentation/fit_systems/lbfgs_polish.py:67 | kept: docstring content required by amendment 7
co N246 | model_augmentation/fit_systems/lbfgs_polish.py:87 | kept: docstring content required by amendment 7
co N247 | model_augmentation/fit_systems/lbfgs_polish.py:104 | kept: docstring content required by amendment 7
co N248 | model_augmentation/fit_systems/lbfgs_polish.py:126 | kept: docstring content required by amendment 7
co N249 | model_augmentation/fit_systems/lbfgs_polish.py:138 | kept: docstring content required by amendment 7
co N250 | model_augmentation/fit_systems/lbfgs_polish.py:141 | kept: the role sentence amendment 7 asks for
co N251 | model_augmentation/fit_systems/lbfgs_polish.py:160 | kept: why the snapshot is taken
co N252 | model_augmentation/fit_systems/lbfgs_polish.py:166 | kept: docstring content required by amendment 7
co N253 | model_augmentation/fit_systems/lbfgs_polish.py:177 | kept: docstring content required by amendment 7
co N254 | model_augmentation/fit_systems/lbfgs_polish.py:195 | kept: numbered phase header (amendment 7)
co N255 | model_augmentation/fit_systems/lbfgs_polish.py:205 | kept: docstring content required by amendment 7
co N256 | model_augmentation/fit_systems/lbfgs_polish.py:211 | kept: numbered phase header (amendment 7)
co N257 | model_augmentation/fit_systems/lbfgs_polish.py:212 | kept: the reason for the probe
co N258 | model_augmentation/fit_systems/lbfgs_polish.py:213 | kept: continuation of N257
co N259 | model_augmentation/fit_systems/lbfgs_polish.py:220 | kept: the retry rule is not obvious from the boolean
co N260 | model_augmentation/fit_systems/lbfgs_polish.py:227 | kept: why the references are dropped
co N261 | model_augmentation/fit_systems/lbfgs_polish.py:254 | kept: numbered phase header (amendment 7)
co N262 | model_augmentation/fit_systems/lbfgs_polish.py:255 | kept: answers the agreed L-BFGS limits question
co N263 | model_augmentation/fit_systems/lbfgs_polish.py:273 | kept: says why no gradient
co N264 | model_augmentation/fit_systems/lbfgs_polish.py:294 | kept: numbered phase header (amendment 7)
co N265 | model_augmentation/fit_systems/obc.py:18 | kept: docstring content required by amendment 7
co N266 | model_augmentation/fit_systems/obc.py:19 | kept: docstring content required by amendment 7
co N267 | model_augmentation/fit_systems/obc.py:22 | kept: docstring content required by amendment 7
co N268 | model_augmentation/fit_systems/obc.py:34 | kept: numbered phase header (amendment 7)
co N269 | model_augmentation/fit_systems/obc.py:38 | kept: numbered phase header (amendment 7)
co N270 | model_augmentation/fit_systems/obc.py:39 | kept: formula comment above its step (amendment 4)
co N271 | model_augmentation/fit_systems/obc.py:41 | kept: numbered phase header (amendment 7)
co N272 | model_augmentation/fit_systems/obc.py:54 | kept: docstring content required by amendment 7
co N273 | model_augmentation/fit_systems/obc.py:65 | kept: formula comment above its step (amendment 4)
co N274 | model_augmentation/fit_systems/obc.py:70 | kept: docstring content required by amendment 7
co N275 | model_augmentation/fit_systems/obc.py:78 | kept: docstring content required by amendment 7
co N276 | model_augmentation/fit_systems/obc.py:85 | kept: docstring content required by amendment 7
co N277 | model_augmentation/fit_systems/obc.py:102 | kept: numbered phase header (amendment 7)
co N278 | model_augmentation/fit_systems/obc.py:110 | kept: numbered phase header (amendment 7)
co N279 | model_augmentation/fit_systems/obc.py:113 | kept: formula comment above its step (amendment 4)
co N280 | model_augmentation/fit_systems/obc.py:118 | kept: numbered phase header (amendment 7)
co N281 | model_augmentation/fit_systems/obc.py:121 | kept: formula comment above its step (amendment 4)
co N282 | model_augmentation/fit_systems/obc.py:133 | kept: docstring content required by amendment 7
co N283 | model_augmentation/fit_systems/obc.py:142 | kept: docstring content required by amendment 7
co N284 | model_augmentation/fit_systems/obc.py:150 | kept: numbered phase header (amendment 7)
co N285 | model_augmentation/fit_systems/obc.py:157 | kept: numbered phase header (amendment 7)
co N286 | model_augmentation/fit_systems/obc.py:176 | kept: docstring content required by amendment 7
co N287 | model_augmentation/fit_systems/obc.py:188 | kept: docstring content required by amendment 7
co N288 | model_augmentation/fit_systems/pre_encoder.py:1 | kept: docstring content required by amendment 7
co N289 | model_augmentation/fit_systems/pre_encoder.py:6 | kept: CHANGED / @added / origin marker or Jan attribution
co N290 | model_augmentation/fit_systems/pre_encoder.py:7 | kept: CHANGED / @added / origin marker or Jan attribution
co N291 | model_augmentation/fit_systems/pre_encoder.py:12 | kept: CHANGED / @added / origin marker or Jan attribution
co N292 | model_augmentation/fit_systems/pre_encoder.py:34 | kept: docstring content required by amendment 7
co N293 | model_augmentation/fit_systems/pre_encoder.py:54 | kept: numbered phase header (amendment 7)
co N294 | model_augmentation/fit_systems/pre_encoder.py:55 | deleted (agreed)
co N295 | model_augmentation/fit_systems/pre_encoder.py:64 | kept: numbered phase header (amendment 7)
co N296 | model_augmentation/fit_systems/pre_encoder.py:73 | kept: numbered phase header (amendment 7)
co N297 | model_augmentation/fit_systems/pre_encoder.py:82 | kept: docstring content required by amendment 7
co N298 | model_augmentation/fit_systems/pre_encoder.py:94 | kept: formula comment above its step (amendment 4)
co N299 | model_augmentation/fit_systems/pre_encoder.py:102 | kept: formula comment above its step (amendment 4)
co N300 | model_augmentation/fit_systems/ps2_opening.py:46 | kept: docstring content required by amendment 7
co N301 | model_augmentation/fit_systems/ps2_opening.py:61 | kept: the example defines 'class'
co N302 | model_augmentation/fit_systems/ps2_opening.py:70 | kept: docstring content required by amendment 7
co N303 | model_augmentation/fit_systems/ps2_opening.py:71 | kept: docstring content required by amendment 7
co N304 | model_augmentation/fit_systems/ps2_opening.py:75 | kept: the column invariant
co N305 | model_augmentation/fit_systems/ps2_opening.py:95 | kept: docstring content required by amendment 7
co N306 | model_augmentation/fit_systems/ps2_opening.py:99 | kept: docstring content required by amendment 7
co N307 | model_augmentation/fit_systems/ps2_opening.py:117 | kept: numbered phase header (amendment 7)
co N308 | model_augmentation/fit_systems/ps2_opening.py:119 | kept: numbered phase header (amendment 7)
co N309 | model_augmentation/fit_systems/ps2_opening.py:214 | kept: formula comment above its step (amendment 4)
co N310 | model_augmentation/fit_systems/ps2_opening.py:216 | kept: formula comment above its step (amendment 4)
co N311 | model_augmentation/fit_systems/ps2_opening.py:220 | kept: docstring content required by amendment 7
co N312 | model_augmentation/fit_systems/ps2_opening.py:224 | kept: docstring content required by amendment 7
co N313 | model_augmentation/fit_systems/ps2_opening.py:232 | kept: numbered phase header (amendment 7)
co N314 | model_augmentation/fit_systems/ps2_opening.py:235 | kept: numbered phase header (amendment 7)
co N315 | model_augmentation/fit_systems/ps2_opening.py:237 | kept: numbered phase header (amendment 7)
co N316 | model_augmentation/fit_systems/ps2_opening.py:238 | kept: formula comment above its step (amendment 4)
co N317 | model_augmentation/fit_systems/ps2_opening.py:240 | kept: formula comment above its step (amendment 4)
co N318 | model_augmentation/fit_systems/ps2_opening.py:242 | kept: numbered phase header (amendment 7)
co N319 | model_augmentation/fit_systems/ps2_opening.py:252 | kept: docstring content required by amendment 7
co N320 | model_augmentation/fit_systems/ps2_opening.py:267 | kept: docstring content required by amendment 7
co N321 | model_augmentation/fit_systems/ps2_opening.py:285 | kept: docstring content required by amendment 7
co N322 | model_augmentation/fit_systems/ps2_opening.py:289 | kept: numbered phase header (amendment 7)
co N323 | model_augmentation/fit_systems/ps2_opening.py:293 | kept: docstring content required by amendment 7
co N324 | model_augmentation/fit_systems/ps2_opening.py:294 | kept: docstring content required by amendment 7
co N325 | model_augmentation/fit_systems/ps2_opening.py:296 | kept: numbered phase header (amendment 7)
co N326 | model_augmentation/fit_systems/ps2_opening.py:301 | kept: formula comment above its step (amendment 4)
co N327 | model_augmentation/fit_systems/ps2_opening.py:305 | kept: the reason the start is a candidate
co N328 | model_augmentation/fit_systems/ps2_opening.py:323 | kept: numbered phase header (amendment 7)
co N329 | model_augmentation/systems/gantry_ss.py:60 | kept: formula comment above its step (amendment 4)
co N330 | model_augmentation/systems/gantry_ss.py:67 | kept: formula comment above its step (amendment 4)
co N331 | model_augmentation/systems/gantry_ss.py:105 | kept: formula comment above its step (amendment 4)
co N332 | model_augmentation/systems/gantry_ss.py:112 | kept: formula comment above its step (amendment 4)
co N333 | model_augmentation/utils/deepSI_corrections.py:5 | kept: CHANGED / @added / origin marker or Jan attribution
co N334 | model_augmentation/utils/deepSI_corrections.py:6 | kept: CHANGED / @added / origin marker or Jan attribution
co N335 | model_augmentation/utils/deepSI_corrections.py:7 | kept: CHANGED / @added / origin marker or Jan attribution
co N336 | model_augmentation/utils/deepSI_corrections.py:15 | kept: docstring content required by amendment 7
co N337 | model_augmentation/utils/deepSI_corrections.py:20 | kept: docstring content required by amendment 7
co N338 | model_augmentation/utils/deepSI_corrections.py:23 | deleted
co N339 | model_augmentation/utils/deepSI_corrections.py:51 | kept: docstring content required by amendment 7
co N340 | model_augmentation/utils/deepSI_corrections.py:58 | kept: docstring content required by amendment 7
co N341 | model_augmentation/utils/deepSI_corrections.py:60 | kept: docstring content required by amendment 7
co N342 | model_augmentation/utils/deepSI_corrections.py:69 | kept: CHANGED / @added / origin marker or Jan attribution
co N343 | model_augmentation/utils/deepSI_corrections.py:75 | kept: formula comment above its step (amendment 4)
co N344 | model_augmentation/utils/torch_nets.py:6 | kept: CHANGED / @added / origin marker or Jan attribution
co N345 | model_augmentation/utils/torch_nets.py:7 | kept: CHANGED / @added / origin marker or Jan attribution
co N346 | model_augmentation/utils/torch_nets.py:11 | kept: CHANGED / @added / origin marker or Jan attribution
co N347 | model_augmentation/utils/torch_nets.py:24 | kept: docstring content required by amendment 7
co N348 | model_augmentation/utils/torch_nets.py:37 | kept: formula comment above its step (amendment 4)
co N349 | model_augmentation/utils/torch_nets.py:48 | kept: docstring content required by amendment 7
co N350 | model_augmentation/utils/torch_nets.py:50 | kept: docstring content required by amendment 7
co N351 | model_augmentation/utils/torch_nets.py:59 | kept: the reason the physical rows are the first nx_phys
co N352 | model_augmentation/utils/torch_nets.py:61 | kept: formula comment above its step (amendment 4)
co N353 | model_augmentation/utils/torch_nets.py:65 | kept: docstring content required by amendment 7
co N354 | model_augmentation/utils/torch_nets.py:67 | kept: docstring content required by amendment 7
co N355 | model_augmentation/utils/utils.py:1 | kept: CHANGED / @added / origin marker or Jan attribution
co N356 | model_augmentation/utils/utils.py:2 | kept: CHANGED / @added / origin marker or Jan attribution
co N357 | model_augmentation/utils/utils.py:3 | kept: CHANGED / @added / origin marker or Jan attribution
co N358 | model_augmentation/utils/utils.py:9 | kept: CHANGED / @added / origin marker or Jan attribution
co N359 | model_augmentation/utils/utils.py:14 | kept: docstring content required by amendment 7
co N360 | model_augmentation/utils/utils.py:15 | kept: CHANGED / @added / origin marker or Jan attribution
co N361 | model_augmentation/utils/utils.py:20 | kept: docstring content required by amendment 7
co N362 | model_augmentation/utils/utils.py:27 | kept: docstring content required by amendment 7
co N363 | model_augmentation/utils/utils.py:34 | kept: docstring content required by amendment 7
co N364 | model_augmentation/utils/utils.py:41 | kept: docstring content required by amendment 7
```
