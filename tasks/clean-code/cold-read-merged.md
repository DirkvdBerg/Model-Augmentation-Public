# Cold read, merged (D-284 amendment 11)

Sources: `cold-read-codex.md` (251 questions, 364 noise), `cold-read-claude.md` (95 questions, 95 noise). Item ids are
the readers' own: `co` = Codex, `cl` = Claude. Line numbers refer to `final-step5a`.

| Step | Codex | Claude | Result |
|-|-|-|-|
| Noise hitting a standing ruling | 312 | 83 | kept (or label shortened), no ruling needed |
| Noise found by both | 11 | 11 | 11 lines, applied |
| Noise found by one | 41 | 1 | for Dirk: recommend delete 6, keep 36 |
| Questions found by both | 121 | 75 | 68 groups, answered |
| Questions found by one | 130 | 20 | for Dirk: recommend answer 82, answer the what only 9, reject 59 |

## 1. Noise hitting a standing ruling (kept, no ruling needed)

Filtered first (handoff section 2, amendments 4 and 7). Per ruling, the items:

- **kept: CHANGED / @added / __project_origin__ marker or the attribution naming Jan's commit (CLAUDE.md Tracking Additions, D-282, BSD)** (63). co: N097, N098, N099, N101, N102, N103, N107, N108, N109, N111, N114, N117, N194, N195, N196, N197, N198, N199, N201, N203, N207, N208, N209, N215, N220, N221, N226, N227, N229, N289, N290, N291, N333, N334, N335, N342, N344, N345, N346, N355, N356, N357, N358, N360. cl: N027, N028, N029, N030, N031, N059, N060, N061, N063, N064, N065, N073, N074, N087, N088, N090, N091, N094, N095.
- **kept: Jan's own docstring (Jan's code 1:1, amendment 10 point 1)** (1). co: N104. cl: none.
- **kept: formula comment above its step (amendment 4); none repeats the code token for token** (115). co: N021, N022, N030, N034, N035, N039, N040, N042, N051, N052, N064, N071, N075, N076, N077, N078, N122, N126, N127, N129, N136, N137, N138, N139, N140, N141, N143, N144, N145, N148, N149, N150, N152, N153, N154, N160, N161, N162, N165, N166, N168, N169, N175, N176, N178, N179, N180, N191, N192, N240, N270, N273, N279, N281, N298, N299, N309, N310, N316, N317, N326, N329, N330, N331, N332, N343, N348, N352. cl: N012, N013, N015, N016, N017, N018, N019, N021, N023, N024, N033, N035, N039, N040, N041, N042, N044, N045, N046, N048, N049, N050, N051, N052, N053, N054, N055, N056, N057, N058, N067, N069, N070, N071, N072, N076, N077, N078, N079, N080, N081, N084, N085, N086, N089, N092, N093.
- **kept: numbered phase header (amendment 7 point 2)** (88). co: N003, N005, N006, N008, N009, N010, N023, N024, N026, N027, N028, N029, N038, N041, N043, N046, N047, N048, N049, N050, N053, N054, N056, N057, N059, N060, N062, N065, N066, N067, N069, N073, N091, N092, N093, N094, N096, N120, N124, N125, N128, N131, N133, N134, N135, N142, N146, N157, N158, N187, N190, N225, N228, N231, N237, N238, N239, N254, N256, N261, N264, N268, N269, N271, N277, N278, N280, N284, N285, N293, N295, N296, N307, N308, N313, N314, N315, N318, N322, N325, N328. cl: N020, N034, N036, N037, N038, N043, N068.
- **kept: docstring line required by amendment 7 point 1 (thesis name, formula, role, Args/Returns with shapes; one sentence for helpers)** (116). co: N011, N012, N014, N017, N018, N020, N036, N037, N045, N061, N079, N080, N083, N084, N085, N086, N087, N088, N089, N100, N105, N106, N110, N112, N113, N115, N116, N118, N121, N123, N132, N147, N155, N156, N159, N163, N164, N167, N170, N171, N172, N173, N177, N181, N183, N193, N200, N202, N206, N210, N211, N212, N213, N216, N217, N218, N219, N222, N230, N232, N234, N235, N236, N241, N242, N243, N245, N246, N247, N248, N249, N252, N253, N255, N265, N266, N267, N272, N274, N275, N276, N282, N283, N286, N287, N288, N292, N297, N300, N302, N303, N305, N306, N311, N312, N319, N320, N321, N323, N324, N336, N337, N339, N340, N341, N347, N349, N350, N353, N354, N359, N361, N362, N363, N364. cl: N010.
- **kept: THEORY/HEURISTIC label: never deleted; shortened into the source tag of the amendment 11 pattern when its setting is answered** (12). co: N019, N025, N174. cl: N004, N005, N006, N007, N008, N009, N011, N014, N083.

Labels found as noise: Claude N004 to N009 (`config.py` 'of the thesis runs') lose the provenance phrase when their
settings get the pattern; `config.py:174` and `controller.py:108` (both readers) keep the tag, shortened.

The phase-header and docstring rulings extend the handoff's list from amendment 7 (Codex flagged 81 phase headers and
115 docstring lines, many of them the first line amendment 7 requires). One of these was found by both readers:
`config.py:159` ("True on the OBC arm.", co N018, cl N010), kept as the one-sentence helper docstring.
Continuation lines of a CHANGED comment count as the marker (for example `interconnect.py:10`, `:19`, `:384`).

## 2. Noise found by both (applied)

| co | cl | Line | Action |
|-|-|-|-|
| N072 | N022 | `gantry_dynamic/obc_gantry.py:84` | delete the history clause; end line 83 at the stencil range (Codex Q097 asks the reason, see there) |
| N081 | N025 | `gantry_dynamic/obc_gantry.py:209` | keep the formula (amendment 4), delete the pointer clause 'same span as ... (module docstring)' |
| N095 | N026 | `gantry_dynamic/training.py:317` | delete |
| N001 | N001 | `gantry_interconnect_dynamic.py:22` | delete |
| N004 | N002 | `gantry_interconnect_dynamic.py:80` | delete |
| N007 | N003 | `gantry_interconnect_dynamic.py:112` | delete |
| N119 | N032 | `model_augmentation/fit_systems/blocks.py:148` | delete (the docstring at line 141 states closed_form) |
| N151 | N047 | `model_augmentation/fit_systems/blocks.py:401` | delete |
| N214 | N062 | `model_augmentation/fit_systems/interconnect.py:332` | delete; the seed-0 answer of the agreed question (seed-0 batch order) goes in the docstring at line 325 |
| N233 | N066 | `model_augmentation/fit_systems/interconnect.py:537` | delete |
| N294 | N075 | `model_augmentation/fit_systems/pre_encoder.py:55` | delete; the window sizes go in the na, nb Args line (agreed question na_nb) |

## 3. Questions found by both (answered)

| Group | cl | co | Topic |
|-|-|-|-|
| G01 | Q004 | Q001, Q114 | load_checkpoint: what it restores and the meta keys |
| G02 | Q006 | Q003 | bestfit: which score, before or after the polish |
| G03 | Q009 | Q005 | resume from best or last (the from flag) |
| G04 | Q002 | Q125 | the pre-polish checkpoint: which weights |
| G05 | Q008 | Q013 | compile_mode: effect on results, speed, memory |
| G06 | Q013 | Q007, Q008 | AA_TRIM_S / record_trim: reference of the 0.01 rule, also on the noise-free set |
| G07 | Q015 | Q012 | ps2: what the opening is and what it does |
| G08 | Q016 | Q010 | start='correct': relative to which parameters |
| G09 | Q017 | Q011 | n_a: what the added states are, why 0, 2, 8 |
| G10 | Q018, Q039 | Q044, Q232 | data set names: noise setting and the dynamics not in the baseline |
| G11 | Q019 | Q014 | truncate_s |
| G12 | Q020 | Q015 | epochs: what one epoch is |
| G13 | Q021 | Q016 | n_its: what it counts, relation to epochs, None |
| G14 | Q022 | Q017, Q122 | its_per_val, and its replacement by n_its |
| G15 | Q023 | Q018, Q190, Q197 | L-BFGS limits max_iter and inner_iter |
| G16 | Q024 | Q019, Q191 | lbfgs_windows: how drawn, effect of the count |
| G17 | Q049 | Q020, Q069, Q113 | start_phase, and 'lbfgs' at evaluation |
| G18 | Q025, Q026, Q038 | Q022, Q034, Q057, Q075, Q135, Q240 | state, input and output ordering, frame and units |
| G19 | Q027 | Q024 | stride: unit, effect |
| G20 | Q028 | Q026, Q206, Q211 | na_nb: which samples, current sample included |
| G21 | Q029 | Q028, Q134 | up_sample |
| G22 | Q031 | Q029 | lbfgs_tol_grad, lbfgs_tol_change |
| G23 | Q033 | Q030, Q072 | the ten combinations theta_base and their order |
| G24 | Q076 | Q031, Q032, Q033, Q140, Q234, Q235, Q236, Q237, Q238, Q239 | the fourteen raw parameters: element and unit |
| G25 | Q036 | Q035, Q242 | Y_op / Y: design Y, origin, range |
| G26 | Q037 | Q037, Q039 | gain and use_gain |
| G27 | Q040 | Q045 | test-I and test-E |
| G28 | Q041 | Q047 | K1 records only |
| G29 | Q042 | Q049 | val_ckpt_data versus the lists |
| G30 | Q043 | Q050, Q131, Q150 | D (Dd_np), C^n and feedthrough |
| G31 | Q044 | Q053 | only y - r filtered |
| G32 | Q046 | Q051 | sample alignment of u_k and y_k |
| G33 | Q047 | Q058 | backward-difference velocities for the normalisation |
| G34 | Q048 | Q063 | 'true' parameters in the evaluation |
| G35 | Q052 | Q077 | connect_block_signals / connect_signals in build_model |
| G36 | Q053, Q054 | Q079 | init_model after norm and encoder are set; dtype cast |
| G37 | Q055 | Q082 | OBC coefficient synced before the start simulation |
| G38 | Q056, Q077 | Q084, Q136, Q149, Q159 | the per-objective matrices (step_kw, mats, pass_mats, obc_pass) |
| G39 | Q057 | Q083 | zero initial velocity and states at the first sample |
| G40 | Q059 | Q085 | encoder start on simulated, not measured, windows |
| G41 | Q061 | Q089 | b - enc(0, 0) |
| G42 | Q062 | Q091, Q092, Q198 | OBC notation: vbar, varphi, Phi, Gamma, a, c |
| G43 | Q063 | Q093 | Z_step versus Z_field |
| G44 | Q064, Q065 | Q094, Q095, Q101 | reference tuples: measured, at xbar = 0, fixed |
| G45 | Q066 | Q096 | fourth-order central versus backward difference |
| G46 | Q067 | Q100, Q187 | theta_frozen |
| G47 | Q068 | Q107 | space='affine', refresh='epoch' |
| G48 | Q089 | Q099, Q188 | per-epoch refresh of beta_lin |
| G49 | Q069 | Q110, Q121 | PS phases S1, S2, S3 |
| G50 | Q070, Q078 | Q112, Q144 | coordinate_version / COORDINATE_VERSION |
| G51 | Q071 | Q118 | the lazy initialisation completed by one eager validation |
| G52 | Q073 | Q124, Q109, Q002 | done_epochs on resume and early stop |
| G53 | Q075 | Q138, Q243 | the LFR signals z and w |
| G54 | Q079 | Q152 | controller state x^c = 0 at each window start |
| G55 | Q080 | Q155 | defaults nf = 25, stride = 1 |
| G56 | Q081 | Q065, Q147 | free run: measured y in the residual controller |
| G57 | Q082 | Q158 | input_signal_sizes / output_signal_sizes |
| G58 | Q083 | Q163, Q164, Q165 | init_forward: the node graph, shift and roll |
| G59 | Q084 | Q169 | first additive connection turned into a concatenation |
| G60 | Q085 | Q177, Q247 | na_right, nb_right |
| G61 | Q086 | Q176 | seed-0 batch order |
| G62 | Q087 | Q196 | OBC basis during the polish |
| G63 | Q090 | Q207, Q208, Q210, Q250 | encoder start: W^a Xavier, added rows zero |
| G64 | Q092 | Q214 | PS residual horizon M = H = n + 1 |
| G65 | Q093 | Q233 | float32 nominal values |
| G66 | Q094 | Q246 | System_data arguments x, cheat_n, normed, dt |
| G67 | Q095 | Q248 | unused force_multi_u, force_multi_y, online_construct |
| G68 | Q030 | Q178, Q195 | one Adam rate and one parameter group for all parameters (also in the polish) |

## 4. Noise found by one reader (Dirk rules)

| Reader | Id | Line | Recommendation |
|-|-|-|-|
| co | N002 | `gantry_interconnect_dynamic.py:35` | delete 'nothing moves here' (filler); keep the rest of the docstring |
| co | N013 | `gantry_dynamic/config.py:41` | keep: says the dict maps fields to config.json keys and who reads it |
| co | N015 | `gantry_dynamic/config.py:68` | keep: section header; says where the values change |
| co | N016 | `gantry_dynamic/config.py:89` | keep: section header; says which values config.json records |
| co | N031 | `gantry_dynamic/data.py:22` | keep: not history, says where the folders sit (Claude Q039 extends it) |
| co | N032 | `gantry_dynamic/data.py:24` | replace by the answer to the agreed question test-I/test-E |
| co | N033 | `gantry_dynamic/data.py:26` | keep: the name SOS does not say anti-aliasing filter |
| co | N044 | `gantry_dynamic/evaluation.py:36` | keep: TEX does not say what it is for |
| co | N055 | `gantry_dynamic/model.py:79` | keep: the routing of expansion_matrix is not visible in the call |
| co | N058 | `gantry_dynamic/model.py:106` | keep: the reason for refusing the CPU fallback |
| co | N063 | `gantry_dynamic/model.py:240` | keep: says what the driver setting is for |
| co | N068 | `gantry_dynamic/obc_gantry.py:29` | keep: says what P_INV_T64 is |
| co | N070 | `gantry_dynamic/obc_gantry.py:80` | keep: float64 and resampling are not visible in loadmat |
| co | N074 | `gantry_dynamic/obc_gantry.py:124` | keep: the row-to-column invariant the projection relies on |
| co | N082 | `gantry_dynamic/obc_gantry.py:228` | keep: says what the cache holds and its key |
| co | N090 | `gantry_dynamic/training.py:248` | delete: the print below says it |
| co | N130 | `model_augmentation/fit_systems/blocks.py:235` | delete the history clause 'kept because the thesis results used it' |
| co | N182 | `model_augmentation/fit_systems/closed_loop.py:151` | keep: says what None means |
| co | N184 | `model_augmentation/fit_systems/closed_loop.py:187` | keep: the reason for the condition |
| co | N185 | `model_augmentation/fit_systems/closed_loop.py:216` | keep: the reason no check follows |
| co | N186 | `model_augmentation/fit_systems/closed_loop.py:250` | keep: shape and unit |
| co | N188 | `model_augmentation/fit_systems/closed_loop.py:293` | keep: when the list is filled |
| co | N189 | `model_augmentation/fit_systems/closed_loop.py:295` | keep: says what step_kw is (Claude Q056 asks it) |
| co | N204 | `model_augmentation/fit_systems/interconnect.py:100` | keep: short, says what None means |
| co | N205 | `model_augmentation/fit_systems/interconnect.py:112` | keep: why this block gets the matrices |
| co | N223 | `model_augmentation/fit_systems/interconnect.py:407` | keep: the reason no measure is passed |
| co | N224 | `model_augmentation/fit_systems/interconnect.py:419` | keep: the device contract of the dump |
| co | N244 | `model_augmentation/fit_systems/lbfgs_polish.py:53` | keep: names the check that guards this call |
| co | N250 | `model_augmentation/fit_systems/lbfgs_polish.py:141` | keep: the role sentence amendment 7 asks for |
| co | N251 | `model_augmentation/fit_systems/lbfgs_polish.py:160` | keep: why the snapshot is taken |
| co | N257 | `model_augmentation/fit_systems/lbfgs_polish.py:212` | keep: the reason for the probe |
| co | N258 | `model_augmentation/fit_systems/lbfgs_polish.py:213` | keep: continuation of N257 |
| co | N259 | `model_augmentation/fit_systems/lbfgs_polish.py:220` | keep: the retry rule is not obvious from the boolean |
| co | N260 | `model_augmentation/fit_systems/lbfgs_polish.py:227` | keep: why the references are dropped |
| co | N262 | `model_augmentation/fit_systems/lbfgs_polish.py:255` | keep: answers the agreed L-BFGS limits question |
| co | N263 | `model_augmentation/fit_systems/lbfgs_polish.py:273` | keep: says why no gradient |
| co | N301 | `model_augmentation/fit_systems/ps2_opening.py:61` | keep: the example defines 'class' |
| co | N304 | `model_augmentation/fit_systems/ps2_opening.py:75` | keep: the column invariant |
| co | N327 | `model_augmentation/fit_systems/ps2_opening.py:305` | keep: the reason the start is a candidate |
| co | N338 | `model_augmentation/utils/deepSI_corrections.py:23` | delete the provenance clause '(deepSI's loop, for every stride)' |
| co | N351 | `model_augmentation/utils/torch_nets.py:59` | keep: the reason the physical rows are the first nx_phys |
| cl | N082 | `model_augmentation/systems/gantry_ss.py:2` | delete 'from kamtin-fp-model/03 Simulink gantry/main.m (the immutable ground truth)' (not shipped, repo jargon); the THEORY tag at line 10 keeps the source, shortened to the paper |

## 5. Questions found by one reader (Dirk rules)

A = answer; W = answer what it is or does, reject the why; R = reject. Reject reasons: *design choice* (the reason
belongs to the thesis, the code is the plain form), *contract* (interface of an internal helper or of Jan's API with
one caller in `Clean/`), *hypothetical* (a use the code never meets), *answered* (the code or an existing comment says it).

| Reader | Id | Line | Rec. | Reason |
|-|-|-|-|-|
| cl | Q001 | `gantry_interconnect_dynamic.py:28` | A | where the _best and _last dumps land and what sets the folder; a user looks for them |
| cl | Q003 | `gantry_interconnect_dynamic.py:119` | A | Args line: servo errors per record and what the loss weights take from them |
| cl | Q005 | `gantry_interconnect_dynamic.py:130` | A | Args line for resume |
| cl | Q007 | `gantry_interconnect_dynamic.py:138` | W | define 'final checkpoint'; the CPU-versus-CUDA rounding part is not answerable without a run |
| cl | Q010 | `gantry_dynamic/config.py:16` | A | what the roots are and the flag that moves them; whether data ship with Clean/ is a README matter |
| cl | Q011 | `gantry_dynamic/config.py:20` | A | which fs and which error the AA comment means |
| cl | Q012 | `gantry_dynamic/config.py:22` | A | AA_FC_HZ must follow fs_new; a real trap |
| cl | Q014 | `gantry_dynamic/config.py:27` | A | one line: the rows are the thesis runs; validate() limits the values |
| cl | Q032 | `gantry_dynamic/config.py:145` | A | what CODE is (refuses an unconverted thesis folder); the converter is not shipped |
| cl | Q034 | `gantry_dynamic/controller.py:20` | A | one clause why controller.py keeps its own nominal copy |
| cl | Q035 | `gantry_dynamic/controller.py:20` | W | name the cited .m files as the MATLAB data-generation source; no path to files outside Clean/ |
| cl | Q045 | `gantry_dynamic/data.py:120` | A | one clause why the noise-free set needs no anti-aliasing filter |
| cl | Q050 | `gantry_dynamic/model.py:31` | A | name which beta (the training coordinates); the name collision is real |
| cl | Q051 | `gantry_dynamic/model.py:56` | A | params_init versus combo_init in one Args line each |
| cl | Q058 | `gantry_dynamic/model.py:185` | A | one clause: the closed-loop step calls output_only |
| cl | Q060 | `gantry_dynamic/model.py:225` | R | the unused return is a code question (write-only output); flag as residue, not a comment |
| cl | Q072 | `gantry_dynamic/training.py:238` | A | n_xb = number of physical states; one Args line |
| cl | Q074 | `model_augmentation/fit_systems/blocks.py:4` | A | one module-docstring line separating the mass terms alpha, beta, gamma from beta (coordinates) and the margin |
| cl | Q088 | `model_augmentation/fit_systems/lbfgs_polish.py:248` | A | 'opening loss' is misleading wording; say the loss at the polish start |
| cl | Q091 | `model_augmentation/fit_systems/ps2_opening.py:37` | A | s2_check = the check cadence in S2 steps |
| co | Q004 | `gantry_interconnect_dynamic.py:151` | A | which state the polish-only flag starts from |
| co | Q006 | `gantry_interconnect_dynamic.py:158` | W | what the closed-form flag bypasses (rounding only); cost/differentiability part not needed |
| co | Q009 | `gantry_dynamic/config.py:63` | A | arm has no comment; core axis |
| co | Q021 | `gantry_dynamic/config.py:90` | A | what the seed seeds; PS has its own (Codex Q218) |
| co | Q023 | `gantry_dynamic/config.py:97` | A | fs_new [Hz] and what follows it (AA_FC_HZ, Claude Q012) |
| co | Q025 | `gantry_dynamic/config.py:100` | A | nf_seconds: rollout length after the encoder window |
| co | Q027 | `gantry_dynamic/config.py:104` | A | what n_hidden_layers counts |
| co | Q036 | `gantry_dynamic/controller.py:77` | R | the THEORY tag names the source of the pole placement |
| co | Q038 | `gantry_dynamic/controller.py:144` | A | Args line: label or key |
| co | Q040 | `gantry_dynamic/controller.py:168` | A | Args line: per-axis std of the normaliser |
| co | Q041 | `gantry_dynamic/controller.py:200` | A | Args line naming the normaliser class |
| co | Q042 | `gantry_dynamic/controller.py:234` | A | one clause: the first samples have no full encoder window |
| co | Q043 | `gantry_dynamic/controller.py:237` | R | the W_r formula (controller.py:236) and V (interconnect.py:619) state it |
| co | Q046 | `gantry_dynamic/data.py:30` | A | the manifest columns the loader reads; a user with own data needs them |
| co | Q048 | `gantry_dynamic/data.py:45` | A | the controller fields read from the .mat (one line, with Q046) |
| co | Q052 | `gantry_dynamic/data.py:108` | A | the record fields read (one line, with Q046) |
| co | Q054 | `gantry_dynamic/data.py:127` | R | internal helper with one caller |
| co | Q055 | `gantry_dynamic/data.py:140` | R | hypothetical use (records from several folders) |
| co | Q056 | `gantry_dynamic/data.py:157` | A | one clause: k0 = first sample with a full window |
| co | Q059 | `gantry_dynamic/data.py:230` | R | pooled statistics are the plain form; no reason to state |
| co | Q060 | `gantry_dynamic/data.py:231` | W | one clause: 1e-8 is a division guard; check it carries its HEURISTIC tag |
| co | Q061 | `gantry_dynamic/evaluation.py:43` | A | one clause: theta_aux is the OBC coefficient, zeroing it removes f_obc |
| co | Q062 | `gantry_dynamic/evaluation.py:55` | A | how the baseline-only run gets its initial states; matters for the comparison |
| co | Q064 | `gantry_dynamic/evaluation.py:98` | R | interface contract of an internal helper |
| co | Q066 | `gantry_dynamic/evaluation.py:103` | W | say what the floor is; its assumptions belong to the thesis |
| co | Q067 | `gantry_dynamic/evaluation.py:181` | A | one clause why axes 0 and 2 |
| co | Q068 | `gantry_dynamic/evaluation.py:205` | R | edge case of a resumed run |
| co | Q070 | `gantry_dynamic/evaluation.py:272` | A | one clause: same records and resampling, so the floor compares like with like |
| co | Q071 | `gantry_dynamic/evaluation.py:278` | A | origin of t |
| co | Q073 | `gantry_dynamic/model.py:28` | A | hp versus cfg precedence |
| co | Q074 | `gantry_dynamic/model.py:32` | A | discrete increments, not derivatives; key semantics (also Q251) |
| co | Q076 | `gantry_dynamic/model.py:70` | R | activation choice is a thesis design choice |
| co | Q078 | `gantry_dynamic/model.py:95` | A | answered with the agreed na_nb group (history ordering) |
| co | Q080 | `gantry_dynamic/model.py:120` | A | one clause, with the agreed OBC-sync group |
| co | Q081 | `gantry_dynamic/model.py:126` | R | interface contract of an internal helper |
| co | Q086 | `gantry_dynamic/model.py:196` | R | design choice; the code shows it |
| co | Q087 | `gantry_dynamic/model.py:206` | W | what failing the R^2 check does, plus its HEURISTIC tag; no invented reason |
| co | Q088 | `gantry_dynamic/model.py:213` | R | observability argument belongs to the thesis |
| co | Q090 | `gantry_dynamic/model.py:274` | A | answered with the agreed encoder-start group |
| co | Q097 | `gantry_dynamic/obc_gantry.py:87` | R | no method reason; the only reason is identical results to the thesis runs, which is history (see agreed noise obc_gantry.py:84) |
| co | Q098 | `gantry_dynamic/obc_gantry.py:112` | A | what the protected rows are (also Q200) |
| co | Q102 | `gantry_dynamic/obc_gantry.py:187` | R | hypothetical use |
| co | Q103 | `gantry_dynamic/obc_gantry.py:200` | A | Returns line: shape and columns of J |
| co | Q104 | `gantry_dynamic/obc_gantry.py:210` | A | one clause: same span as [Phi Gamma] |
| co | Q105 | `gantry_dynamic/obc_gantry.py:217` | R | hypothetical use |
| co | Q106 | `gantry_dynamic/obc_gantry.py:244` | R | interface contract of an internal helper |
| co | Q108 | `gantry_dynamic/training.py:31` | A | answered with the agreed load_checkpoint group |
| co | Q111 | `gantry_dynamic/training.py:70` | R | interface contract |
| co | Q115 | `gantry_dynamic/training.py:164` | A | per_channel and score: RMS, unit, aggregation |
| co | Q116 | `gantry_dynamic/training.py:165` | A | update and epoch: cumulative or local, and -1 |
| co | Q117 | `gantry_dynamic/training.py:198` | R | selection design choice |
| co | Q119 | `gantry_dynamic/training.py:220` | A | one clause: which model a rejected row describes |
| co | Q120 | `gantry_dynamic/training.py:229` | A | one clause why bestfit restarts at infinity |
| co | Q123 | `gantry_dynamic/training.py:264` | R | edge case; ps2_summary records it |
| co | Q126 | `gantry_dynamic/training.py:310` | R | answered by the code |
| co | Q127 | `gantry_dynamic/training.py:333` | A | one clause: summary for the record, tuple for the resume |
| co | Q128 | `gantry_dynamic/training.py:346` | A | unit and origin of time |
| co | Q129 | `model_augmentation/fit_systems/blocks.py:9` | R | units follow from the parameter units (agreed group) |
| co | Q130 | `model_augmentation/fit_systems/blocks.py:25` | R | Jan's base-class contract |
| co | Q132 | `model_augmentation/fit_systems/blocks.py:119` | A | answered with the agreed per-objective matrices group |
| co | Q133 | `model_augmentation/fit_systems/blocks.py:137` | A | answered with the agreed ordering group |
| co | Q137 | `model_augmentation/fit_systems/blocks.py:216` | A | one clause: the rational inverse is not the obvious form (amendment 4 allows the reason) |
| co | Q139 | `model_augmentation/fit_systems/blocks.py:246` | A | one clause: same reason rule as Q137 |
| co | Q141 | `model_augmentation/fit_systems/blocks.py:369` | R | hypothetical inputs |
| co | Q142 | `model_augmentation/fit_systems/blocks.py:477` | R | identifiability argument belongs to the thesis |
| co | Q143 | `model_augmentation/fit_systems/blocks.py:489` | W | what M_DIFF_RHO bounds and what changing it does, plus its tag |
| co | Q145 | `model_augmentation/fit_systems/blocks.py:496` | R | hypothetical inputs |
| co | Q146 | `model_augmentation/fit_systems/blocks.py:549` | R | hypothetical inputs |
| co | Q148 | `model_augmentation/fit_systems/closed_loop.py:25` | R | built by controller.py; ordering is the agreed ordering group |
| co | Q151 | `model_augmentation/fit_systems/closed_loop.py:111` | A | u_data = recorded actuator input; key semantics |
| co | Q153 | `model_augmentation/fit_systems/closed_loop.py:148` | R | interface contract |
| co | Q154 | `model_augmentation/fit_systems/closed_loop.py:202` | A | answered with Q180 (normalised windows) |
| co | Q156 | `model_augmentation/fit_systems/closed_loop.py:221` | R | check design, answered by the code |
| co | Q157 | `model_augmentation/fit_systems/closed_loop.py:239` | R | selection criterion is a thesis choice |
| co | Q160 | `model_augmentation/fit_systems/interconnect.py:75` | R | Jan's design |
| co | Q161 | `model_augmentation/fit_systems/interconnect.py:96` | R | interface contract (Jan's) |
| co | Q162 | `model_augmentation/fit_systems/interconnect.py:118` | R | interface contract (Jan's) |
| co | Q166 | `model_augmentation/fit_systems/interconnect.py:148` | R | Jan's graph; one wiring in build_model |
| co | Q167 | `model_augmentation/fit_systems/interconnect.py:164` | R | hypothetical: the model has one physics block |
| co | Q168 | `model_augmentation/fit_systems/interconnect.py:196` | A | answered with the agreed D/feedthrough group |
| co | Q170 | `model_augmentation/fit_systems/interconnect.py:235` | R | Jan's internals |
| co | Q171 | `model_augmentation/fit_systems/interconnect.py:256` | R | Jan's API; our one call is the agreed connect-signals group |
| co | Q172 | `model_augmentation/fit_systems/interconnect.py:256` | R | Jan's API |
| co | Q173 | `model_augmentation/fit_systems/interconnect.py:256` | R | Jan's API |
| co | Q174 | `model_augmentation/fit_systems/interconnect.py:268` | R | Jan's API |
| co | Q175 | `model_augmentation/fit_systems/interconnect.py:324` | A | answered with Q180 |
| co | Q179 | `model_augmentation/fit_systems/interconnect.py:365` | R | hypothetical use |
| co | Q180 | `model_augmentation/fit_systems/interconnect.py:370` | A | Returns line of make_training_data: arrays, order, normalised |
| co | Q181 | `model_augmentation/fit_systems/interconnect.py:442` | A | answered with the agreed epoch group |
| co | Q182 | `model_augmentation/fit_systems/interconnect.py:452` | A | one clause: the start model is a candidate |
| co | Q183 | `model_augmentation/fit_systems/interconnect.py:478` | A | answered with the agreed PS-phases group |
| co | Q184 | `model_augmentation/fit_systems/interconnect.py:516` | A | one clause why the basis is rebuilt after the reload |
| co | Q185 | `model_augmentation/fit_systems/interconnect.py:528` | A | what is not pickled and who rebuilds it |
| co | Q186 | `model_augmentation/fit_systems/interconnect.py:550` | A | answered with the agreed per-epoch refresh group |
| co | Q189 | `model_augmentation/fit_systems/interconnect.py:608` | R | gradient-path analysis is thesis depth |
| co | Q192 | `model_augmentation/fit_systems/lbfgs_polish.py:75` | A | one clause what the stored stride is used for |
| co | Q193 | `model_augmentation/fit_systems/lbfgs_polish.py:119` | R | the comment at lbfgs_polish.py:212 gives the reason |
| co | Q194 | `model_augmentation/fit_systems/lbfgs_polish.py:147` | A | Returns lines of the outcome dict |
| co | Q199 | `model_augmentation/fit_systems/obc.py:21` | R | hypothetical; the class is general |
| co | Q200 | `model_augmentation/fit_systems/obc.py:28` | A | answered with Q098 |
| co | Q201 | `model_augmentation/fit_systems/obc.py:91` | R | interface contract |
| co | Q202 | `model_augmentation/fit_systems/obc.py:107` | R | gradient-path analysis, as Q189 |
| co | Q203 | `model_augmentation/fit_systems/obc.py:129` | W | what max_consecutive_rank_loss does, plus its tag |
| co | Q204 | `model_augmentation/fit_systems/obc.py:155` | R | answered by the code |
| co | Q205 | `model_augmentation/fit_systems/obc.py:187` | R | interface contract |
| co | Q209 | `model_augmentation/fit_systems/pre_encoder.py:74` | R | architecture choice of the thesis |
| co | Q212 | `model_augmentation/fit_systems/pre_encoder.py:89` | A | answered with the agreed ordering group (normalised x^n) |
| co | Q213 | `model_augmentation/fit_systems/pre_encoder.py:102` | A | one clause: the permutation follows Jan's encoder input order |
| co | Q215 | `model_augmentation/fit_systems/ps2_opening.py:22` | A | units of the PS2Spec defaults (updates, samples) |
| co | Q216 | `model_augmentation/fit_systems/ps2_opening.py:24` | W | what lr does; the factor 100 keeps its HEURISTIC tag, no invented reason |
| co | Q217 | `model_augmentation/fit_systems/ps2_opening.py:26` | R | hypothetical settings |
| co | Q218 | `model_augmentation/fit_systems/ps2_opening.py:42` | A | what the PS seed seeds (with Q021) |
| co | Q219 | `model_augmentation/fit_systems/ps2_opening.py:53` | A | what 'last' means; the reason clause if it is in the thesis |
| co | Q220 | `model_augmentation/fit_systems/ps2_opening.py:54` | A | one clause why calibration records stay in V |
| co | Q221 | `model_augmentation/fit_systems/ps2_opening.py:73` | A | answered with the agreed ps2 group (n_a = 0) |
| co | Q222 | `model_augmentation/fit_systems/ps2_opening.py:89` | A | Returns line: None in the summary |
| co | Q223 | `model_augmentation/fit_systems/ps2_opening.py:103` | R | PS design choice |
| co | Q224 | `model_augmentation/fit_systems/ps2_opening.py:149` | R | PS design choice (HEURISTIC) |
| co | Q225 | `model_augmentation/fit_systems/ps2_opening.py:181` | A | one clause why M + 1 samples are excluded |
| co | Q226 | `model_augmentation/fit_systems/ps2_opening.py:204` | R | PS design choice |
| co | Q227 | `model_augmentation/fit_systems/ps2_opening.py:243` | R | PS design choice |
| co | Q228 | `model_augmentation/fit_systems/ps2_opening.py:279` | R | PS design choice |
| co | Q229 | `model_augmentation/fit_systems/ps2_opening.py:288` | R | PS design choice |
| co | Q230 | `model_augmentation/fit_systems/ps2_opening.py:301` | R | the formula comment at line 301 states it |
| co | Q231 | `model_augmentation/fit_systems/ps2_opening.py:305` | R | PS design choice |
| co | Q241 | `model_augmentation/systems/gantry_ss.py:46` | R | hypothetical inputs |
| co | Q244 | `model_augmentation/systems/gantry_ss.py:95` | A | M1_, M2_ multiply Y and Y^2 |
| co | Q245 | `model_augmentation/systems/gantry_ss.py:98` | A | act on generalised forces, continuous time |
| co | Q249 | `model_augmentation/utils/torch_nets.py:16` | A | one clause: which builder the PS decoder uses |
| co | Q251 | `model_augmentation/utils/torch_nets.py:72` | A | answered with Q074 |
