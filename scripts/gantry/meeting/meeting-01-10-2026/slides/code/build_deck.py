"""Builds the 1 October 2026 meeting deck from ../OUTLINE.md: white slides, one message per slide, notes = what to say.
Run: conda run -n GraduationProject python build_deck.py   ->  ../meeting-01-10-2026.pptx
"""
__project_origin__ = "added"

import os
import re

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figures')
GANTRY = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
BAND = os.path.join(GANTRY, 'coulomb-tanh-gain', 'outputs', 'band_tanh_g1000', 'figures')
OUT = os.path.join(HERE, '..', 'meeting-01-10-2026.pptx')

FONT = 'Calibri'
BLACK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x59, 0x59, 0x59)
BLUE = RGBColor(0x1F, 0x4E, 0x79)
HEAD = RGBColor(0xD9, 0xD9, 0xD9)
BAND_ROW = RGBColor(0xF2, 0xF2, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
W, H = 13.333, 7.5
SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def sci(s):
    """3.0e-1 -> 3.0×10⁻¹ (errors and lengths in scientific notation, per the project convention)."""
    def rep(m):
        man, ex = m.group(1), str(int(m.group(2)))
        return ('10' if man == '1' else man + '×10') + ex.translate(SUP)
    return re.sub(r'(?<![\w.])(\d+(?:\.\d+)?)e([+-]?\d+)\b', rep, s)


def run_fmt(r, size, bold=False, color=BLACK, italic=False):
    r.font.name, r.font.size, r.font.bold, r.font.italic = FONT, Pt(size), bold, italic
    r.font.color.rgb = color


def text(slide, x, y, w, h, items, size=16, color=BLACK, bold=False, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT,
         space=6):
    """items: str or list of str / (str, level) / (str, level, dict). Level >= 1 gets a bullet."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    if isinstance(items, str):
        items = [items]
    for i, it in enumerate(items):
        if isinstance(it, str):
            it = (it, 0, {})
        elif len(it) == 2:
            it = (it[0], it[1], {})
        s, lvl, opt = it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        if lvl >= 1:
            bullet(p, lvl)
        # **bold** segments
        for k, seg in enumerate(re.split(r'\*\*', sci(s))):
            if not seg:
                continue
            r = p.add_run()
            r.text = seg
            run_fmt(r, opt.get('size', size - 2 * max(lvl - 1, 0)), bold=bold or k % 2 == 1 or opt.get('bold', False),
                    color=opt.get('color', color), italic=opt.get('italic', False))
    return tb


def bullet(p, lvl):
    from pptx.oxml.ns import qn
    pPr = p._p.get_or_add_pPr()
    ind = 0.28 * lvl
    pPr.set('marL', str(int(Inches(ind))))
    pPr.set('indent', str(int(-Inches(0.22))))
    for tag in ('a:buNone', 'a:buChar', 'a:buAutoNum'):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    bu = pPr.makeelement(qn('a:buChar'), {'char': '•' if lvl == 1 else '–'})
    pPr.append(bu)


def new_slide(prs, title, message=None, notes=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    text(s, 0.5, 0.3, W - 1.0, 0.7, title, size=28, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    if message:
        text(s, 0.5, 0.98, W - 1.0, 0.55, message, size=18, color=BLUE, anchor=MSO_ANCHOR.TOP)
    if notes:
        s.notes_slide.notes_text_frame.text = sci(notes)
    return s


def picture(slide, path, x, y, w, h):
    """Fit image into the box, keep aspect, centred."""
    iw, ih = Image.open(path).size
    sc = min(w / iw, h / ih)
    pw, ph = iw * sc, ih * sc
    return slide.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def table(slide, x, y, w, rows, colw, size=12, rowh=0.3):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(rowh * nr))
    tbl = shp.table
    tbl.first_row = False
    tbl.horz_banding = False
    tot = sum(colw)
    for j, cw in enumerate(colw):
        tbl.columns[j].width = Inches(w * cw / tot)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(rowh)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = HEAD if i == 0 else (BAND_ROW if i % 2 == 0 else WHITE)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            bold = i == 0
            if val.startswith('**') and val.endswith('**'):
                val, bold = val[2:-2], True
            r = tf.paragraphs[0].add_run()
            r.text = sci(val)
            run_fmt(r, size, bold=bold)
    return tbl


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    # 0 title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, 0.8, 2.3, W - 1.6, 1.4, 'Thesis data set, and why the absorber is not learned above its pole',
         size=36, bold=True, anchor=MSO_ANCHOR.BOTTOM)
    text(s, 0.8, 3.9, W - 1.6, 0.6, 'Model augmentation for a dual-gantry motion system  |  meeting 1 October 2026',
         size=18, color=GREY)

    # 1 goal
    s = new_slide(prs, 'Goal of this meeting', notes=(
        'Four slides on the data set (records, friction, noise, band), then the first results and the main issue. '
        'The last slide lists the decisions I need.'))
    for k, (hd, body) in enumerate([
            ('Show the thesis data set', 'records, friction, measurement noise, excitation band'),
            ('Discuss the main issue', 'the added states do not learn the absorber resonance; above the 212 Hz pole '
                                       'the augmented model is worse than it should be'),
            ('Leave with decisions', 'on the questions of the last slide')]):
        y = 1.7 + 1.5 * k
        text(s, 0.8, y, 0.8, 1.0, str(k + 1), size=44, bold=True, color=BLUE)
        text(s, 1.7, y + 0.1, 10.5, 1.2, [(hd, 0, {'bold': True, 'size': 22}), (body, 0, {'color': GREY, 'size': 18})])

    # 2 training / validation sets
    s = new_slide(prs, 'Data sets: training and validation',
                  'Training covers Y from -3.0e-1 to 3.0e-1 m; validation stays inside that range',
                  notes=('Every record shares one truth, one controller and one timing. Validation only selects the '
                         'checkpoint; test records are never used for any choice. Stroke of the machine: X +-3.75e-1 m, '
                         'Y +-4.0e-1 m, so there is room outside the training range for extrapolation tests.'))
    text(s, 0.5, 1.6, W - 1.0, 0.8, [
        '**Every record:** same truth (FP baseline + payload absorber + tanh rail friction), controller K1 (designed '
        'at Y = 0), 12 s at 20 kHz, trained at 4 kHz, plus a noise-free twin.',
        '**Stroke:** X ±3.75e-1 m, Y ±4.0e-1 m.'], size=15, space=3)
    table(s, 0.5, 2.55, W - 1.0, [
        ['Set', 'Records', 'Y covered [m]', 'X covered [m]', 'Motion', 'Multisine'],
        ['Training', '5 standstill', '-3.0e-1, -1.5e-1, 0, 1.5e-1, 3.0e-1', '0', 'none', 'yes'],
        ['Training', '3 Y sweeps', '-3.0e-1 to 3.0e-1', '0', 'sine 0.2 / 0.5 / 0.75 Hz', 'yes'],
        ['Training', '4 S-curve moves', '-3.0e-1 to 3.0e-1', '-2.8e-1 to 2.8e-1', 'point-to-point, 25 to 75 % of max acc.', 'yes'],
        ['Training', '2 Lissajous', '-3.0e-1 to 3.0e-1', '-2.8e-1 to 2.8e-1', 'combined X and Y sines', 'yes'],
        ['Training', '4 ILC-shape routes', '-3.0e-1 to 3.0e-1', '-8.0e-2 to 8.0e-2', 'measured ASMPT move shape, 45 and 75 % of max', 'no'],
        ['Validation', '1 standstill', '1.0e-1', '0', 'none', 'yes'],
        ['Validation', 'sweep, S-curve, Lissajous', '-2.5e-1 to 3.0e-1', '-2.8e-1 to 2.8e-1', 'new centres and setpoints', 'yes'],
        ['Validation', '2 ILC-shape routes', '-2.8e-1 to 2.8e-1', '-8.0e-2 to 8.0e-2', 'stops at Y no training route uses', 'no'],
    ], [1.1, 1.9, 2.6, 1.9, 3.4, 0.9], size=13, rowh=0.42)
    text(s, 0.5, 6.55, W - 1.0, 0.5, 'Validation is only used to select the checkpoint.', size=15, color=GREY)

    # 3 test sets
    s = new_slide(prs, 'Data sets: test records',
                  'Each extrapolation record (E) differs from its interpolation partner (I) in exactly one quantity',
                  notes=('Because E and its I partner differ in one quantity, the change in error is attributable to that '
                         'quantity. Honest caveat: I2 is held out as a standstill point, but training moves pass through '
                         'its Y; the truly unseen positions are only |Y| > 3.0e-1 m (E1, E2). TF records are only for the '
                         'FRF check (result R3).'))
    table(s, 0.5, 1.6, W - 1.0, [
        ['Record', 'Partner', 'What is new', 'Y covered [m]', 'Multisine'],
        ['I1', '', 'S-curve moves as in training, new phases and setpoints', '-3.0e-1 to 2.9e-1', 'yes'],
        ['I2', '', 'standstill between two training points', '-2.25e-1', 'yes'],
        ['I3, I4, I5', '', 'ASMPT routes: X and Y together, X-only / Y-only, S-curve', '-1.7e-1 to 1.5e-1', 'no'],
        ['I6', '', 'long strokes at 1.5 m/s', '-2.8e-1 to 2.8e-1', 'no'],
        ['**E1**', 'I2', 'position beyond training (standstill)', '-3.9e-1', 'yes'],
        ['**E2**', 'I3', 'position beyond training while moving', '-3.3e-1 to 3.9e-1', 'no'],
        ['**E3**', 'I5', 'acceleration 30 / 50 m/s² (training max 22.5 / 37.5)', 'as I5', 'no'],
        ['**E4**', 'I6', 'velocity 1.95 m/s (training max 1.5)', 'as I6', 'no'],
        ['**E5**', 'I3', 'controller gains +20 %', 'as I3', 'no'],
        ['**E6**', 'I3', 'measured ILC move, X acceleration 1.31 × max (stress case)', 'as I3', 'no'],
        ['TF-1 to TF-3', '', 'FRF check: standstill, 4 phase realisations each', '1.5e-1, 2.25e-1, 3.5e-1', 'yes'],
    ], [1.5, 1.0, 6.0, 2.6, 1.2], size=13, rowh=0.38)
    text(s, 0.5, 6.35, W - 1.0, 0.8, [
        ('Truly unseen positions are only |Y| > 3.0e-1 m (E1, E2): training moves pass through the Y of I2.', 1),
        ('Test records are never used for any choice.', 1)], size=15, color=GREY, space=2)

    # 4 friction
    s = new_slide(prs, 'Friction in the truth',
                  'Rail friction is Coulomb, smoothed with tanh, and only in the truth (the baseline has none)',
                  notes=('Why smooth: a set-valued stick law made the simulated stage stick-slip at standstill, which the '
                         'machine does not do. How to read the figure: at g = 1000 the curve is flat (Coulomb) where the '
                         'rails move under the multisine, at g = 100 it is still rising, so it acts as a viscous damper. '
                         'At g = 1000 the stage still settles at standstill (6e-11 / 1e-10 / 1.7e-9 m); g = 3000 already '
                         'creeps. ASMPT: higher is more accurate. Limit: below 1e-3 m/s it is a steep damper, slope cc g '
                         'about 1.7e4 N s/m, not a stick.'))
    text(s, 0.5, 1.7, 5.4, 5.3, [
        ('Fᵢ = ccᵢ · tanh(g · vᵢ)  per rail X1, X2, Y', 0, {'bold': True}),
        ('cc = 16.8 / 18.35 / 11.6 N (Garcia)', 1),
        ('Why smooth', 0, {'bold': True}),
        ('a hard stick law made the simulation stick-slip at standstill; the machine does not', 1),
        ('Why g = 1000 s/m', 0, {'bold': True}),
        ('multisine moves the rails at 7e-3 to 8e-3 m/s rms', 1),
        ('g = 100: still rising there, acts as a damper', 1),
        ('g = 1000: flat (Coulomb) there, stage still settles', 1),
        ('g = 3000: stage creeps at standstill', 1),
        ('Limit', 0, {'bold': True}),
        ('below 1e-3 m/s it is a steep damper (≈ 1.7e4 N s/m), not a stick', 1),
    ], size=16, space=4)
    picture(s, os.path.join(FIG, 'F3_friction_tanh.png'), 6.1, 1.8, 6.8, 5.2)

    # 5 noise
    s = new_slide(prs, 'Measurement noise',
                  'Measured standstill noise is injected on the encoder, so the simulated servo error matches the machine',
                  notes=('The controller sees y = q + v, reacts to the noise and moves the plant, as on the machine; '
                         'training reads the measured y. Inside the loop the recorded error is e = -S v, so the loop '
                         'reshapes the noise; we inject v with spectrum Phi_meas / |S|^2. Below the loop bandwidth |S| is '
                         'very small, so that correction would give a large slow encoder error that the controller would '
                         'follow and physically move the stage; so below 50 Hz the measured spectrum is injected without '
                         'correction. Cost: the simulated error there is 0.7 to 0.9 of the measured, under 1 % of the total '
                         'variance. 50 Hz is a heuristic (noise design note CN-012), to confirm with Jasper. Check (what '
                         'Jasper asked, shape): per band above 50 Hz the ratio simulated / measured is 0.97 to 1.03.'))
    text(s, 0.5, 1.7, 6.3, 5.3, [
        ('Where', 0, {'bold': True}),
        ('controller sees y = q + v: it reacts to the noise and moves the stage, as on the machine', 1),
        ('Source', 0, {'bold': True}),
        ('standstill parts of 145 Telica logs, rms X1 9.8e-9, X2 1.04e-8, Y 6.3e-9 m', 1),
        ('Shape', 0, {'bold': True}),
        ('the loop filters the noise (e = −S v), so inject Φₘₑₐₛ / |S|²', 1),
        ('below 50 Hz: no correction, else the stage would follow a large slow encoder error', 1),
        ('cost below 50 Hz: 0.7 to 0.9 of measured, < 1 % of the variance', 2),
        ('Check', 0, {'bold': True}),
        ('above 50 Hz, simulated / measured = 0.97 to 1.03 per band', 1),
        ('total rms within 1 % on all three axes', 1),
    ], size=16, space=4)
    picture(s, os.path.join(FIG, 'F1_noise_shape.png'), 7.0, 1.55, 5.9, 5.8)

    # 6 band
    s = new_slide(prs, 'Choosing the multisine band from the FRF',
                  'The multisine (106 to 297 Hz) excites where truth and baseline differ, above the controller bandwidth',
                  notes=('Step 1: closed-loop FRF of the truth (with friction, so its BLA) with K1 at the five training Y, '
                         'broadband 1 Hz to 1 kHz, 10 phase realisations; subtract the baseline FRF. Step 2: a difference '
                         'counts where it exceeds the measurement spread (2.45 sigma, 95 %). Step 3: below the 106 Hz '
                         'crossover the motion references already excite the differences (70 to 85 % resolved) and all ten '
                         'parameter combinations are separable with the references alone. Step 4: 90 % is a choice; 80 % '
                         'gives 111 to 276 Hz, 95 % gives 106 to 323 Hz. Above 297 Hz the difference falls off steeply '
                         '(2.3e-7 m/N at 250 Hz to about 3e-9 m/N at 1 kHz). The same band serves the nominal and the 10 % '
                         'detuned baseline. Multisine: 383 lines on a 0.5 Hz grid (repeats every 2 s), random phases, lowest '
                         'crest factor of 30 draws, 240 N / 87 N m / 180 N, inside all force and position limits. Bridge: '
                         'the largest difference sits at 230 to 297 Hz on Y, above the 212 Hz pole; that is where the model '
                         'fails.'))
    text(s, 0.5, 1.7, 5.6, 5.3, [
        ('1. Measure', 0, {'bold': True}),
        ('closed-loop FRF of the truth, K1, 5 training Y; subtract the baseline', 1),
        ('2. Keep only real differences', 0, {'bold': True}),
        ('larger than the measurement spread (2.45σ, 95 %)', 1),
        ('3. Lower edge 106 Hz = controller crossover', 0, {'bold': True}),
        ('below it the motion references already excite the differences', 1),
        ('4. Upper edge 297 Hz', 0, {'bold': True}),
        ('narrowest band holding 90 % of the remaining difference', 1),
        ('Result', 0, {'bold': True}),
        ('band holds the anti-resonance (150 Hz), the absorber pole (212 Hz) and the closed-loop peak (264 Hz)', 1),
        ('largest difference: 230 to 297 Hz, above the pole', 1),
    ], size=16, space=4)
    picture(s, os.path.join(FIG, 'F2_frf_band.png'), 6.2, 1.55, 6.8, 5.8)

    # 7 first results
    s = new_slide(prs, 'First results: runs 41 to 49',
                  'The augmentation lowers the validation error by 30 to 40 %, but the added states contribute little',
                  notes=('U = unconstrained augmentation, OBC = orthogonal projection. Numbers: best validation free-run rms '
                         'error from the server logs of 30 September; the OBC runs were not finished. Key point: OBC with 0 '
                         'added states (run 47) reaches the same level as OBC with 2 (run 41). Update from the server before '
                         'the meeting if possible.'))
    table(s, 0.5, 1.7, 8.2, [
        ['Run', 'Arm', 'Added states', 'Seed', 'Start [m]', 'Best [m]', 'Progress'],
        ['42', 'U', '2', '1', '1.68e-5', '1.027e-5', 'done'],
        ['44', 'U', '2', '2', '1.48e-5', '1.029e-5', 'done'],
        ['46', 'U', '2', '3', '1.48e-5', '1.020e-5', 'done'],
        ['41', 'OBC', '2', '1', '1.68e-5', '1.156e-5', '39 %'],
        ['43', 'OBC', '2', '2', '1.48e-5', '1.149e-5', '36 %'],
        ['45', 'OBC', '2', '3', '1.48e-5', '1.051e-5', '82 %'],
        ['**47**', '**OBC**', '**0**', '**1**', '**1.68e-5**', '**1.150e-5**', '**97 %**'],
        ['48', 'OBC', '4', '1', '1.68e-5', '1.108e-5', '49 %'],
        ['49', 'OBC', '8', '1', '1.68e-5', 'none yet', '3 %'],
    ], [0.7, 0.8, 1.3, 0.7, 1.3, 1.3, 1.1], size=14, rowh=0.44)
    text(s, 9.1, 1.7, 3.8, 4.5, [
        ('Best validation free-run rms error', 0, {'color': GREY, 'size': 14}),
        ('U = unconstrained, OBC = orthogonal projection', 0, {'color': GREY, 'size': 14}),
        ('Run 47 has no added states, yet matches run 41 (2 states)', 0, {'bold': True}),
        ('so the added states add almost nothing yet', 0),
    ], size=17, space=10)

    # 8 main issue
    s = new_slide(prs, 'Main issue: the absorber is not learned above its pole',
                  'Below the 212 Hz pole the model removes about 90 % of the absorber error, above it only 22 %',
                  notes=('The figure shows learned correction divided by the correction the truth needs, Y channel, on the '
                         'multisine validation records; 1 is perfect. It is 1.0 at 150 Hz, 0.60 at 212, 0.26 at 264 and '
                         '0.42 at 296 Hz: a broad over-damped bump, not the sharp pole. Coherence is 0.98 to 1.00, so it is '
                         'systematic, not noise. Same shape in runs 42, 44 and 48 (U and OBC, 2 and 4 states, two seeds), '
                         'within 0.02 in gain. The band numbers are Y error energy removed, run 42.'))
    picture(s, os.path.join(FIG, 'F4_absorber_learned.png'), 0.4, 1.65, 8.4, 4.8)
    text(s, 9.2, 1.7, 3.8, 0.5, 'Y error energy removed, run 42', size=15, color=GREY)
    for k, (band, pct, c) in enumerate([('106 to 140 Hz', '90 %', BLUE), ('140 to 230 Hz', '89 %', BLUE),
                                         ('230 to 297 Hz', '22 %', RGBColor(0xC0, 0x39, 0x2B))]):
        y = 2.25 + 1.25 * k
        text(s, 9.2, y, 1.9, 0.9, pct, size=40, bold=True, color=c, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 11.1, y, 2.0, 0.9, band, size=16, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.5, 6.5, W - 1.0, 0.6, 'Same shape in runs 42, 44, 48 (U and OBC, 2 and 4 states, two seeds): systematic.',
         size=16, bold=True)

    # 9 diagnosis
    s = new_slide(prs, 'What causes it (diagnosis, 30 September)',
                  'An optimisation problem: the model class and the objective both allow the resonance',
                  notes=('Ruled out, one line each in the backup: data, noise, friction, controller, weighting, too few '
                         'epochs, capacity, seed, OBC, pipeline. With the static path switched off (local test) the added '
                         'states learn the same static map with one sample delay; growing a resonance from there is still '
                         'uphill. So whatever path learns first learns the memoryless part of the absorber, and from there '
                         'the resonance is uphill. Side finding: the OBC reference set uses x_a = 0, so the projection '
                         'constrains only the static part; to check.'))
    table(s, 0.5, 1.7, W - 1.0, [
        ['Finding', 'Evidence'],
        ['The added states realise no resonance', 'only fast real poles (|z| ≤ 0.6), no pair near 212 Hz, in any checkpoint'],
        ['The model class can', 'an exact member with the 212 Hz pole has 7× lower training loss'],
        ['The objective rewards it', 'the truth scores 3.8× better on the windowed loss, even with its absorber state unknown at each window start'],
        ['What blocks it', 'the learned static correction takes the absorber’s place first; a correct absorber on top gains 15 %, without the static part 94 %'],
        ['Static path switched off', 'the added states learn the same static map, one sample delayed; the resonance is still uphill'],
    ], [3.2, 9.1], size=15, rowh=0.62)
    text(s, 0.5, 5.75, W - 1.0, 1.4, [
        ('Whatever path learns first takes the memoryless part of the absorber; from there the resonance is uphill.', 0,
         {'bold': True}),
        ('Ruled out: data, noise, friction, controller, weighting, epochs, capacity, seed, OBC, pipeline (backup).', 0,
         {'color': GREY, 'size': 15}),
    ], size=17, space=6)

    # 10 options
    s = new_slide(prs, 'Options (for discussion, nothing launched)',
                  'Four ways forward; the first alone is not enough according to the local test',
                  notes=('Option 2 refers to Hoekstra arXiv:2602.17297, Eq. 29: a ResNet form with a linear part, with the '
                         'added states co-estimated in the encoder. Option 3: our learning rate is 1e-5 against Hoekstra\'s '
                         '1e-3. Option 4: report that the augmentation fits the absorber below the pole and that the added '
                         'states stay static, with the error stated.'))
    for k, (hd, body) in enumerate([
            ('Two-phase training', 'static path off first; local test: not sufficient on its own'),
            ('Hoekstra parameterisation (arXiv:2602.17297, Eq. 29)', 'ResNet form with a linear part, added states co-estimated in the encoder'),
            ('Faster start-up', 'learning rate 1e-5 now, against 1e-3 in Hoekstra'),
            ('Accept and report', 'the augmentation fits the absorber below the pole; the added states stay static')]):
        y = 1.8 + 1.25 * k
        text(s, 0.8, y, 0.8, 1.0, str(k + 1), size=36, bold=True, color=BLUE)
        text(s, 1.6, y + 0.05, 11.0, 1.1, [(hd, 0, {'bold': True, 'size': 20}), (body, 0, {'color': GREY, 'size': 17})],
             space=2)

    # 11 questions
    s = new_slide(prs, 'Questions for the supervisors', notes=(
        'Settling: show a fixed 2e-1 to 4e-1 s post-move interval, or leave it out. Jasper and Dragan for the ASMPT '
        'routes and the noise. Phase margin: 28.5 deg with K1, 25 deg with K2.'))
    text(s, 0.5, 1.4, W - 1.0, 5.6, [
        ('Which option of the previous slide, or another idea from experience with added states?', 1),
        ('Is a learned model without a resonance in its added states acceptable for the thesis, if the error is stated?', 1),
        ('Noise: is the 50 Hz corner acceptable, and the 10 % tolerance? (Jasper)', 1),
        ('Settling: show a fixed 2e-1 to 4e-1 s interval after each move, or leave it out?', 1),
        ('Are the ASMPT routes I3 to I6 acceptable as ASMPT motion? (Jasper, Dragan)', 1),
        ('Is the Y-loop phase margin of 28.5° with K1 (25° with K2) acceptable?', 1),
    ], size=20, space=16)

    # backup divider
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, 0.8, 3.0, W - 1.6, 1.2, 'Backup', size=36, bold=True, anchor=MSO_ANCHOR.MIDDLE)

    # B1 results
    s = new_slide(prs, 'Backup: thesis results and the records they use',
                  notes='Confirm this list matches the 28 September agreement. R5 needs a separate truth that is not generated yet.')
    table(s, 0.5, 1.4, W - 1.0, [
        ['Result', 'Question', 'Records'],
        ['R1', 'does Y scheduling (LPV) beat a frozen LTI baseline', 'I2 with E1, I3 with E2'],
        ['R2', 'how many added states the absorber needs (0, 2, 8)', 'chosen on validation, reported on I1 to I6'],
        ['R3', 'does the trained model reproduce the plant FRF (BLA)', 'TF-1 to TF-3'],
        ['R4', 'does OBC keep the physical parameters interpretable', 'training (parameters), I1 to I6 (accuracy)'],
        ['R5', 'true parameter recovery with an orthogonal addition', 'needs a separate truth, not generated'],
        ['R6', 'generalisation against the black box and LTI', 'I / E pairs, cross-coupling on I4'],
        ['R7', 'controller transfer', 'E5 against I3'],
        ['R8', 'training effort and robustness against the black box', 'all runs'],
    ], [1.0, 6.4, 4.9], size=15, rowh=0.52)

    # B2 downsampling
    s = new_slide(prs, 'Backup: from 20 kHz data to 4 kHz training',
                  'Filter before downsampling, so the noise does not fold into the band',
                  notes=('Filtering the full position instead of the servo error would distort it by 5.5e-8 to 1.7e-7 m. '
                         'The 1.6 kHz cut-off is above 5 x the 212 Hz absorber pole (rule of Janot et al. 2019).'))
    text(s, 0.5, 1.7, W - 1.0, 5.3, [
        ('Taking every 5th sample would fold the noise between 2 and 10 kHz into the band (+28 to 40 % rms)', 1),
        ('So y is low-pass filtered first: 8th-order Butterworth at 1.6 kHz, forward and backward (no phase shift)', 1),
        ('applied to y − r (the servo error), then r is added back', 2),
        ('Forces are averaged over each 4 kHz sample interval; 5e-3 s is cut at both record ends', 1),
        ('The same filtered positions feed everything: rollouts, normalisation, OBC basis velocities', 1),
        ('1.6 kHz > 5 × the 212 Hz pole (Janot et al. 2019); basis angle to the noise-free basis: sin 1.9e-3', 2),
    ], size=18, space=10)

    # B3 diagnosis decision table
    s = new_slide(prs, 'Backup: causes ruled out', notes='Source: scripts/gantry/absorber-learning-diagnosis/DIAGNOSIS.md.')
    table(s, 0.5, 1.2, W - 1.0, [
        ['Candidate cause', 'Verdict', 'Deciding number'],
        ['Too few epochs', 'refuted', 'curves flat in last quarter (run 42: 1.031e-5 to 1.027e-5 m)'],
        ['Nothing to learn', 'refuted', 'ceiling 1.42e-5 m = 124 % of the no-added-state plateau'],
        ['Band above the pole', 'supported', '230 to 297 Hz: 2.52e-5 to 2.23e-5 m (22 %)'],
        ['Friction masks absorber', 'refuted', 'friction / absorber signal 0.01 to 0.03 in 140 to 230 Hz'],
        ['Noise', 'refuted', 'noise floor 5e-8 to 1.2e-7 m in band'],
        ['Controller', 'refuted', 'absorber visibility 1.02×'],
        ['Parameter substitution', 'refuted', 'tangent span 34 % (thesis) vs 65 % (old data)'],
        ['OBC projection', 'refuted', 'OBC and U give the same response'],
        ['Capacity (2 to 4 states)', 'refuted', '|H| within 0.02 at 250 to 280 Hz'],
        ['Seed', 'refuted', 'run 44 within 0.02 of run 42'],
        ['Closed-loop peak 264 Hz', 'refuted', '|S_YY| smooth 1.28 to 1.56'],
        ['Pipeline change', 'refuted', 'old checkpoint reproduced: 5.8588e-6 m'],
        ['Objective prefers damped shape', 'refuted', 'training loss: truth 1.35e-9 vs model 5.09e-9'],
    ], [3.4, 1.4, 7.5], size=13, rowh=0.41)

    # B4 excitation adequacy
    s = new_slide(prs, 'Backup: excitation adequacy and band weight',
                  notes='Figure: cumulative resolved difference weight over frequency; band shaded. 95 % is reached at 323 Hz.')
    text(s, 0.5, 1.4, W - 1.0, 1.8, [
        ('Multisine at least 67 dB above the noise-induced force in band', 1),
        ('Motion references 36 to 51 dB below the crossover', 1),
        ('All ten parameter combinations separable (Brun γ = 1.27)', 1),
    ], size=18, space=6)
    picture(s, os.path.join(BAND, 'fig12_band_weight.png'), 0.8, 3.2, W - 1.6, 4.0)

    # B5 full difference figure
    p = os.path.join(BAND, 'fig11_differences_avg.png')
    if os.path.exists(p):
        s = new_slide(prs, 'Backup: all FRF differences (3 × 3)')
        picture(s, p, 0.5, 1.2, W - 1.0, 6.0)

    prs.save(OUT)
    print('saved', os.path.abspath(OUT), len(prs.slides), 'slides')


if __name__ == '__main__':
    build()
