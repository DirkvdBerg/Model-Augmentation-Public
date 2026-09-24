"""Extract the 2026-09-22 literature synthesis from session transcript 8bf42294 (G0, CN-002).

    python tools/extract_literature.py list        # index of assistant text messages
    python tools/extract_literature.py write       # write LITERATURE.md from the selected messages

Selection rule (handoff 2026-09-23 section 9, G0): assistant text messages whose text starts with
"SQ1", "SQ4", "SQ2" or "All four agents are in", plus the two assistant answers of 2026-09-22
16:49 and 17:11 UTC (18:49 and 19:11 local). Verbatim; only a heading per message is added.
"""
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CN = os.path.dirname(HERE)
TRANSCRIPT = os.path.join(os.path.expanduser('~'), '.claude', 'projects',
                          'C--Users-20203253-OneDrive---TU-Eindhoven-Graduation-Project-Baseline-FP-'
                          'model-Baseline-LPV-Augmentation',
                          '8bf42294-1c7b-495d-a10a-e84592f5160f.jsonl')
PREFIXES = ('SQ1', 'SQ4', 'SQ2', 'All four agents are in')
UTC_TIMES = ('2026-09-22 16:49', '2026-09-22 17:11')


def messages():
    """[(local time str, utc time str, text)] of assistant text blocks, in order (sidechains excluded)."""
    out = []
    with open(TRANSCRIPT, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get('type') != 'assistant' or rec.get('isSidechain'):
                continue
            content = rec.get('message', {}).get('content', [])
            text = '\n'.join(c.get('text', '') for c in content
                             if isinstance(c, dict) and c.get('type') == 'text').strip()
            if not text:
                continue
            ts = rec.get('timestamp', '')
            try:
                t = dt.datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone()
                loc = t.strftime('%Y-%m-%d %H:%M')
                utc = t.astimezone(dt.timezone.utc).strftime('%Y-%m-%d %H:%M')
            except ValueError:
                loc = utc = ts
            out.append((loc, utc, text))
    return out


def select(msgs):
    sel = []
    for loc, utc, text in msgs:
        head = text.lstrip('#* \n')
        if head.startswith(PREFIXES) or utc in UTC_TIMES:
            sel.append((loc, text))
    return sel


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'list'
    msgs = messages()
    print(f'[extract] {len(msgs)} assistant text messages in the transcript')
    if mode == 'list':
        for i, (loc, utc, text) in enumerate(msgs):
            first = text.lstrip('#* \n')[:70].replace('\n', ' ')
            print(f'{i:3d} {loc} len={len(text):6d} | {first}')
        return
    sel = select(msgs)
    print(f'[extract] selected {len(sel)} messages: ' + ', '.join(loc for loc, _ in sel))
    lines = ['# Literature synthesis of 2026-09-22 (session 8bf42294), saved verbatim (G0, CN-002)', '',
             'Output of session `8bf42294-1c7b-495d-a10a-e84592f5160f` (the four-agent deep-research',
             'sweep of 2026-09-22), extracted from the transcript by `tools/extract_literature.py`.',
             'This is that session\'s text, not new work of this session. Selection rule: assistant',
             'messages starting "SQ1", "SQ4", "SQ2", "All four agents are in", plus the answers of',
             '2026-09-22 16:49 and 17:11 UTC (18:49 and 19:11 local). Headings show local time. The only edit:',
             'em and en dashes replaced (project dash rule).', '']
    for i, (loc, text) in enumerate(sel, 1):
        lines += [f'## Message {i} ({loc})', '', text, '']
    body = '\n'.join(lines)
    # Project rule: no em-dashes in any output. The only edit to the verbatim text.
    for bad, rep in ((chr(0x2014), ', '), (chr(0x2013), '-')):
        n = body.count(bad)
        if n:
            print(f'[extract] replaced {n} x U+{ord(bad):04X} (dash rule)')
            body = body.replace(bad, rep)
    with open(os.path.join(CN, 'LITERATURE.md'), 'w', encoding='utf-8') as fh:
        fh.write(body)
    print(f'[extract] wrote LITERATURE.md ({len(body)} chars)')


if __name__ == '__main__':
    main()
