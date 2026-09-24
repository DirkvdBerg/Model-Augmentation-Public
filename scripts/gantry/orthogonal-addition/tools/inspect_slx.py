"""Read-only inspection of a Simulink .slx (a zip of XML): blocks, lines and chart scripts.

Usage: python tools/inspect_slx.py <model.slx> [substring filter]
Nothing is extracted to disk; the archive is read in memory.
"""
__project_origin__ = "added"

import re
import sys
import zipfile

path = sys.argv[1]
filt = sys.argv[2] if len(sys.argv) > 2 else None
z = zipfile.ZipFile(path)
names = z.namelist()
print('[slx] %d members' % len(names))
for n in names:
    if n.endswith('.xml') and ('blockdiagram' in n or 'system' in n or 'stateflow' in n
                               or 'configSet' in n):
        s = z.read(n).decode('utf-8', 'replace')
        print('\n==== %s (%d chars)' % (n, len(s)))
        if 'configSet' in n:
            for key in ('SolverType', 'Solver', 'FixedStep', 'StopTime', 'MaxStep'):
                m = re.search(r'<P Name="%s"[^>]*>([^<]*)</P>' % key, s)
                if m:
                    print('   %s = %s' % (key, m.group(1)))
            continue
        for m in re.finditer(r'<Block BlockType="([^"]+)" Name="([^"]+)" SID="([^"]+)"', s):
            line = '   BLOCK %-22s %-40s SID %s' % m.groups()
            if filt is None or filt in line:
                print(line)
        for m in re.finditer(r'<Line>(.*?)</Line>', s, re.S):
            body = re.sub(r'\s+', ' ', m.group(1))
            src = re.search(r'<P Name="Src">([^<]*)</P>', body)
            dsts = re.findall(r'<P Name="Dst">([^<]*)</P>', body)
            print('   LINE %s -> %s' % (src.group(1) if src else '?', ', '.join(dsts)))
        for key in ('InitialCondition', 'Indices', 'InputPortWidth', 'VariableName',
                    'SampleTime', 'Gain'):
            for m in re.finditer(r'<P Name="%s">([^<]*)</P>' % key, s):
                print('   P %s = %s' % (key, m.group(1)))
        for m in re.finditer(r'<script>(.*?)</script>', s, re.S):
            print('   SCRIPT:\n' + m.group(1)[:1500])
