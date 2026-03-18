import subprocess, xml.etree.ElementTree as ET

slx = r'C:\Users\20203253\OneDrive - TU Eindhoven\Graduation Project\Baseline FP model\Baseline-LPV-Augmentation\kamtin-fp-model\03 Simulink gantry\gantry_2025a.slx'

result = subprocess.run(['unzip', '-p', slx, 'simulink/systems/system_root.xml'],
                        capture_output=True, text=True)
root = ET.fromstring(result.stdout)

# Print all lines with Src and Dst
print("=== ALL CONNECTIONS ===")
for line in root.iter('Line'):
    parts = {}
    for p in line.findall('P'):
        parts[p.get('Name')] = p.text
    # also check Branch children
    branches = []
    for branch in line.findall('Branch'):
        bp = {p.get('Name'): p.text for p in branch.findall('P')}
        branches.append(bp)
    if parts:
        src = parts.get('Src', '')
        dst = parts.get('Dst', '')
        name = parts.get('Name', '')
        print(f"  {src} -> {dst}  [{name}]")
        for bp in branches:
            print(f"    branch -> {bp.get('Dst','?')} [{bp.get('Name','')}]")

print("\n=== system_47 ALL CONNECTIONS ===")
result2 = subprocess.run(['unzip', '-p', slx, 'simulink/systems/system_47.xml'],
                         capture_output=True, text=True)
sub = ET.fromstring(result2.stdout)
print("All blocks:")
for block in sub.iter('Block'):
    name = block.get('Name', '')
    btype = block.get('BlockType', '')
    params = {p.get('Name'): p.text for p in block.findall('P')}
    print(f"  {btype}: {name}")
    for k,v in params.items():
        if k in ('FunctionName', 'Script', 'Gain', 'InitialCondition', 'Port', 'Indices', 'InputPortWidth', 'Value'):
            print(f"    {k} = {v}")
