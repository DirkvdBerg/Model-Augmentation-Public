import subprocess, xml.etree.ElementTree as ET

slx = r'C:\Users\20203253\OneDrive - TU Eindhoven\Graduation Project\Baseline FP model\Baseline-LPV-Augmentation\kamtin-fp-model\03 Simulink gantry\gantry_2025a.slx'

result = subprocess.run(['unzip', '-p', slx, 'simulink/systems/system_root.xml'],
                        capture_output=True, text=True)
root = ET.fromstring(result.stdout)

# Build SID -> name map
sid_map = {}
for block in root.iter('Block'):
    sid_elem = block.find('P[@Name="SID"]')
    if sid_elem is not None:
        sid = sid_elem.text
        sid_map[sid] = f"{block.get('BlockType','?')}:{block.get('Name','?')}"

print("=== SID -> Block Name ===")
for sid, name in sorted(sid_map.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
    print(f"  {sid:5s} -> {name}")

print("\n=== CONNECTIONS (resolved) ===")
def resolve(port_str):
    if '#' in port_str:
        sid = port_str.split('#')[0]
        return sid_map.get(sid, f'SID:{sid}') + f" ({port_str})"
    return port_str

for line in root.iter('Line'):
    parts = {p.get('Name'): p.text for p in line.findall('P')}
    branches = [{p.get('Name'): p.text for p in b.findall('P')} for b in line.findall('Branch')]
    src = parts.get('Src', '')
    dst = parts.get('Dst', '')
    lname = parts.get('Name', '')
    if src:
        dsts = [resolve(dst)] if dst else []
        for bp in branches:
            if bp.get('Dst'):
                dsts.append(resolve(bp['Dst']))
        print(f"  {resolve(src)}")
        for d in dsts:
            print(f"    -> {d}  [{lname}]")
