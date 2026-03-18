import subprocess, xml.etree.ElementTree as ET

def parse_system(slx_path, system_file):
    result = subprocess.run(
        ['unzip', '-p', slx_path, f'simulink/systems/{system_file}'],
        capture_output=True, text=True
    )
    return ET.fromstring(result.stdout)

slx = r'C:\Users\20203253\OneDrive - TU Eindhoven\Graduation Project\Baseline FP model\Baseline-LPV-Augmentation\kamtin-fp-model\03 Simulink gantry\gantry_2025a.slx'

root = parse_system(slx, 'system_root.xml')

print("=== TO WORKSPACE blocks (what signals are saved as q) ===")
for block in root.iter('Block'):
    btype = block.get('BlockType', '')
    name = block.get('Name', '')
    if btype == 'ToWorkspace':
        params = {p.get('Name'): p.text for p in block.findall('P')}
        print(f"\n  ToWorkspace: '{name}'")
        for k, v in params.items():
            print(f"    {k} = {v}")

print("\n=== SIGNAL LINES connected to Selector1 and Selector2 ===")
for line in root.iter('Line'):
    src = line.find('P[@Name="Src"]')
    dst = line.find('P[@Name="Dst"]')
    name_elem = line.find('P[@Name="Name"]')
    if src is not None and dst is not None:
        s, d = src.text, dst.text
        n = name_elem.text if name_elem is not None else ''
        if 'Selector' in s or 'Selector' in d:
            print(f"  {s} -> {d}  name='{n}'")

print("\n=== SIGNAL LINES connected to To Workspace blocks ===")
for line in root.iter('Line'):
    src = line.find('P[@Name="Src"]')
    dst = line.find('P[@Name="Dst"]')
    name_elem = line.find('P[@Name="Name"]')
    if src is not None and dst is not None:
        s, d = src.text, dst.text
        n = name_elem.text if name_elem is not None else ''
        if 'Workspace' in s or 'Workspace' in d or 'To Workspace' in s or 'To Workspace' in d:
            print(f"  {s} -> {d}  name='{n}'")

print("\n=== Single H-gantry subsystem outports ===")
# Look at system_47 (the large one, likely Single H-gantry)
sub = parse_system(slx, 'system_47.xml')
print("Blocks in system_47:")
for block in sub.iter('Block'):
    btype = block.get('BlockType', '')
    name = block.get('Name', '')
    if btype in ('Outport', 'Inport', 'SubSystem', 'Integrator', 'MATLABFcn', 'MATLAB Function'):
        params = {p.get('Name'): p.text for p in block.findall('P')}
        print(f"  {btype}: {name}")
        for k,v in params.items():
            if k in ('Port', 'InitialCondition', 'FunctionName'):
                print(f"    {k} = {v}")
