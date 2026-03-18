import subprocess, xml.etree.ElementTree as ET

result = subprocess.run(
    ['unzip', '-p',
     r'C:\Users\20203253\OneDrive - TU Eindhoven\Graduation Project\Baseline FP model\Baseline-LPV-Augmentation\kamtin-fp-model\03 Simulink gantry\gantry_2025a.slx',
     'simulink/systems/system_root.xml'],
    capture_output=True, text=True
)

root = ET.fromstring(result.stdout)

print("=== ALL BLOCKS ===")
for block in root.iter('Block'):
    name = block.get('Name', '')
    btype = block.get('BlockType', '')
    params = {p.get('Name'): p.text for p in block.findall('P')}
    if 'Selector' in name or btype == 'Selector' or 'selector' in name.lower():
        print(f"\nSELECTOR BLOCK: {btype}: {name}")
        for k, v in params.items():
            print(f"  {k} = {v}")

print("\n=== ALL OUTPORT BLOCKS ===")
for block in root.iter('Block'):
    btype = block.get('BlockType', '')
    name = block.get('Name', '')
    if btype == 'Outport' or 'q' in name.lower() or name in ['q', 'q1', 'q2', 'q3']:
        params = {p.get('Name'): p.text for p in block.findall('P')}
        print(f"\nOUTPORT: {btype}: {name}")
        for k, v in params.items():
            print(f"  {k} = {v}")

print("\n=== ALL BLOCK NAMES (brief) ===")
for block in root.iter('Block'):
    print(f"  {block.get('BlockType','?'):30s} {block.get('Name','')}")
