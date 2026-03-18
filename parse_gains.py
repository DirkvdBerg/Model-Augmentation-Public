import subprocess, xml.etree.ElementTree as ET

slx = r'C:\Users\20203253\OneDrive - TU Eindhoven\Graduation Project\Baseline FP model\Baseline-LPV-Augmentation\kamtin-fp-model\03 Simulink gantry\gantry_2025a.slx'
result = subprocess.run(['unzip', '-p', slx, 'simulink/systems/system_root.xml'], capture_output=True, text=True)
root = ET.fromstring(result.stdout)

print("GAIN BLOCKS:")
for block in root.iter('Block'):
    if block.get('BlockType') == 'Gain':
        sid = block.get('SID', '?')
        name = block.get('Name', '?')
        params = {p.get('Name'): p.text for p in block.findall('P')}
        print(f"  SID={sid} Name={name}")
        for k, v in params.items():
            if k in ('Gain', 'Multiplication'):
                print(f"    {k} = {v}")
