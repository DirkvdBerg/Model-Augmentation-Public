import subprocess, xml.etree.ElementTree as ET

slx = r'C:\Users\20203253\OneDrive - TU Eindhoven\Graduation Project\Baseline FP model\Baseline-LPV-Augmentation\kamtin-fp-model\03 Simulink gantry\gantry_2025a.slx'

result = subprocess.run(['unzip', '-p', slx, 'simulink/systems/system_root.xml'],
                        capture_output=True, text=True)
root = ET.fromstring(result.stdout)

# Print first block to see all attributes
for i, block in enumerate(root.iter('Block')):
    if i == 0:
        print("First block attributes:", block.attrib)
        print("First block children:", [c.tag for c in block])
    break

# Build SID -> name using all attributes
sid_map = {}
for block in root.iter('Block'):
    # Try 'SID' as attribute
    sid = block.get('SID', None)
    if sid:
        sid_map[sid] = f"{block.get('BlockType','?')}:{block.get('Name','?')}"

if sid_map:
    print("\n=== SID -> Block Name (from attrib) ===")
    for sid, name in sorted(sid_map.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        print(f"  {sid:5s} -> {name}")
else:
    print("\nNo SID attributes found. Printing all block attrib keys:")
    for block in root.iter('Block'):
        print("  attrs:", list(block.attrib.keys()))
        break

    # Try looking for the SIDs in all child elements
    print("\nAll P element names for first block:")
    for block in root.iter('Block'):
        for p in block:
            print(f"  tag={p.tag} attrib={p.attrib} text={p.text[:50] if p.text else ''}")
        break
