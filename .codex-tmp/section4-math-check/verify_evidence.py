from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
snippets = [
 ('C1','scripts/gantry/gantry_dynamic/obc_gantry.py',84,84),
 ('C2','scripts/gantry/gantry_dynamic/obc_gantry.py',126,134),
 ('C3','scripts/gantry/gantry_dynamic/obc_gantry.py',141,147),
 ('C4','scripts/gantry/gantry_dynamic/obc_gantry.py',293,301),
 ('C5','scripts/gantry/gantry_dynamic/obc_gantry.py',322,324),
 ('C6','scripts/gantry/gantry_dynamic/obc_gantry.py',349,350),
 ('C7','model_augmentation/fit_systems/obc.py',247,248),
 ('C8','model_augmentation/fit_systems/obc.py',449,450),
 ('C9','model_augmentation/fit_systems/obc.py',501,509),
 ('C10','model_augmentation/fit_systems/interconnect.py',168,170),
 ('C11','model_augmentation/fit_systems/interconnect.py',1406,1409),
 ('C12','model_augmentation/fit_systems/interconnect.py',1364,1366),
 ('C13','model_augmentation/fit_systems/interconnect.py',1148,1149),
 ('C14','model_augmentation/fit_systems/ps2_opening.py',185,189),
 ('C15','model_augmentation/fit_systems/ps2_opening.py',152,155),
 ('C16','model_augmentation/fit_systems/ps2_opening.py',197,200),
 ('C17','model_augmentation/fit_systems/ps2_opening.py',226,229),
 ('C18','model_augmentation/fit_systems/lbfgs_polish.py',87,89),
 ('C19','scripts/gantry/gantry_dynamic/training.py',698,698),
 ('C20','model_augmentation/fit_systems/blocks.py',1402,1407),
 ('C21','model_augmentation/fit_systems/blocks.py',1442,1445),
 ('C22','model_augmentation/fit_systems/blocks.py',926,929),
 ('C23','model_augmentation/fit_systems/blocks.py',967,968),
 ('C24','model_augmentation/fit_systems/blocks.py',979,992),
 ('C25','model_augmentation/fit_systems/blocks.py',1530,1531),
 ('C26','scripts/gantry/gantry_interconnect_dynamic.py',333,333),
 ('C27','scripts/gantry/gantry_interconnect_dynamic.py',343,343),
 ('C28','model_augmentation/fit_systems/interconnect.py',1417,1418),
 ('C29','model_augmentation/fit_systems/interconnect.py',1350,1358),
 ('C30','model_augmentation/fit_systems/lbfgs_polish.py',612,615),
 ('C31','scripts/gantry/gantry_dynamic/model.py',207,207),
 ('C32','scripts/gantry/gantry_dynamic/model.py',217,218),
 ('C33','scripts/gantry/gantry_dynamic/data.py',337,338),
 ('C34','model_augmentation/fit_systems/blocks.py',825,835),
 ('C35','model_augmentation/fit_systems/ps2_opening.py',151,151),
 ('C36','scripts/gantry/gantry_dynamic/config.py',276,279),
 ('C37','model_augmentation/utils/utils.py',112,117),
 ('C38','model_augmentation/utils/utils.py',137,143),
 ('C39','model_augmentation/fit_systems/obc.py',464,475),
 ('C40','scripts/gantry/gantry_dynamic/obc_gantry.py',202,206),
 ('C41','scripts/gantry/gantry_dynamic/obc_gantry.py',210,212),
 ('C42','scripts/gantry/gantry_dynamic/training.py',690,698),
 ('C43','scripts/gantry/gantry_dynamic/training.py',746,747),
 ('C44','model_augmentation/fit_systems/lbfgs_polish.py',599,599),
 ('C45','model_augmentation/fit_systems/interconnect.py',887,889),
 ('C46','scripts/gantry/gantry_dynamic/model.py',360,360),
]
for key,path,start,end in snippets:
 lines=Path(path).read_text(encoding='utf-8').splitlines(True)
 q=root/f'{key}.txt'
 q.write_text(''.join(lines[start-1:end]),encoding='utf-8')
 p=subprocess.run([sys.executable,'scripts/verify_quote.py',path,str(start),str(end),str(q)],capture_output=True,text=True,encoding='utf-8')
 if p.returncode or 'MATCH OK' not in p.stdout:
  print(p.stdout,p.stderr); raise SystemExit(1)
 print(f'{key}: {path}:{start}-{end}: MATCH OK\n'+q.read_text(encoding='utf-8'),flush=True)
