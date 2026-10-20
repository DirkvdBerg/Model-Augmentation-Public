from pathlib import Path
import csv,hashlib,shutil

root=Path.cwd()
destination=root/'Thesis-writeup/References'
old=root/'literature/thesis-referenced'
new=root/'literature/thesis-reference-new'
entries=[
('beintema2023subnet','Beintema-2023-Deep-subspace-encoders.pdf',old/'beintema2023subnet.pdf','Published journal article','10.1016/j.automatica.2023.111210',''),
('downey2017psrnn','Downey-2017-Predictive-state-recurrent-neural-networks.pdf',new/'Downey-NIPS-2017-predictive-state-recurrent-neural-networks-Paper.pdf','Published NeurIPS proceedings paper','',''),
('drenth2025lpvlfr','Drenth-2026-LPV-models-linear-fractional-representation-arXiv-v1.pdf',new/'Roel-drenth-Efficient Learning of Affine and Rational Dependency LPV Models With Linear Fractional Representation.pdf','arXiv v1; accepted for IFAC WC 2026; no published version found','10.48550/arXiv.2605.12203','Citation key retains 2025; paper year is 2026. Section 4.1 differs from the old local manuscript.'),
('forssell1999revisited','Forssell-1999-Closed-loop-identification-revisited.pdf',new/'Urban-Forsell-Closed-loop identification revisited.pdf','Published journal article','10.1016/S0005-1098(99)00022-9','Citation currently occurs only in deleted text in the audit snapshot.'),
('garcia2013model','Garcia-Herreros-2013-Dual-drive-gantry-decoupling-control.pdf',new/'Garcia-Model-based decoupling control method for dual-drive gantry stages A case study with experimental validations.pdf','Published journal article','10.1016/j.conengprac.2012.10.010',''),
('gautier1990minimum','Gautier-1990-Minimum-inertial-parameters-serial-robots.pdf',new/'Direct_calculation_of_minimum_set_of_inertial_parameters_of_serial_robots.pdf','Published journal article; scanned PDF','10.1109/70.56655','Only one copy included; the second downloaded Gautier PDF is byte-identical.'),
('gyorok2025l4dc','Gyorok-2025-Orthogonal-projection-regularization.pdf',new/'gyorok25a-Orthogonal projection-based regularization for efficient model augmentation.pdf','Published PMLR proceedings paper','','PMLR 283, proceedings pages 166–178; internal PDF pages 1–13.'),
('gyorok2026obc','Gyorok-2026-Orthogonal-by-construction.pdf',new/'Orthogonal-by-construction augmentation of physics-based input–output models.pdf','Published journal article','10.1016/j.ifacsc.2026.100376',''),
('hefny2015supervised','Hefny-2015-Supervised-learning-dynamical-systems.pdf',new/'Hefny-NIPS-2015-supervised-learning-for-dynamical-system-learning-Paper.pdf','Published NeurIPS proceedings paper','',''),
('hoekstra2026encoder','Hoekstra-2026-Encoder-initialisation-arXiv-v1.pdf',old/'hoekstra2026encoder.pdf','arXiv v1; submitted to IFAC WC 2026; no published version found','10.48550/arXiv.2602.13108',''),
('hoekstra2026lfr','Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf',old/'hoekstra2026lfr.pdf','arXiv v1; submitted to Automatica; no published version found','10.48550/arXiv.2602.17297',''),
('irds2022','IRDS-2022-Executive-summary.pdf',old/'irds2022.pdf','Published IEEE roadmap report','10.60627/C13Z-V363',''),
('karniadakis2021physicsinformed','Karniadakis-2021-Physics-informed-machine-learning.pdf',old/'karniadakis2021physicsinformed.pdf','Published journal article','10.1038/s42254-021-00314-5',''),
('kessels2025ai','Kessels-2025-AI-based-model-updating.pdf',old/'kessels2025ai.pdf','Published PhD thesis','',''),
('noel2018greybox','Noel-2018-Grey-box-state-space-identification.pdf',new/'JP-Noël-and-J-Schoukens-Grey-box state-space identification of nonlinear mechanical vibrations.pdf','Published journal article','10.1080/00207179.2017.1308557',''),
('ovchinnikov2021computing','Ovchinnikov-2021-Identifiable-functions-ODE-models.pdf',new/'Computing all identifiable functions of parameters for ODE models.pdf','Published journal article','10.1016/j.sysconle.2021.105030',''),
('raissi2017physicsinformedpart1','Raissi-2017-Physics-informed-deep-learning-Part-I-arXiv-v1.pdf',old/'raissi2017physicsinformedpart1.pdf','arXiv v1; no standalone published version found','10.48550/arXiv.1711.10561','Part I; do not substitute the later combined journal article under this key.'),
('retzler2024','Retzler-2024-Physics-based-model-augmentation-robot.pdf',old/'retzler2024.pdf','Published journal article','10.1017/dce.2024.8',''),
('teo2007dynamic','Teo-2007-H-type-gantry-modeling-adaptive-control.pdf',old/'teo2007dynamic.pdf','Published journal article','10.1016/j.mechatronics.2007.04.004',''),
('toth2010modeling','Toth-2010-Modeling-identification-LPV-systems.pdf',old/'toth2010modeling.pdf','Published book','10.1007/978-3-642-13812-6',''),
('venkatraman2017psd','Venkatraman-2017-Predictive-state-decoders.pdf',new/'NIPS-2017-predictive-state-decoders-encoding-the-future-into-recurrent-networks-Paper.pdf','Published NeurIPS proceedings paper','','Citation currently occurs only in a todo in the audit snapshot.'),
('willard2020integrating','Willard-2022-Scientific-knowledge-machine-learning.pdf',new/'Williard- Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems.pdf','Published journal article','10.1145/3514228','Citation key retains 2020; published PDF gives November 2022, article 66.'),
('workman2021diewafer','Workman-2021-Die-to-wafer-hybrid-bonding.pdf',new/'Workman-Die_to_Wafer_Hybrid_Bonding_and_Fine_Pitch_Considerations.pdf','Published IEEE ECTC proceedings paper','10.1109/ECTC32696.2021.00326',''),
]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

# Preflight every source and target before copying anything.
assert len(entries)==23
assert len({e[0] for e in entries})==23
assert len({e[1].lower() for e in entries})==23
assert not destination.exists(),'Destination already exists; inspect before changing it.'
source_hashes={}
for key,name,source,version,doi,note in entries:
    assert source.is_file(),source
    assert '/' not in name and '\\' not in name,name
    source_hashes[source]=sha(source)
    with source.open('rb') as f:assert f.read(5)==b'%PDF-',source

destination.mkdir()
for key,name,source,version,doi,note in entries:
    target=destination/name
    shutil.copy2(source,target)
    assert sha(target)==source_hashes[source],target

with (destination/'index.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['Citation key','Filename','Version','DOI','Original file (repository-relative)','Bibliography review','Notes','SHA-256'])
    for key,name,source,version,doi,note in entries:
        writer.writerow([key,name,version,doi,source.relative_to(root).as_posix(),'Joint review against refs.bib pending',note,source_hashes[source]])

assert len(list(destination.glob('*.pdf')))==23
assert all(sha(source)==value for source,value in source_hashes.items())
with (destination/'index.csv').open(encoding='utf-8-sig',newline='') as f:assert len(list(csv.DictReader(f)))==23
print('Created Thesis-writeup/References/: 23 PDFs and index.csv.')
print('All copies match source SHA-256 hashes; all original files remain unchanged.')
print('Four preprints have explicit arXiv-v1 filename labels. Bibliography review remains pending.')
