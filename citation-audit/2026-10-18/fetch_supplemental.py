from pathlib import Path
import urllib.request,json,subprocess
O=Path(__file__).resolve().parent
urls={
'ovchinnikov2021computing-published.pdf':'https://par.nsf.gov/servlets/purl/10297271',
'hefny2015supervised-published.pdf':'https://proceedings.neurips.cc/paper_files/paper/2015/file/9a3d458322d70046f63dfd8b0153ece4-Paper.pdf',
'downey2017psrnn-published.pdf':'https://proceedings.neurips.cc/paper/2017/file/2bb0502c80b7432eee4c5847a5fd077b-Paper.pdf',
'hefny2015supervised-record.html':'https://proceedings.neurips.cc/paper/2015/hash/9a3d458322d70046f63dfd8b0153ece4-Abstract.html',
'downey2017psrnn-record.html':'https://proceedings.neurips.cc/paper/2017/hash/2bb0502c80b7432eee4c5847a5fd077b-Abstract.html',
'kessels2025ai-record.html':'https://research.tue.nl/en/publications/ai-based-model-updating-for-nonlinear-dynamical-systems/',
'gyorok2025l4dc-record.html':'https://proceedings.mlr.press/v283/gyorok25a.html',
'drenth2025lpvlfr-record.html':'https://arxiv.org/abs/2605.12203',
'hoekstra2026lfr-record.html':'https://arxiv.org/abs/2602.17297',
'hoekstra2026encoder-record.html':'https://arxiv.org/abs/2602.13108',
'raissi2017physicsinformedpart1-record.html':'https://arxiv.org/abs/1711.10561',
}
log=[]
for n,u in urls.items():
 try:
  p=O/'external-sources'/n
  urllib.request.urlretrieve(u,p)
  if n.endswith('.pdf'):
   assert p.read_bytes().startswith(b'%PDF'), 'Not a PDF'
   subprocess.run(['pdftotext','-layout',str(p),str(O/'source-text'/(p.stem+'.txt'))],capture_output=True,check=True)
  log.append({'artifact':n,'url':u,'status':'saved'})
 except Exception as e:log.append({'artifact':n,'url':u,'error':str(e)})
(O/'supplemental-retrieval-log.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
print(json.dumps(log,indent=2))
