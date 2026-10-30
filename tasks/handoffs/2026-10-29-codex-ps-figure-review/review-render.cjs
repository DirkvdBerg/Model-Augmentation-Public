const fs=require('fs'),path=require('path'),vm=require('vm');
const root='tasks/handoffs/2026-10-29-codex-ps-figure-review';const out=path.join(root,'review-evidence');fs.mkdirSync(out,{recursive:true});
for(const [folder,prefix] of [['04_15-approaches_2026-10-29/canvas-source','04'],['03b_canvas-PS-stages-figures_2026-10-28/project','03b'],['03_ps-slide-design_2026-10-28/canvas-PS-slide-concepts/project','03'],['02_ps-visual-design_2026-10-27-28/canvas-PS-slide-design/project','02']]){
 const files=fs.readdirSync(path.join(root,'attempts',folder)).filter(x=>x.endsWith('.dc.html'));let articles=[];
 for(const file of files){let s=fs.readFileSync(path.join(root,'attempts',folder,file),'utf8');const code=s.match(/<script type="text\/x-dc"[^>]*>([\s\S]*?)<\/script>/);let vals={};if(code) vals=vm.runInNewContext('class DCLogic {}\n'+code[1]+'\nnew Component().renderVals()',{}, {timeout:1000});
 s=s.replace(/\{\{(\w+)\}\}/g,(_,x)=>{if(!(x in vals))throw Error('Missing '+x);return vals[x]});
 s=s.replace(/<script\b[\s\S]*?<\/script>/g,'').replace(/<link\b[^>]*>/g,'').replace(/<\/?helmet>/g,'').replace(/<\/?x-dc>/g,'');
 fs.writeFileSync(path.join(out,prefix+'-'+file.replace('.dc','')),s);
 }
}
console.log(out);
