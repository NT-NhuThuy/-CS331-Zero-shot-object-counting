// Runs the shipped review script with a minimal DOM. Checks export logic, not rendering.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const script=fs.readFileSync(process.argv[2],'utf8');
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.value='';this.checked=false;this.width=420;this.height=280;}
 append(...args){this.children.push(...args);}
 getContext(){return {drawImage(){},beginPath(){},arc(){},fill(){},stroke(){}};}
 getBoundingClientRect(){return {left:0,top:0,width:120,height:80};}
 click(){}
}
class MockImage{constructor(){this.naturalWidth=420;this.naturalHeight=280;}set src(v){this.onload?.();}}
function setup(payload){
 const elements=Object.fromEntries(['data','instructions','reviewer','export','summary','cards'].map(id=>[id,new Element('div')]));
 elements.data.textContent=JSON.stringify(payload);
 const blobs=[],alerts=[];
 vm.runInNewContext(script,{document:{getElementById:id=>elements[id],createElement:tag=>new Element(tag)},
  Image:MockImage,Blob,URL:{createObjectURL:b=>{blobs.push(b);return 'blob:test';},revokeObjectURL(){}},
  alert:m=>alerts.push(m),setTimeout:fn=>fn(),Date,Math,JSON,Object,Number,String,Infinity,console});
 return {elements,blobs,alerts};
}
function find(e,tag){return e.children.flatMap(c=>typeof c==='object'?[...(c.tag===tag?[c]:[]),...find(c,tag)]:[]);}
(async()=>{
 const regions=setup({kind:'regions',rows:[{id:'r1',image:'a.jpg',source_class:'apples',labels:[
   {query:'apples',label:1,evidence:'FSC147 exemplar'},{query:'pears',label:null,evidence:''}]}],selected:[{index:0,preview:'data:image/jpeg;base64,AA=='}]});
 const selects=find(regions.elements.cards,'select');selects[1].value='0';selects[1].onchange();
 regions.elements.export.onclick();assert.equal(regions.blobs.length,0);assert.equal(regions.alerts.length,1);
 regions.elements.reviewer.value='reviewer A';regions.elements.export.onclick();
 const r=JSON.parse((await regions.blobs[0].text()).trim());assert.equal(r.labels[1].label,0);assert.match(r.labels[1].evidence,/reviewer A/);
 assert.equal(r.labels[0].evidence,'FSC147 exemplar');assert.ok(!JSON.stringify(r).includes('_edited'));
 const queries=setup({kind:'queries',rows:[{id:'q1',image:'a.jpg',query:'pears',width:120,height:80,status:'unknown',gt_points:null,evidence:null}],
 selected:[{index:0,preview:'data:image/jpeg;base64,AA=='}]});
 const select=find(queries.elements.cards,'select')[0],canvas=find(queries.elements.cards,'canvas')[0];
 select.value='present';select.onchange();canvas.onclick({clientX:60,clientY:40});queries.elements.reviewer.value='reviewer B';
 queries.elements.export.onclick();assert.equal(queries.blobs.length,0);
 const checkbox=find(queries.elements.cards,'input')[0];checkbox.checked=true;checkbox.onchange();
 queries.elements.export.onclick();const q=JSON.parse((await queries.blobs[0].text()).trim());
 assert.deepEqual(q.points,[[60,40]]);assert.equal(q.status,'present');assert.match(q.evidence,/reviewer B/);
 select.value='absent';select.onchange();queries.elements.export.onclick();
 const absent=JSON.parse((await queries.blobs[1].text()).trim());assert.equal(absent.status,'absent');assert.deepEqual(absent.points,[]);
 console.log('Review JavaScript export tests: PASS (region evidence, reviewer guard, complete point guard, absent export)');
})().catch(e=>{console.error(e);process.exitCode=1;});
