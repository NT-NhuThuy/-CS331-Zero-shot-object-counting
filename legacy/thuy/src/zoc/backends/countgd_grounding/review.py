"""Self-contained local HTML review; no server, network calls, or auto-labeling."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw
from .io import read_jsonl
from .protocol import validate_row, checked_region_labels


def render_review(ds, source, output, kind='regions', offset=0, limit=100):
    if offset < 0 or limit < 1 or kind not in ('regions', 'queries'):
        raise ValueError('Invalid review options')
    rows = list(read_jsonl(source))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate review IDs')
    selected = []
    for index in range(offset, min(offset + limit, len(rows))):
        row = rows[index]
        if kind == 'regions':
            checked_region_labels(ds, row, row['split'])
        else:
            validate_row(ds, row)
        with Image.open(ds.image_path(row['image'])) as source_image:
            image = source_image.convert('RGB')
        if kind == 'regions':
            x1, y1, x2, y2 = row['box_xyxy']
            w, h = image.size
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            halfw, halfh = (x2 - x1) * 0.9, (y2 - y1) * 0.9
            crop = (max(0, int(cx-halfw)), max(0, int(cy-halfh)),
                    min(w, int(cx+halfw+1)), min(h, int(cy+halfh+1)))
            image = image.crop(crop)
            oldw, oldh = image.size
            image.thumbnail((420, 280))
            sx, sy = image.width / oldw, image.height / oldh
            ImageDraw.Draw(image).rectangle(((x1-crop[0])*sx, (y1-crop[1])*sy,
                                            (x2-crop[0])*sx, (y2-crop[1])*sy), outline='red', width=3)
        else:
            image.thumbnail((1000, 700))
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        selected.append({'index': index, 'preview': 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()})
    payload = json.dumps({'kind': kind, 'rows': rows, 'selected': selected}, ensure_ascii=False).replace('<', '\\u003c')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(HTML.replace('__PAYLOAD__', payload), encoding='utf-8')
    return {'html': str(output), 'displayed': len(selected), 'total_records': len(rows), 'offset': offset}


HTML = r'''<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Grounding label review</title><style>
body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;background:#f5f7fa;color:#172333}
header{position:sticky;top:0;background:#fff;padding:16px;border-bottom:2px solid #ddd;z-index:2}
article{background:#fff;padding:18px;margin:16px 0;border:1px solid #ccd4de;border-radius:8px}
img,canvas{max-width:100%;height:auto}canvas{cursor:crosshair;border:1px solid #ccc}
label{display:block;margin:10px 0}select,input,button{font:inherit;padding:7px}button{cursor:pointer}
.note{color:#45556d}h2{font-size:18px}.pair{display:flex;gap:20px;align-items:center;margin:8px 0}
</style><header><b>Kiểm duyệt nhãn grounding</b><p class="note" id="instructions"></p>
<label>Người kiểm duyệt: <input id="reviewer" placeholder="Tên / mã reviewer"></label>
<button id="export">Xuất JSONL đã kiểm duyệt</button><span id="summary"></span>
<p class="note">Thay đổi chỉ nằm trong trang này cho đến khi xuất file. Xuất trước khi đóng/reload.
Chỉ xác nhận 0 khi chắc chắn prompt không phù hợp trong vùng; với query cấp ảnh phải kiểm tra toàn ảnh.</p></header>
<main id="cards"></main><script id="data" type="application/json">__PAYLOAD__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent), rows=data.rows;
const el=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;};
document.getElementById('instructions').textContent=data.kind==='regions'
 ? 'Khung đỏ là vùng được gán nhãn: 1 = phù hợp prompt; 0 = không phù hợp; unknown = chưa xác nhận. Không suy diễn negative từ nhãn lớp chính.'
 : 'Chọn present/absent/unknown. Với present: click tất cả target points rồi xác nhận annotation đầy đủ. GT lớp chính được khóa.';
for(const item of data.selected){
 const r=rows[item.index], card=el('article');card.append(el('h2',`${item.index+1}. ${r.image} | ${r.id}`));
 if(data.kind==='regions'){
  card.append(el('p',`Nguồn: ${r.source_class}. Review phải xem vùng và thuộc tính thực tế.`));
  const img=el('img');img.src=item.preview;card.append(img);
  for(const pair of r.labels){
   const div=el('div');div.className='pair';div.append(el('span',pair.query));const select=el('select');
   for(const [value,label] of [['null','unknown'],['1','1 — phù hợp'],['0','0 — không phù hợp']]){
    const opt=el('option',label);opt.value=value;select.append(opt);
   }
   select.value=pair.label===null?'null':String(pair.label);
   select.onchange=()=>{pair.label=select.value==='null'?null:Number(select.value);pair._edited=true;};
   div.append(select);card.append(div);
  }
 }else{
  const locked=r.evidence==='FSC147 primary class annotation';
  card.append(el('p',`${r.query}${locked?' — GT primary, chỉ xem':''}`));
  const select=el('select');for(const s of ['unknown','present','absent']){const o=el('option',s);o.value=s;select.append(o);}
  select.value=r.status;select.disabled=locked;card.append(select);
  const canvas=el('canvas'), image=new Image();card.append(canvas);let ready=false;
  const points=()=>r.gt_points||[];
  const draw=()=>{if(!ready)return;const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0,canvas.width,canvas.height);
   ctx.fillStyle='#00ffff';ctx.strokeStyle='#000';for(const [x,y] of points()){
    ctx.beginPath();ctx.arc(x/r.width*canvas.width,y/r.height*canvas.height,4,0,2*Math.PI);ctx.fill();ctx.stroke();}};
  image.onload=()=>{canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;ready=true;draw();};image.src=item.preview;
  const fullLabel=el('label'), complete=el('input');complete.type='checkbox';complete.checked=locked||r.status==='present';complete.disabled=locked;
  r._complete=complete.checked;fullLabel.append(complete,el('span',' Tôi xác nhận đã annotate TẤT CẢ target instances của query này.'));card.append(fullLabel);
  complete.onchange=()=>{r._complete=complete.checked;r._edited=true;};
  const eraseLabel=el('label'), erase=el('input');erase.type='checkbox';eraseLabel.append(erase,el('span',' Chế độ xóa điểm gần vị trí click'));card.append(eraseLabel);
  select.onchange=()=>{r.status=select.value;r.gt_points=r.status==='present'?[]:r.status==='absent'?[]:null;
    r._complete=false;complete.checked=false;r._edited=true;draw();};
  canvas.onclick=event=>{if(locked||!ready)return;if(r.status!=='present'){alert('Chọn present trước khi annotate.');return;}
   const b=canvas.getBoundingClientRect(),x=(event.clientX-b.left)/b.width*r.width,y=(event.clientY-b.top)/b.height*r.height;
   if(erase.checked){let index=-1,best=Infinity;points().forEach((p,i)=>{const d=Math.hypot(p[0]-x,p[1]-y);if(d<best){best=d;index=i;}});
    if(index>=0&&best<Math.hypot(r.width,r.height)*0.03)r.gt_points.splice(index,1);
   }else{if(!r.gt_points)r.gt_points=[];r.gt_points.push([x,y]);}
   r._edited=true;r._complete=false;complete.checked=false;draw();};
 }
 document.getElementById('cards').append(card);
}
document.getElementById('export').onclick=()=>{
 const reviewer=document.getElementById('reviewer').value.trim(), evidence=`manual review: ${reviewer}; ${new Date().toISOString()}`;
 let output=[];
 if(data.kind==='regions'){
  const needs=rows.some(r=>r.labels.some(p=>p._edited&&p.label!==null));
  if(needs&&!reviewer){alert('Nhập reviewer để ghi provenance cho nhãn mới.');return;}
  output=rows.map(r=>({...r,labels:r.labels.map(p=>({query:p.query,label:p.label,
    evidence:p.label===null?'':p._edited?evidence:p.evidence}))}));
 }else{
  for(const r of rows){
   if(r.evidence==='FSC147 primary class annotation'||r.status==='unknown')continue;
   if(r._edited&&!reviewer){alert('Nhập reviewer.');return;}
   if(r.status==='present'&&(!r.gt_points||!r.gt_points.length||(r._edited&&!r._complete))){alert(`Cần annotate đầy đủ và xác nhận: ${r.image}/${r.query}`);return;}
   output.push({image:r.image,query:r.query,status:r.status,points:r.status==='present'?r.gt_points:[],evidence:r._edited?evidence:r.evidence});
  }
 }
 const clean=output.map(r=>Object.fromEntries(Object.entries(r).filter(([k])=>!k.startsWith('_'))));
 const blob=new Blob([clean.map(r=>JSON.stringify(r)).join('\n')+'\n'],{type:'application/x-ndjson'}),url=URL.createObjectURL(blob);
 const a=el('a');a.href=url;a.download=data.kind==='regions'?'regions.reviewed.jsonl':'query_overrides.jsonl';a.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);document.getElementById('summary').textContent=` Đã xuất ${clean.length} records.`;
};
</script></html>'''
