#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Desktop tool to manually label FSC-147 fresh-negative image/query pairs.

Run: python review_freshneg.py
Requires: Pillow, tkinter (normally included with Windows Python).

Ground-truth safety: single reviewer decisions NEVER auto-adjudicate.
After two reviewers agree on ABSENT/PRESENT, the adjudicated_presence
column is filled automatically. Disagreements remain blank for adjudication.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps

APP_DIR = Path(__file__).resolve().parent
ORIGINAL = APP_DIR / 'V75_FreshNeg_Reviewed_160.csv'
RESULTS = APP_DIR / 'V75_FreshNeg_Reviewed_160_MANUAL.csv'
SETTINGS = APP_DIR / 'review_settings.json'

EN_VI = {
    'alarm clocks':'đồng hồ báo thức', 'apples':'quả táo', 'apricots':'quả mơ',
    'armchairs':'ghế bành', 'avocados':'quả bơ', 'bagels':'bánh mì vòng bagel',
    'bats':'dơi', 'batteries':'pin', 'beads':'hạt cườm', 'beans':'hạt đậu',
    'beetles':'bọ cánh cứng', 'benches':'ghế băng', 'bicycles':'xe đạp',
    'blueberries':'quả việt quất', 'bottles':'chai / bình dạng chai',
    'buckets':'xô / thùng có quai', 'butterflies':'bướm', 'buttons':'cúc áo / nút áo',
    'camels':'lạc đà', 'candles':'nến', 'cans':'lon / hộp kim loại',
    'carrots':'cà rốt', 'clams':'nghêu / ngao / sò hai mảnh', 'coins':'đồng xu',
    'comic books':'truyện tranh', 'confetti':'giấy kim tuyến / giấy vụn trang trí',
    'cookies':'bánh quy', 'cows':'bò', 'cranes':'chim sếu', 'crayons':'bút sáp màu',
    'cucumbers':'dưa leo / dưa chuột', 'dragonflies':'chuồn chuồn',
    'dresses':'váy liền / đầm', 'fish fingers':'thanh cá tẩm bột',
    'flies':'ruồi', 'fried calamari':'mực tẩm bột chiên',
    'garlic cloves':'tép tỏi', 'guitars':'đàn guitar', 'helmets':'mũ bảo hiểm',
    'herons':'chim diệc / cò', 'horses':'ngựa', 'juice boxes':'hộp nước trái cây',
    'limes':'quả chanh xanh', 'lipsticks':'thỏi son', 'llamas':'lạc đà không bướu (llama)',
    'magazines':'tạp chí', 'marbles':'viên bi', 'mints':'kẹo bạc hà',
    'motorcycles':'xe máy / mô tô', 'olives':'quả ô liu', 'pants':'quần dài',
    'paper flowers':'hoa giấy trang trí', 'paper towel rolls':'cuộn giấy lau bếp',
    'pigeons':'chim bồ câu', 'pins':'ghim / đinh ghim',
    'plant watering cans':'bình tưới cây', 'plastic cups':'cốc nhựa',
    'potatoes':'khoai tây', 'radishes':'củ cải', 'rice crackers':'bánh gạo',
    'rocks':'đá / tảng đá', 'soda cans':'lon nước ngọt', 'spiders':'nhện',
    'stools':'ghế đẩu', 'storks':'chim cò / chim hạc (stork)',
    'suitcases':'va li', 'surfboards':'ván lướt sóng',
    'telephones':'điện thoại', 'terns':'chim nhạn biển (tern)',
    'tomatoes':'cà chua', 'towels':'khăn', 'traffic lights':'đèn giao thông',
    'umbrellas':'ô / dù', 'vases':'bình hoa', 'zebras':'ngựa vằn',
}

BASE_COLS = [
 'candidate_id','group_id','image_name','source_class','target',
 'candidate_type','eval_partition','reviewer_1_decision',
 'reviewer_2_decision','adjudicated_presence','review_notes'
]
EXTRA_COLS = ['reviewer_1_notes','reviewer_2_notes','reviewer_1_utc','reviewer_2_utc','adjudication_status']
VALID = {'ABSENT','PRESENT','UNCERTAIN'}


def load_rows(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        rows=list(reader)
        fields=reader.fieldnames or []
    return rows,fields


def save_csv_atomic(path, rows, cols):
    """Write a complete valid CSV to temp, then atomically replace the output."""
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='review_tmp_',suffix='.csv',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def reconcile(row):
    """Never promote a one-reviewer label to ground truth."""
    a=row.get('reviewer_1_decision','').strip().upper()
    b=row.get('reviewer_2_decision','').strip().upper()
    if a in ('ABSENT','PRESENT') and a==b:
        row['adjudicated_presence']=a
        row['adjudication_status']='AGREED_TWO_REVIEWERS'
    elif a and b:
        row['adjudicated_presence']=''
        row['adjudication_status']='NEEDS_ADJUDICATION'
    else:
        row['adjudicated_presence']=''
        row['adjudication_status']='WAITING_SECOND_REVIEWER'


def validate_rows(rows):
    ids=[r.get('candidate_id','') for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError('CSV có candidate_id bị trùng')
    for r in rows:
        for field in ('reviewer_1_decision','reviewer_2_decision'):
            val=r.get(field,'').strip().upper()
            if val and val not in VALID:
                raise ValueError(f"Nhãn không hợp lệ: {r.get('candidate_id')} {field}={val}")


def read_settings():
    try: return json.loads(SETTINGS.read_text(encoding='utf-8'))
    except (OSError,ValueError): return {}


def write_settings(settings):
    try: SETTINGS.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    except OSError: pass


class ReviewApp:
    def __init__(self, root):
        self.root=root
        self.root.title('FSC-147 · Fresh Hard-Negative — Xác minh thủ công')
        self.root.geometry('1360x880')
        self.root.minsize(880,640)
        self.bg='#111827';self.panel='#1f2937'; self.text='#f9fafb';self.muted='#9ca3af'
        root.configure(bg=self.bg)
        self.settings=read_settings()
        self.original_dir=Path(self.settings.get('image_dir','')) if self.settings.get('image_dir') else None
        self.image_index={}
        if self.original_dir and self.original_dir.exists(): self.index_images()
        src = RESULTS if RESULTS.exists() else ORIGINAL
        self.rows,fields=load_rows(src)
        validate_rows(self.rows)
        if not self.rows:raise ValueError('Không có hàng dữ liệu trong CSV')
        self.columns=list(dict.fromkeys(fields+BASE_COLS+EXTRA_COLS))
        self.reviewer_var=tk.StringVar(value=self.settings.get('reviewer','1'))
        self.index=0
        self.note_var=tk.StringVar()
        self.picture=None
        self.last_img=None
        self.last_image_path=None
        self.source_note=''
        self.review_field='reviewer_1_decision'
        self.make_ui()
        self.index=self.find_first_unreviewed(0)
        self.refresh()
        self.root.protocol('WM_DELETE_WINDOW',self.on_close)
        root.bind('<KeyPress>',self.on_key)
        root.bind('<Configure>',self.on_resize)

    def make_ui(self):
        top=tk.Frame(self.root,bg=self.bg)
        top.pack(fill='x',padx=22,pady=(17,8))
        tk.Label(top,text='XÁC MINH HARD-NEGATIVE · FSC-147',bg=self.bg,fg=self.text,
                 font=('Segoe UI',20,'bold')).pack(side='left')
        tk.Label(top,text='Người đánh giá:',bg=self.bg,fg=self.muted,font=('Segoe UI',11)).pack(side='left',padx=(26,7))
        switch=ttk.Combobox(top,textvariable=self.reviewer_var,values=('1','2'),width=4,state='readonly')
        switch.pack(side='left')
        switch.bind('<<ComboboxSelected>>',self.switch_reviewer)
        tk.Button(top,text='Chọn thư mục ảnh gốc',command=self.choose_originals,
                  font=('Segoe UI',10),padx=12).pack(side='right')
        tk.Button(top,text='Mở CSV kết quả',command=self.show_csv,
                  font=('Segoe UI',10),padx=12).pack(side='right',padx=8)
        stats=tk.Frame(self.root,bg=self.bg)
        stats.pack(fill='x',padx=22)
        self.stats_var=tk.StringVar()
        tk.Label(stats,textvariable=self.stats_var,bg=self.bg,fg='#93c5fd',font=('Segoe UI',11,'bold')).pack(anchor='w')
        self.progress=ttk.Progressbar(stats,mode='determinate',maximum=len(self.rows))
        self.progress.pack(fill='x',pady=(8,4))

        main=tk.PanedWindow(self.root,orient='horizontal',bg=self.bg,sashwidth=7,bd=0)
        main.pack(fill='both',expand=True,padx=18,pady=10)
        left=tk.Frame(main,bg='#0b1220',highlightbackground='#374151',highlightthickness=1)
        right=tk.Frame(main,bg=self.panel,width=345,highlightbackground='#374151',highlightthickness=1)
        main.add(left,minsize=440,stretch='always')
        main.add(right,minsize=310,stretch='never')
        self.canvas=tk.Canvas(left,bg='#0b1220',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True,padx=5,pady=5)

        right_inner=tk.Frame(right,bg=self.panel)
        right_inner.pack(fill='both',expand=True,padx=18,pady=18)
        self.title_var=tk.StringVar()
        tk.Label(right_inner,textvariable=self.title_var,bg=self.panel,fg=self.muted,
                 anchor='w',font=('Segoe UI',10)).pack(fill='x')
        tk.Label(right_inner,text='ĐỐI TƯỢNG CẦN KIỂM TRA',bg=self.panel,fg='#93c5fd',
                 anchor='w',font=('Segoe UI',10,'bold')).pack(fill='x',pady=(18,6))
        self.en_var=tk.StringVar();self.vi_var=tk.StringVar()
        tk.Label(right_inner,textvariable=self.en_var,bg=self.panel,fg=self.text,
                 wraplength=300,justify='left',anchor='w',font=('Segoe UI',23,'bold')).pack(fill='x')
        tk.Label(right_inner,textvariable=self.vi_var,bg=self.panel,fg='#fbbf24',
                 wraplength=300,justify='left',anchor='w',font=('Segoe UI',17,'bold')).pack(fill='x',pady=(5,16))
        self.meta_var=tk.StringVar()
        tk.Label(right_inner,textvariable=self.meta_var,bg=self.panel,fg='#d1d5db',
                 anchor='w',justify='left',wraplength=300,font=('Segoe UI',10)).pack(fill='x',pady=(0,16))
        self.image_type=tk.StringVar()
        tk.Label(right_inner,textvariable=self.image_type,bg=self.panel,fg=self.muted,
                 anchor='w',justify='left',wraplength=300,font=('Segoe UI',9)).pack(fill='x',pady=(0,14))
        self.btn_absent=tk.Button(right_inner,text='1   ABSENT  ·  KHÔNG CÓ',bg='#047857',fg='white',
               activebackground='#065f46',activeforeground='white',relief='flat',cursor='hand2',
               font=('Segoe UI',13,'bold'),pady=13,command=lambda:self.vote('ABSENT'))
        self.btn_absent.pack(fill='x',pady=4)
        self.btn_present=tk.Button(right_inner,text='2   PRESENT  ·  CÓ',bg='#b45309',fg='white',
               activebackground='#92400e',activeforeground='white',relief='flat',cursor='hand2',
               font=('Segoe UI',13,'bold'),pady=13,command=lambda:self.vote('PRESENT'))
        self.btn_present.pack(fill='x',pady=4)
        tk.Button(right_inner,text='3   UNCERTAIN  ·  KHÔNG CHẮC',bg='#374151',fg='white',
               activebackground='#4b5563',activeforeground='white',relief='flat',
               font=('Segoe UI',11,'bold'),pady=11,command=lambda:self.vote('UNCERTAIN')).pack(fill='x',pady=4)
        tk.Label(right_inner,text='Ghi chú (tùy chọn):',bg=self.panel,fg=self.muted,
                 font=('Segoe UI',10)).pack(anchor='w',pady=(14,3))
        self.note_entry=tk.Entry(right_inner,textvariable=self.note_var,font=('Segoe UI',11))
        self.note_entry.pack(fill='x',ipady=7)
        tk.Button(right_inner,text='Lưu ghi chú',command=self.save_note_only,
                  font=('Segoe UI',10)).pack(anchor='e',pady=(5,10))
        self.saved_var=tk.StringVar()
        tk.Label(right_inner,textvariable=self.saved_var,bg=self.panel,fg='#86efac',
                 wraplength=305,anchor='w',justify='left',font=('Segoe UI',10)).pack(fill='x')
        bottom=tk.Frame(self.root,bg=self.bg)
        bottom.pack(fill='x',padx=22,pady=(4,18))
        tk.Button(bottom,text='←  Ảnh trước  (←)',command=self.previous,padx=14,pady=8).pack(side='left')
        tk.Button(bottom,text='Ảnh tiếp  (→ hoặc Space)',command=self.next,padx=14,pady=8).pack(side='left',padx=8)
        tk.Button(bottom,text='Đến ảnh chưa đánh giá',command=self.jump_unreviewed,padx=14,pady=8).pack(side='left')
        tk.Label(bottom,text='TỰ LƯU NGAY SAU MỖI LẦN BẤM 1 / 2 / 3',bg=self.bg,fg='#86efac',
                 font=('Segoe UI',10,'bold')).pack(side='right')

    def index_images(self):
        self.image_index={}
        if not self.original_dir or not self.original_dir.exists():return
        for p in self.original_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'):
                self.image_index.setdefault(p.name.lower(),p)

    def choose_originals(self):
        directory=filedialog.askdirectory(title='Chọn thư mục chứa ảnh FSC-147 (.jpg)')
        if not directory:return
        self.original_dir=Path(directory)
        self.index_images()
        self.settings['image_dir']=directory
        write_settings(self.settings)
        self.refresh()
        messagebox.showinfo('Đã chọn ảnh',f'Tìm thấy {len(self.image_index)} ảnh.\nỨng dụng ưu tiên ảnh gốc; nếu thiếu thì dùng preview.')

    def show_csv(self):
        self.save_all()
        messagebox.showinfo('CSV kết quả',f'Đã lưu tự động tại:\n{RESULTS}\n\nBạn có thể mở file bằng Excel hoặc gửi CSV này để kiểm tra.')

    def find_first_unreviewed(self,start):
        field=f'reviewer_{self.reviewer_var.get()}_decision'
        for j in range(start,len(self.rows)):
            if not self.rows[j].get(field,'').strip(): return j
        for j in range(0,start):
            if not self.rows[j].get(field,'').strip(): return j
        return max(0,min(start,len(self.rows)-1))

    def switch_reviewer(self,_event=None):
        self.review_field=f'reviewer_{self.reviewer_var.get()}_decision'
        self.settings['reviewer']=self.reviewer_var.get()
        write_settings(self.settings)
        self.index=self.find_first_unreviewed(0)
        self.refresh()

    def refresh(self):
        row=self.rows[self.index]
        self.review_field=f'reviewer_{self.reviewer_var.get()}_decision'
        voted=sum(bool(x.get(self.review_field,'').strip()) for x in self.rows)
        agreed=sum(x.get('adjudication_status')=='AGREED_TWO_REVIEWERS' for x in self.rows)
        conflicting=sum(x.get('adjudication_status')=='NEEDS_ADJUDICATION' for x in self.rows)
        self.progress['value']=voted
        self.stats_var.set(f'Người đánh giá {self.reviewer_var.get()} · Đã xác minh {voted}/{len(self.rows)}  ·  Hai người đồng thuận {agreed}  ·  Cần phân xử {conflicting}')
        self.title_var.set(f'{row.get("candidate_id", "")}  |  Ảnh {self.index+1}/{len(self.rows)}')
        en=row.get('target','')
        self.en_var.set(en.upper())
        self.vi_var.set(EN_VI.get(en.lower(),f'[Chưa có bản dịch: {en}]'))
        self.meta_var.set(f'Ảnh: {row.get("image_name", "")}\nLoại ứng viên: {row.get("candidate_type", "")}\nLớp gốc ảnh: {row.get("source_class", "")}')
        my_label=row.get(self.review_field,'')
        self.saved_var.set(('Đã lưu nhãn của bạn: '+my_label) if my_label else 'Chưa có nhãn ở ảnh này.')
        self.note_var.set(row.get(f'reviewer_{self.reviewer_var.get()}_notes',''))
        self.load_picture(row)

    def load_picture(self,row):
        p=self.image_index.get(row.get('image_name','').lower())
        if p and p.exists():
            self.source_note='ẢNH GỐC · có thể nhìn chi tiết khi phóng cửa sổ'
        else:
            p=APP_DIR/'previews'/f'{row.get("candidate_id","")}.jpg'
            self.source_note='ẢNH XEM TRƯỚC TỪ CONTACT SHEET · ảnh nhỏ; nếu khó hãy chọn thư mục ảnh gốc.'
        self.last_image_path=p
        self.image_type.set(self.source_note)
        try:
            with Image.open(p) as im:self.last_img=ImageOps.exif_transpose(im).convert('RGB')
        except (OSError,ValueError):self.last_img=None
        self.draw_picture()

    def draw_picture(self):
        self.canvas.delete('all')
        w=max(440,self.canvas.winfo_width()-20)
        h=max(320,self.canvas.winfo_height()-20)
        if self.last_img is None:
            self.canvas.create_text(w//2,h//2,text='Không tìm thấy ảnh.\nHãy chọn thư mục FSC-147.',
                    fill='#fca5a5',font=('Segoe UI',16),justify='center')
            return
        image=self.last_img.copy()
        image.thumbnail((w,h),Image.Resampling.LANCZOS)
        self.picture=ImageTk.PhotoImage(image)
        self.canvas.create_image(w//2+10,h//2+10,image=self.picture,anchor='center')

    def on_resize(self,event):
        if event.widget is self.canvas or event.widget is self.root:
            if hasattr(self,'_redraw_handle'):
                try:self.root.after_cancel(self._redraw_handle)
                except Exception:pass
            self._redraw_handle=self.root.after(130,self.draw_picture)

    def save_all(self):
        save_csv_atomic(RESULTS,self.rows,self.columns)

    def vote(self,label):
        from datetime import datetime,timezone
        row=self.rows[self.index]
        row[self.review_field]=label
        suffix=self.reviewer_var.get()
        row[f'reviewer_{suffix}_notes']=self.note_var.get().strip()
        row[f'reviewer_{suffix}_utc']=datetime.now(timezone.utc).isoformat(timespec='seconds')
        reconcile(row)
        try:self.save_all()
        except OSError as e:
            messagebox.showerror('Không lưu được CSV',str(e))
            return
        if self.index<len(self.rows)-1:self.index+=1
        else:
            remain=sum(not r.get(self.review_field,'').strip() for r in self.rows)
            if remain:
                self.index=self.find_first_unreviewed(0)
            else:
                self.refresh()
                messagebox.showinfo('Hoàn tất',f'Đã đánh giá {len(self.rows)} ảnh!\nKết quả lưu tại:\n{RESULTS}\n\nNếu cần người thứ hai, chuyển Người đánh giá sang 2.')
                return
        self.refresh()

    def save_note_only(self):
        row=self.rows[self.index]
        row[f'reviewer_{self.reviewer_var.get()}_notes']=self.note_var.get().strip()
        self.save_all()
        self.saved_var.set('Đã lưu ghi chú vào CSV.')

    def previous(self):
        self.index=max(0,self.index-1)
        self.refresh()

    def next(self):
        self.index=min(len(self.rows)-1,self.index+1)
        self.refresh()

    def jump_unreviewed(self):
        self.index=self.find_first_unreviewed(self.index+1)
        self.refresh()

    def on_key(self,event):
        if event.widget is self.note_entry: return
        if event.keysym in ('1','KP_1'):self.vote('ABSENT')
        elif event.keysym in ('2','KP_2'):self.vote('PRESENT')
        elif event.keysym in ('3','KP_3'):self.vote('UNCERTAIN')
        elif event.keysym=='Left':self.previous()
        elif event.keysym in ('Right','space'):self.next()

    def on_close(self):
        try:self.save_all()
        except OSError as e:
            if not messagebox.askyesno('Không lưu được',f'{e}\nVẫn đóng chương trình?'):return
        self.root.destroy()


def self_test():
    rows, cols = load_rows(ORIGINAL)
    assert len(rows)==160,len(rows)
    assert len(set(r['candidate_id'] for r in rows))==160
    keys=sorted({r['target'] for r in rows})
    missing=[k for k in keys if k not in EN_VI]
    assert not missing,missing
    absent=[r['candidate_id'] for r in rows if not (APP_DIR/'previews'/f"{r['candidate_id']}.jpg").exists()]
    assert not absent,absent
    row=dict(rows[0]);row['reviewer_1_decision']='ABSENT';reconcile(row)
    assert not row['adjudicated_presence']
    row['reviewer_2_decision']='ABSENT';reconcile(row)
    assert row['adjudicated_presence']=='ABSENT'
    row['reviewer_2_decision']='PRESENT';reconcile(row)
    assert not row['adjudicated_presence']
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'saved.csv';save_csv_atomic(path,rows,cols)
        test,_=load_rows(path)
        assert len(test)==160
    print('SELF TEST OK: 160 cặp; 75 bản dịch; 160 ảnh xem trước; ghi CSV và tính đồng thuận hợp lệ.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if not ORIGINAL.exists():
        raise SystemExit(f'Thiếu {ORIGINAL.name}. Hãy giải nén trọn bộ ZIP.')
    root=tk.Tk()
    try:ReviewApp(root)
    except Exception as e:
        messagebox.showerror('Không khởi động được',str(e));root.destroy();return
    root.mainloop()

if __name__=='__main__':main()
