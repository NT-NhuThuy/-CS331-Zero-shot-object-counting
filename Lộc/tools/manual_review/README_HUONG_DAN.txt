FSC-147 | CHUONG TRINH KIEM TRA 160 CAP HARD-NEGATIVE BANG PYTHON
===================================================================

1. Dung tren may tinh Windows/macOS/Linux co giao dien do hoa.
   KHONG chay bang Run All tren Kaggle hoac Colab: Tkinter can cua so desktop.

2. Giai nen TOAN BO file ZIP ra mot thu muc tren may.
   KHONG chay file .py truc tiep ben trong file ZIP.

3. May can Python 3.10+ va Pillow:
      python -m pip install Pillow
   Tren Windows, co the bam dup CHAY_TREN_WINDOWS.bat (may phai co Python).
   Hoac mo Terminal / CMD tai thu muc da giai nen va go:
      python review_freshneg.py

4. Chuong trinh co san anh xem truoc cat tu 8 contact sheets de ban
   bat dau danh gia ngay, KHONG can tai FSC-147 moi bat dau.
   DE XAC MINH CHAC CHAN, nen tai bo anh FSC-147 goc (hihihi5) ve may,
   bam "Chon thu muc anh goc" va chon thu muc cha chua anh .jpg.
   Chuong trinh tim .jpg trong tat ca thu muc con, uu tien anh goc.

5. Phim nhanh:
      1 = ABSENT (doi tuong DUOC HOI khong xuat hien)
      2 = PRESENT (doi tuong DUOC HOI co xuat hien)
      3 = UNCERTAIN (khong chac / anh nho / can xem anh goc)
      Mui ten trai = anh truoc
      Mui ten phai / Space = anh tiep
   Sau moi lan bam 1/2/3, ket qua TU LUU ngay vao file:
      V75_FreshNeg_Reviewed_160_MANUAL.csv
   va tu dong chuyen anh tiep theo. Co the dong roi mo lai de tiep tuc.

6. Quan trong: nhan theo DOi TUONG DUOC HOI (target), KHONG nhan theo
   source_class cua anh. Thí du anh 'peaches', target 'apples':
   chi bam 1 neu chac chan khong co bat cu qua tao nao.

7. Neu 2 nguoi cung danh gia doc lap:
   - Nguoi thu nhat giu "Nguoi danh gia: 1" va lam du 160 anh.
   - Nguoi thu hai chon "Nguoi danh gia: 2" va lam du 160 anh.
   - Khi hai nguoi cung chon ABSENT/PRESENT, cot adjudicated_presence
     tu dong nhan nhan do.
   - Neu bat dong hoac mot nguoi chon UNCERTAIN, adjudicated_presence
     de trong, cot adjudication_status la NEEDS_ADJUDICATION.
   - KHONG TINH tat ca 160 anh la hard-negative khi chua xac minh.

8. Neu co anh co the chua bottles (FHN065, FHN132, FHN153), kiem tra
   dac biet can than vi cac binh/chai sua co the la bottles (PRESENT).

9. Khi hoan tat, gui file V75_FreshNeg_Reviewed_160_MANUAL.csv vao ChatGPT
   de kiem tra tinh nhat quan va chuan bi chay V7.5 safety benchmark.

10. De thu xem goi tep day du:
      python review_freshneg.py --self-test

GHI CHU: Xem truoc trong goi duoc tao tu contact sheet do nguoi dung cung cap,
khong phai anh FSC-147 goc o do phan giai cao. Ban can xem anh goc neu
khong dam bao doi tuong co ton tai hay khong.
