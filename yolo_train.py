import os
import shutil
import random
import hashlib
from ultralytics import YOLO


def dosya_hash_hesapla(dosya_yolu):
    """Dosyanın içeriğine göre benzersiz bir parmak izi (hash) oluşturur."""
    hasher = hashlib.md5()
    with open(dosya_yolu, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def temiz_dataset_olustur(kaynak_dizin, hedef_dizin, train_orani=0.8):
    if os.path.exists(hedef_dizin):
        shutil.rmtree(hedef_dizin)
    os.makedirs(hedef_dizin)

    siniflar = ['bus', 'car', 'truck']

    for sinif in siniflar:
        src_path = os.path.join(kaynak_dizin, sinif)
        if not os.path.exists(src_path):
            continue

        benzersiz_resimler = []
        hash_listesi = set()

        for dosya in os.listdir(src_path):
            yol = os.path.join(src_path, dosya)
            if not os.path.isfile(yol):
                continue

            parmak_izi = dosya_hash_hesapla(yol)
            if parmak_izi not in hash_listesi:
                hash_listesi.add(parmak_izi)
                benzersiz_resimler.append(dosya)

        random.shuffle(benzersiz_resimler)
        sinir = int(len(benzersiz_resimler) * train_orani)

        train_yol = os.path.join(hedef_dizin, 'train', sinif)
        val_yol = os.path.join(hedef_dizin, 'val', sinif)
        os.makedirs(train_yol, exist_ok=True)
        os.makedirs(val_yol, exist_ok=True)

        for i, img in enumerate(benzersiz_resimler):
            kaynak = os.path.join(src_path, img)
            hedef = train_yol if i < sinir else val_yol
            shutil.copy(kaynak, os.path.join(hedef, img))

        print(f"{sinif.upper()} | Toplam: {len(benzersiz_resimler)}")

    print("\n✅ Dataset hazırlandı.")


if __name__ == "__main__":
    # 1️⃣ Dataset'i hazırla
    temiz_dataset_olustur(
        kaynak_dizin="ham_veriler",
        hedef_dizin="dataset"
    )

    # 2️⃣ YOLOv8 Classification eğitimi
    model = YOLO("yolov8n-cls.pt")
    model.train(
        data="dataset",
        epochs=20,
        imgsz=224
    )