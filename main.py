# -*- coding: utf-8 -*-
"""
Geliştirilmiş Araç Tespit ve Plaka Okuma Sistemi
PYCHARM / LOCAL VERSION - FINAL STABLE
"""

import cv2
import os
import sys
import torch
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import easyocr
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

# ================================
# YEREL DİZİN AYARLARI
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# ================================
# DEVICE
# ================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Kullanılan cihaz: {device}")

# ================================
# MODELLER
# ================================
reader = easyocr.Reader(['en', 'tr'], gpu=(device == "cuda"))
coco_model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))
license_plate_detector = YOLO(os.path.join(BASE_DIR, "models", "license_plate_detector.pt"))

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


# ================================
# VERİTABANI
# ================================
def db_hazirla():
    conn = sqlite3.connect("guvenlik_sistemi.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS izinli_plakalar (plaka TEXT PRIMARY KEY)""")
    c.execute("""CREATE TABLE IF NOT EXISTS gecis_loglari (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plaka TEXT, arac_tipi TEXT, vlm_yorum TEXT, durum TEXT, tarih TEXT, saat TEXT
    )""")
    # Örnek plaka listesi (Seninkiler burada olmalı)
    ornek_plakalar = [("07EC605",), ("66LC114",)]
    c.executemany("INSERT OR IGNORE INTO izinli_plakalar VALUES (?)", ornek_plakalar)
    conn.commit()
    conn.close()


def plaka_izinli_mi(plaka):
    if not plaka or plaka == "OKUNAMADI": return False
    conn = sqlite3.connect("guvenlik_sistemi.db")
    c = conn.cursor()
    c.execute("SELECT 1 FROM izinli_plakalar WHERE plaka=?", (plaka,))
    r = c.fetchone()
    conn.close(
    )
    return r is not None


def log_kaydet(plaka, tip, vlm, durum):
    conn = sqlite3.connect("guvenlik_sistemi.db")
    c = conn.cursor()
    now = datetime.now()
    c.execute("""INSERT INTO gecis_loglari (plaka, arac_tipi, vlm_yorum, durum, tarih, saat)
                 VALUES (?,?,?,?,?,?)""", (plaka, tip, vlm, durum, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")))
    conn.commit()
    conn.close()


# ================================
# PLAKA FORMAT & OCR
# ================================
def plaka_on_isleme(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1. CLAHE ile kontrast iyileştirme
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    # 2. Otsu Threshold ile ikili görüntü
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [clahe, thresh]


def turk_plaka_formatla(text):
    # Katı yapıyı bozmadan TR ve boşluk temizliği
    text = text.replace("TR", "").replace(" ", "").upper()

    il = ""
    for c in text:
        if c.isdigit():
            il += c
        else:
            break
    if not (1 <= len(il) <= 2): return None

    kalan = text[len(il):]
    harf = ""
    for c in kalan:
        if c.isalpha():
            harf += c
        else:
            break
    if not (1 <= len(harf) <= 3): return None

    num = kalan[len(harf):]
    if not num.isdigit() or not (1 <= len(num) <= 4): return None

    return f"{il} {harf} {num}"


def plaka_oku_coklu_deneme(plate_crop):
    tum_adaylar = []
    print("\n" + "=" * 50 + "\n🔍 PLAKA ANALİZİ BAŞLADI\n" + "=" * 50)

    for scale in [2.5, 3.0]:
        resized = cv2.resize(plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for i, img in enumerate(plaka_on_isleme(resized)):
            # paragraph=True ile parçalar (66 + LC + 114) birleştirilir
            results = reader.readtext(img, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ", paragraph=True)

            for res in results:
                # UNPACK HATASI ÇÖZÜMÜ: paragraph=True modunda 2 veya 3 değer gelebilir
                if len(res) == 3:
                    _, text, conf = res
                elif len(res) == 2:
                    _, text = res
                    conf = 0.90  # Paragraf modunda güven skoru dönmezse varsayılan
                else:
                    continue

                clean = text.replace(" ", "").upper()
                print(f"    [OCR] Ham/Birleşik Metin: '{clean}'")

                formatted = turk_plaka_formatla(clean)
                if formatted:
                    print(f"    [✓] FORMAT ONAYLI: {formatted}")
                    tum_adaylar.append((formatted, conf))
                else:
                    print(f"    [X] FORMAT HATASI: '{clean}' katı kurallara uymuyor.")

    if tum_adaylar:
        tum_adaylar.sort(key=lambda x: x[1], reverse=True)
        final = tum_adaylar[0][0]
        print(f"\n🏆 FİNAL KARAR: {final}\n" + "=" * 50)
        return final

    print("\n❌ SONUÇ: OKUNAMADI\n" + "=" * 50)
    return "OKUNAMADI"


# ================================
# VLM (RENK VE DETAY ODAKLI)
# ================================
def vlm_ile_arac_analizi(arac_crop):
    """VLM kullanarak aracın özelliklerini detaylı çıkar"""
    try:
        print(f"   🤖 VLM analizi başlatılıyor...", end=" ")

        rgb_img = cv2.cvtColor(arac_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        target_size = 224 if device == "cpu" else 384
        pil_img = pil_img.resize((target_size, target_size), Image.LANCZOS)

        inputs = processor(images=pil_img, return_tensors="pt").to(device)

        outputs = vlm_model.generate(
            **inputs,
            max_new_tokens=60 if device == "cpu" else 80,
            num_beams=3 if device == "cpu" else 5,
            repetition_penalty=1.3,
            length_penalty=1.2,
            early_stopping=True,
            do_sample=False
        )

        caption = processor.decode(outputs[0], skip_special_tokens=True)
        caption = caption.strip().upper()

        print(f"✓")
        print(f"      Ham VLM: {caption}")

        # Renk, tip, marka çıkarımı (önceki kod ile aynı)
        renkler = {
            'BEYAZ': ['WHITE', 'SNOW', 'PEARL'],
            'SİYAH': ['BLACK', 'DARK', 'EBONY'],
            'GRİ': ['GRAY', 'GREY', 'SILVER', 'CHARCOAL'],
            'KIRMIZI': ['RED', 'CRIMSON', 'MAROON'],
            'MAVİ': ['BLUE', 'NAVY', 'AZURE'],
            'YEŞİL': ['GREEN', 'OLIVE', 'EMERALD'],
            'SARI': ['YELLOW', 'GOLD', 'AMBER'],
            'KAHVERENGİ': ['BROWN', 'BEIGE', 'TAN'],
            'TURUNCU': ['ORANGE'],
            'MOR': ['PURPLE', 'VIOLET']
        }

        tespit_edilen_renk = None
        for ana_renk, anahtar_kelimeler in renkler.items():
            if any(kelime in caption for kelime in anahtar_kelimeler):
                tespit_edilen_renk = ana_renk
                break

        tipler = {
            'OTOBÜS': ['BUS', 'COACH', 'MINIBUS'],
            'KAMYON': ['TRUCK', 'LORRY', 'PICKUP', 'VAN'],
            'SEDAN': ['SEDAN', 'SALOON'],
            'SUV': ['SUV', 'CROSSOVER', 'JEEP'],
            'HATCHBACK': ['HATCHBACK'],
            'SPORTS': ['SPORTS CAR', 'COUPE', 'CONVERTIBLE'],
            'OTOMOBİL': ['CAR', 'VEHICLE', 'AUTOMOBILE']
        }

        tespit_edilen_tip = None
        for tip, anahtar_kelimeler in tipler.items():
            if any(kelime in caption for kelime in anahtar_kelimeler):
                tespit_edilen_tip = tip
                break

        markalar = ['BMW', 'MERCEDES', 'AUDI', 'TOYOTA', 'HONDA', 'FORD', 'VOLKSWAGEN',
                   'RENAULT', 'FIAT', 'OPEL', 'PEUGEOT', 'CITROEN', 'NISSAN', 'HYUNDAI']
        tespit_edilen_marka = None
        for marka in markalar:
            if marka in caption:
                tespit_edilen_marka = marka
                break

        durumlar = []
        if 'PARKED' in caption or 'PARKING' in caption:
            durumlar.append('PARK HALİNDE')
        if 'MOVING' in caption or 'DRIVING' in caption:
            durumlar.append('HAREKETLİ')

        yorum_parcalari = []
        if tespit_edilen_renk:
            yorum_parcalari.append(tespit_edilen_renk)
        if tespit_edilen_marka:
            yorum_parcalari.append(tespit_edilen_marka)
        if tespit_edilen_tip:
            yorum_parcalari.append(tespit_edilen_tip)
        elif not tespit_edilen_tip:
            yorum_parcalari.append('ARAÇ')
        if durumlar:
            yorum_parcalari.append(f"({', '.join(durumlar)})")

        yorum = ' '.join(yorum_parcalari) if yorum_parcalari else "DETAY TESPİT EDİLEMEDİ"
        return yorum
    except Exception as e:
        print(f"✗ Hata: {str(e)[:50]}")
        return "ANALİZ BAŞARISIZ"


def guvenli_crop(img, x1, y1, x2, y2, pad=30):
    h, w = img.shape[:2]
    return img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]


# ================================
# ANA SİSTEM
# ================================
def final_guvenlik_denetimi(resim_yolu):
    db_hazirla()
    frame = cv2.imread(resim_yolu)
    if frame is None:
        print(f"❌ Resim bulunamadı: {resim_yolu}")
        return

    results = coco_model(frame, conf=0.5, verbose=False)

    for r in results[0].boxes:
        label = coco_model.names[int(r.cls[0])]
        if label not in ["car", "bus", "truck"]: continue

        x1, y1, x2, y2 = map(int, r.xyxy[0])
        arac_crop = guvenli_crop(frame, x1, y1, x2, y2)

        # Plaka tespiti
        plates = license_plate_detector(arac_crop, conf=0.3, verbose=False)
        plaka = "OKUNAMADI"
        if plates[0].boxes:
            px1, py1, px2, py2 = map(int, plates[0].boxes[0].xyxy[0])
            plaka = plaka_oku_coklu_deneme(arac_crop[py1:py2, px1:px2])

        # VLM yorumu
        vlm_yorum = vlm_ile_arac_analizi(arac_crop)

        # İzin kontrolü
        izinli = plaka_izinli_mi(plaka)
        karar = " ONAY VERİLDİ" if izinli else " REDDEDİLDİ"

        # Loglama
        log_kaydet(plaka, label.upper(), vlm_yorum, karar)

        # Görselleştirme
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(arac_crop, cv2.COLOR_BGR2RGB))
        plt.title(f"TIP: {label.upper()} | PLAKA: {plaka}\nVLM: {vlm_yorum}\nKARAR: {karar} (detaylı bilgi için guvenlik sistemi database'ini kontrol ediniz)")
        plt.axis("off")
        plt.show()
        break


if __name__ == "__main__":
    images_dir = os.path.join(BASE_DIR, "images")
    if os.path.exists(images_dir):
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                final_guvenlik_denetimi(os.path.join(images_dir, img_file))
    else:
        print(f"❌ {images_dir} dizini bulunamadı!")