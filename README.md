# 📧 Spam Mail Tespit Uygulaması (Spam Detection Web App)

Bu proje, Makine Öğrenmesi (Logistic Regression) ve Doğal Dil İşleme (NLP) tekniklerini kullanarak, girilen e-posta metninin **Spam (Gereksiz/Zararlı)** mı yoksa **Ham (Güvenli/Normal)** mı olduğunu tespit eden web tabanlı bir uygulamadır.

Arayüz için **Flask**, model eğitimi için **Scikit-Learn** kullanılmıştır.

---

## 🚀 Özellikler

* **Gerçek Zamanlı Analiz:** Kullanıcı arayüzüne girilen metni anlık olarak analiz eder.
* **Yüksek Doğruluk:** TF-IDF vektörleştirme ve Lojistik Regresyon ile eğitilmiş model.
* **Kullanıcı Dostu Arayüz:** Sade ve anlaşılır HTML/CSS tasarımı.
* **Görsel Geri Bildirim:** Sonuca göre renkli (Kırmızı/Yeşil) uyarı sistemi.

---

## 📂 Proje Yapısı

Dosya düzeni aşağıdaki gibidir:

```text
SpamDedektoru/
│
├── app.py                # Flask sunucu dosyası (Backend)
│
├── spam_mail.py          # Spam mail kodu
│
├── mail_data.csv         # Veri seti
│
├── spam_model.pkl        # Eğitilmiş AI Modeli
├── vectorizer.pkl        # TF-IDF Kelime Dönüştürücü
│
├── static/               # Statik dosyalar
│   └── style.css         # Tasarım kodları
│
└── templates/            # HTML şablonları
    └── index.html        # Kullanıcı arayüzü

```


## 🛠️ Kurulum
Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.

1. Gereksinimleri Yükleyin
Python'un yüklü olduğundan emin olun. Ardından terminal veya komut satırında gerekli kütüphaneleri yükleyin:

```

pip install flask scikit-learn pandas numpy

```
2. Modeli Hazırlama (Eğer .pkl dosyaları yoksa)
Eğer klasörde spam_model.pkl ve vectorizer.pkl dosyaları yoksa, önce model eğitim kodlarını (Jupyter Notebook veya Python scripti) çalıştırarak bu dosyaların oluşmasını sağlayın.

3. Uygulamayı Başlatma
Terminali proje klasöründe açın ve şu komutu girin:

```

python app.py

```
Terminalde Running on http://127.0.0.1:5000 yazısını gördüğünüzde tarayıcınızdan bu adrese gidin.

🧪 Test İçin Örnek Veriler
Modeli denemek için aşağıdaki İngilizce metinleri kopyalayıp uygulamaya yapıştırabilirsiniz:

🔴 Spam Örnekleri (Bunlar Spam Çıkmalı)

```

"URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010."

"Congratulations! You've been selected for a chance to win a $1000 Walmart Gift Card. Click here to claim your prize now!"


```
🟢 Ham (Güvenli) Örnekleri (Bunlar Güvenli Çıkmalı)

```

"Hey, are we still on for dinner tonight? Let me know so I can make a reservation."

"Can you send me the report by tomorrow morning? Thanks."

```

🧠 Model Hakkında Teknik Bilgi
Veri Seti: SMS Spam Collection Dataset kullanılmıştır.

Ön İşleme: Stopwords temizliği, küçük harfe çevirme.

Vektörleştirme: TfidfVectorizer (ngram_range=(1,2) kullanılarak kelime grupları dikkate alınmıştır).

Algoritma: LogisticRegression (İkili sınıflandırma için).

👤 İletişim & Geliştirici
Bu proje Yasin Taha İnal tarafından geliştirilmiştir.   