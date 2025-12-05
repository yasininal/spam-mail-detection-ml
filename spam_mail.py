# 🧩 1. Kütüphanelerin Yüklenmesi

import numpy as np              # Sayısal işlemler için kullanılır (array, matematiksel hesaplamalar)
import pandas as pd             # Veri yükleme, düzenleme ve tablo işlemleri için kullanılır
import matplotlib.pyplot as plt # Görselleştirme için kullanılır
import seaborn as sns           # Görselleştirme için kullanılır

from sklearn.model_selection import train_test_split          # Veriyi eğitim ve test olarak ayırmak için
from sklearn.feature_extraction.text import TfidfVectorizer   # Metin verisini sayısal özelliklere dönüştürmek için
from sklearn.linear_model import LogisticRegression           # Lojistik regresyon sınıflandırma modeli
from sklearn.metrics import accuracy_score , confusion_matrix # Modelin doğruluk oranını ölçmek için accuracy_score ve görselleştirmek için de confusion_matrix

print("Tüm gerekli kütüphaneler başarıyla yüklendi!")


# 📂 2. Veriyi Yükleme

# Veriyi csv(comma seperated values) dosyasından pandas dataframe'ine(satır ve sütunlu durum) yükleme işlemi

raw_mail_data = pd.read_csv('/content/mail_data.csv')

# Verimizdeki sütunların incelenmesi

print(raw_mail_data.columns)

print(raw_mail_data)

# Dataframe'deki satır ve sütun sayıları

print(raw_mail_data.shape)

print("Satır sayısı:", raw_mail_data.shape[0])
print("Sütun sayısı:", raw_mail_data.shape[1])

# 🔧 3. Veri Ön Hazırlığı

# TF-IDF NaN ile çalışmaz → Hepsini boş string ile değiştiriyoruz.
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

# Tekrar eden mailleri temizliyoruz
mail_data.drop_duplicates(inplace=True)

# Temizlenmiş Dataframe'deki satır ve sütun sayıları

print(mail_data.shape)

print("\nSatır sayısı:", mail_data.shape[0])
print("Sütun sayısı:", mail_data.shape[1])


print("\nDuplicate Eden Mail Sayısı:", (raw_mail_data.shape[0] - mail_data.shape[0]))

# 🔍 4. Veri Hakkında Bilgi

# Dataframe'deki ilk 5 satırı yazdırma işlemi

mail_data.head()

# 🏷️ 5. Etiket Dönüşümü

# Makine öğrenmesi sayısal veri ister → spam = 1 , ham = 0

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 1
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 0

# ham  -  0

# spam  -  1

# 📊 📌 GÖRSEL: Spam / Ham Dağılımı

plt.figure(figsize=(6,4))
sns.countplot(data=mail_data, x="Category", palette="viridis")
plt.title("Spam ve Ham Dağılımı")
plt.xlabel("Kategori")
plt.ylabel("Sayı")
plt.show()

# 📤 6. Mesaj ve Etiket Ayrımı

# Verileri yazı ve etikete göre ayırma

X = mail_data['Message']

Y = mail_data['Category']

print(X)

print(Y)

# 📊 📌 GÖRSEL: Mesaj Uzunluğu Dağılımı

# Bu grafik, veri setimizin yapısını anlamamızı sağlar. Böylece modelden önce veri hakkında sezgi kazanıyoruz.


mail_data["length"] = mail_data["Message"].apply(lambda x: len(str(x)))

plt.figure(figsize=(10,5))
sns.histplot(mail_data["length"], bins=50, kde=True)
plt.title("Mesaj Uzunluk Dağılımı")
plt.xlabel("Mesaj Uzunluğu (karakter)")
plt.ylabel("Frekans")
plt.show()

# 📊 📌 GÖRSEL: Spam – Ham Mesaj Uzunluğu Karşılaştırması

plt.figure(figsize=(10,5))
sns.histplot(data=mail_data, x="length", hue="Category", bins=50, kde=True, palette="magma")
plt.title("Spam vs Ham Mesaj Uzunluğu")
plt.xlabel("Mesaj Uzunluğu (karakter)")
plt.ylabel("Frekans")
plt.show()

# ✂️ 7. Eğitim – Test Ayrımı

# Veri setini %80 eğitim ve %20 test olarak ayırdık
# 'stratify=Y' parametresini kullanarak spam oranının train ve test veri setlerinde aynı oranda olmasını sağladık
# Böyle modelimiz dengesiz olmuyor

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3, stratify=Y)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

# Değerleri al
values = [
    X.shape[0],        # Toplam veri
    X_train.shape[0],  # Eğitim verisi
    X_test.shape[0]    # Test verisi
]

labels = ['X (Toplam Veri)', 'X_train (Eğitim)', 'X_test (Test)']

# Grafik oluşturma
plt.figure(figsize=(8, 5))
plt.bar(labels, values)

plt.title("Veri Dağılımı (Toplam - Eğitim - Test)")
plt.xlabel("Veri Seti")
plt.ylabel("Adet")
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.show()


# 🧠 8. TF-IDF Dönüşümü

# Metin verilerini Lojistik regresyona girdi olarak kullanılabilecek özellik vektörlerine dönüştürme
# 1️⃣ min_df = 1
# Bir kelimenin modele alınması için en az 1 belgede geçmesi gerektiğini söyler.
# Yani tüm kelimeleri dahil eder (filtre yok).

#2️⃣ stop_words = 'english'
# İngilizce gereksiz kelimeleri kaldırır:
#   “the, is, and, a, an, on, at…” gibi.
# Bu sayede model gereksiz kelimelerle uğraşmaz, daha iyi öğrenir.

#3️⃣ lowercase = True
# Metnin tamamını küçük harfe çevirir.
# “Hello” ve “hello” aynı kelime olur → daha tutarlı bir model.

# TWEAKING DETAYI: 'ngram_range=(1, 2)' parametresini ekledik.
# Bu sayede model sadece "kazan" kelimesine değil, "ödül kazan" ikilisine de bakıyor.
# Bu yöntem spam tespitini çok daha keskin (precise) hale getirir.

feature_extraction = TfidfVectorizer(
    min_df = 1,
    stop_words='english',
    lowercase= True,
    ngram_range=(1, 2)
    )

X_train_features = feature_extraction.fit_transform(X_train)  # öğren ve sayısala dönüştür
X_test_features = feature_extraction.transform(X_test) #testi sadece dönüştür çünkü test setini de öğrenmek istemeyiz - overfitting

# Y_train ve Y_test değerlerini integer(tam sayı)'ya dönüştürme

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print(X_train)

# (0, 2329) 0.387
#   → 0. mailde 2329. kelimenin TF-IDF ağırlığı

print(X_train_features)

# 1. Spam Kelime Analizi Kısmı (Veri Çekme ve Hesaplama)

# 1.1. Feature (Özellik) İsimlerini Alalım
feature_names = feature_extraction.get_feature_names_out()

# 1.2. Y_train'i boolean bir diziye dönüştürelim (Spam: True, Ham: False)
is_spam_train = Y_train == 1

# 1.3. Sadece Spam Maillerin TF-IDF Vektörlerini Filtreleleyelim (Hata Düzeltmeli Kısım)
# Seyrek matrisi indekslemek için filtreyi .to_numpy() ile dönüştürüyoruz.
spam_features = X_train_features[is_spam_train.to_numpy()]

# 1.4. Her Özelliğin Toplam TF-IDF Skorunu Hesaplayalım
spam_scores = np.array(spam_features.sum(axis=0)).flatten()

# 1.5. Skorları ve Kelime İsimlerini Bir DataFrame'de Birleştirelim
df_scores = pd.DataFrame({'feature': feature_names, 'score': spam_scores})

# 1.6. En Yüksek Skorları Sıralayalım ve top_spam_words değişkenine atayalım (İlk 20)
top_spam_words = df_scores.sort_values(by='score', ascending=False).head(20)
print("\n--- En Belirleyici Top 20 Kelime/N-gram ---")
print(top_spam_words)


# 2. Görselleştirme Kısmı

# Yatay grafikte en yüksek skorun en üstte görünmesi için artan sıralama yapılır
top_spam_words_plot = top_spam_words.sort_values(by='score', ascending=True)

plt.figure(figsize=(10, 8))

# Yatay çubukları çizin
plt.barh(top_spam_words_plot['feature'], top_spam_words_plot['score'], color='firebrick')

# Başlıklar ve etiketler ekleme
plt.title('Spam Maillerde En Belirleyici Kelimeler (Gerçek TF-IDF Skorları)', fontsize=14)
plt.xlabel('Toplam TF-IDF Skoru', fontsize=12)
plt.ylabel('Kelime / N-gram', fontsize=12)

# Etiketlerin kesilmemesi için düzeni ayarlama
plt.tight_layout()

# Görseli kaydetme
plt.savefig('top_spam_words_barchart.png')

# 🤖 9. Model Kurma (Logistic Regression)

# Logistic Regression, bir veriyi iki sınıftan birine atamak için kullanılan bir sınıflandırma algoritmasıdır.

model = LogisticRegression()

# Modeli eğitim verisiyle eğitme

model.fit(X_train_features, Y_train)

# 📊 10. Eğitim & Test Doğruluğu

# Eğitim verisi için tahmin yapma

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print('Accuracy on training data : ', accuracy_on_training_data)

# Test Verisi için tahmin yapma

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print('Accuracy on test data : ', accuracy_on_test_data)

# 📊 📌 CONFUSION MATRIX GÖRSELİ

cm = confusion_matrix(Y_test, prediction_on_test_data)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show()

# ✉️ 11. Yeni Bir Maili Test Etme

input_mail = ["Win a Free iPhone Now! Congratulations! You have been selected to receive a free iPhone. Click here to claim your prize."]

# String ifadeyi vektöre dönüştürme
input_data_features = feature_extraction.transform(input_mail)

# Tahmin yapma işlemi
prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==0):
  print('Ham mail')

else:
  print('Spam mail')


# Veri setinde olmayan örnek spam mail test edebilirsiniz:
# Win a Free iPhone Now! Congratulations! You have been selected to receive a free iPhone. Click here to claim your prize.

# Veri setinde olmayan örnek ham mail test edebilirsiniz:
# Class Notes. Hello, I’ve attached the notes from yesterday’s class. Let me know if you have any questions.

# 🧱 12. Pipeline Özet Şeması

'''

1-          HAM / SPAM DATA

                   ↓

2-          Veri Temizleme

                   ↓

3-          TF-IDF Dönüşümü

                   ↓

4-   Model Eğitimi (Logistic Regression)

                   ↓

5-          Doğruluk Ölçme

                   ↓

6-          Yeni Mail Tahmini


'''