from flask import Flask, render_template, request
import pickle
import numpy as np

# Flask uygulamasını başlatıyoruz
app = Flask(__name__)

# 1. Eğitilmiş Modeli ve Vektörleştiriciyi Yüklüyoruz
# 'rb' = read binary (ikili formatta okuma)
try:
    model = pickle.load(open('spam_model.pkl', 'rb'))
    feature_extraction = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    print("HATA: .pkl dosyaları bulunamadı! Lütfen önce 'Adım 0'ı uygulayın.")
    exit()

# Ana Sayfa Rotası (http://127.0.0.1:5000/)
@app.route('/')
def home():
    return render_template('index.html')

# Tahmin Yapma Rotası (Butona basılınca burası çalışır)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Formdan gelen metni al
        mail_message = request.form['mail_content']
        
        # 2. Metni modele uygun formata (vektöre) çevir
        # input_data_features değişkenini modelin anlayacağı dile çeviriyoruz
        data_features = feature_extraction.transform([mail_message])
        
        # 3. Model ile tahmin yap
        prediction = model.predict(data_features)
        
        # 4. Sonucu yorumla
        if prediction[0] == 1:
            res_text = "🚨 DİKKAT: Bu bir SPAM maildir!"
            res_class = "spam" # CSS için sınıf adı
        else:
            res_text = "✅ GÜVENLİ: Bu bir HAM (Normal) maildir."
            res_class = "ham" # CSS için sınıf adı
            
        # 5. Sonucu tekrar index.html sayfasına gönder
        return render_template('index.html', 
                               prediction_text=res_text, 
                               prediction_class=res_class,
                               message=mail_message)

if __name__ == '__main__':
    app.run(debug=True)