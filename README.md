[README.md](https://github.com/user-attachments/files/33323911/README.md)
# 🏠 House Price Prediction

Kaggle **House Prices – Advanced Regression Techniques** veri seti üzerinde ev satış fiyatlarını (`SalePrice`) tahmin eden uçtan uca bir makine öğrenmesi projesi. Keşifçi veri analizi, özellik mühendisliği, kodlama (encoding), model karşılaştırma ve hiperparametre optimizasyonu adımlarını tek bir script içinde barındırır.

---

## 📁 Proje Yapısı

```
.
├── house_price.py     # Tüm analiz ve modelleme akışı
├── train.csv          # Eğitim verisi (Kaggle)
├── test.csv           # Test verisi (Kaggle)
├── houseprice.csv     # Çıktı: Kaggle submission dosyası
└── importances.png    # Çıktı: Özellik önem grafiği
```

## 📊 Veri Seti

- **Kaynak:** [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Hedef değişken:** `SalePrice`
- `train.csv` ve `test.csv` birleştirilerek tüm ön işleme adımları ortak uygulanır; modelleme öncesinde `SalePrice` değerinin dolu/boş olmasına göre tekrar ayrılır.

## ⚙️ Kurulum

Python 3.9+ önerilir.

```bash
pip install numpy pandas seaborn matplotlib scikit-learn xgboost lightgbm catboost joblib
```

> Script içinde `graphviz`, `pydotplus` ve `skompiler` de import ediliyor ancak akışta kullanılmıyor. Kurmak istemezseniz bu import satırlarını kaldırabilirsiniz.

## ▶️ Çalıştırma

`train.csv` ve `test.csv` dosyalarını script ile aynı klasöre koyun:

```bash
python house_price.py
```

> ⏱️ CatBoost için GridSearch 108 parametre kombinasyonunu 5 katlı çapraz doğrulama ile dener (540 fit). Donanıma bağlı olarak uzun sürebilir.

---

## 🔄 İş Akışı

### 1. Veri Yükleme ve Keşif
- Train ve test verileri birleştirilir, tekrarlayan kayıtlar kontrol edilir.
- `grab_col_names()` ile değişkenler **kategorik**, **sayısal** ve **kardinal** olarak ayrılır.
- `MSSubClass` sayısal görünse de kategorik olarak ele alınır.

### 2. Veri Temizliği
- `GarageYrBlt` içindeki hatalı **2207** değeri düzeltilir.
- Yıl değişkenleri yaşa dönüştürülür: `BltAge`, `RemodAge`, `GarageAge` (referans yıl: 2011).
- Bazı sayısal ama düşük kardinaliteli değişkenler (banyo, yatak odası, garaj kapasitesi vb.) sayısal listeye taşınır.

### 3. Eksik Değerler
- Bodrum ve garaj değişkenlerindeki boşluklar anlamlı etiketlerle doldurulur (`NoBsmt`, `NoGar`).
- Eksik oranı **%40'ın üzerindeki** kolonlar silinir.
- Kalan eksikler: sayısallarda **medyan**, kategoriklerde **mod** ile doldurulur.

### 4. Aykırı Değerler
- Çeyrek eşikleri **%5 – %95** alınarak IQR yöntemiyle sınırlar hesaplanır.
- Aykırı değerler silinmek yerine sınır değerlere baskılanır (`replace_with_thresholds`).

### 5. Rare Encoding
- Oranı **%1'in altında** kalan kategoriler `"Rare"` etiketinde toplanır.

### 6. Özellik Mühendisliği
| Yeni Değişken | Açıklama |
|---|---|
| `CompRate` | Bitmiş bodrum alanının kat alanlarına oranı |
| `TotalPorch` | Tüm sundurma alanlarının toplamı |
| `BltAge` / `RemodAge` / `GarageAge` | Bina, tadilat ve garaj yaşı |

- Tek değerli ve neredeyse tek değerli (iki sınıflı, bir sınıfı <%1) kolonlar çıkarılır.

### 7. Encoding
- **Ordinal encoding:** Kalite/durum değişkenleri (`ExterQual`, `BsmtQual`, `KitchenQual`, `GarageQual`, `Functional` vb.) anlamlı sıralı sayılara eşlenir.
- **Target group encoding:** Yüksek kardinaliteli `Neighborhood`, `Exterior1st`, `Exterior2nd`, `MSSubClass` değişkenleri, ortalama `SalePrice` değerine göre 5 gruba (`qcut`) ayrılır → `*_GRP` kolonları.
- **One-hot encoding:** 10 ve daha az sınıflı kalan kategorik değişkenlere uygulanır (`drop_first=True`).

### 8. Modelleme
Aşağıdaki modeller 5 katlı çapraz doğrulama ile RMSE üzerinden karşılaştırılır:

`LinearRegression` · `DecisionTree` · `RandomForest` · `GradientBoosting` · `XGBoost` · `LightGBM` · `CatBoost`

Hedef değişken çarpık dağıldığı için **`log1p` dönüşümü** uygulanır; tahminler `expm1` ile gerçek ölçeğe geri çevrilir.

### 9. Hiperparametre Optimizasyonu (CatBoost)
```python
{
    "iterations":       [500, 1000, 1500],
    "depth":            [4, 6, 8, 10],
    "min_data_in_leaf": [1, 5, 10],
    "learning_rate":    [0.03, 0.05, 0.1]
}
```

### 10. Tahmin ve Çıktı
- En iyi CatBoost modeliyle test seti tahmin edilir ve `houseprice.csv` (`Id`, `SalePrice`) olarak kaydedilir.
- En önemli 20 özellik `importances.png` olarak görselleştirilir.

---

## 📈 Sonuçlar

| Model | Ölçek | RMSE |
|---|---|---|
| LightGBM (5-fold CV) | Log | ~0.1332 |
| LightGBM (hold-out %20) | Gerçek ($) | ~25.765 |
| **CatBoost (optimize, hold-out %20)** | **Gerçek ($)** | **~22.870 – 23.500** |

> Referans: Ortalama `SalePrice` ≈ **180.921 $**. Optimize CatBoost modelinin hatası ortalama fiyatın yaklaşık **%12–13**'ü düzeyindedir.

## 📝 Notlar

- `plot_importance` fonksiyonunda `plt.savefig()` çağrısı `plt.show()` sonrasında yapıldığı için kaydedilen görsel boş olabilir; kaydetmeyi `show()` öncesine almak önerilir.
- Ön işleme train + test birleşik veri üzerinde yapıldığından, aykırı değer eşikleri ve doldurma değerleri test verisinden de etkilenir.

## 🛠️ Kullanılan Teknolojiler

`Python` · `pandas` · `NumPy` · `scikit-learn` · `CatBoost` · `LightGBM` · `XGBoost` · `seaborn` · `matplotlib`
