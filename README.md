# Derin Pekiştirmeli Öğrenme Algoritmalarının Karşılaştırmalı Analizi

Bu proje, derin pekiştirmeli öğrenme (DRL) algoritmalarını (DQN, A3C, PPO) LunarLander-v3 ortamında karşılaştırmayı ve hiperparametre optimizasyonu ile performanslarını artırmayı hedeflemektedir. Çalışma, Bayes optimizasyonu ile elde edilen sonuçları ve algoritmaların eğitim süreçlerini detaylı olarak incelemektedir.

## İçindekiler

- [Proje Tanımı](#proje-tanımı)
- [Kullanılan Algoritmalar](#kullanılan-algoritmalar)
- [Ortam](#ortam)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Dosya Yapısı](#dosya-yapısı)
- [Sonuçlar](#sonuçlar)
- [Katkıda Bulunma](#katkıda-bulunma)

## Proje Tanımı

Bu proje, derin pekiştirmeli öğrenme algoritmalarının performansını pratik bir şekilde değerlendirmek ve hiperparametre optimizasyonunun önemini vurgulamak için geliştirilmiştir. Derin Q-ağları (DQN), eşzamansız avantaj aktör-eleştirmen (A3C) ve yakın politika optimizasyonu (PPO) algoritmalarının performansları, LunarLander-v3 ortamında karşılaştırılmıştır. Her algoritmanın hiperparametreleri, Bayes optimizasyonu ile optimize edilmiş ve bu optimizasyonun eğitim süreçlerine olan etkisi incelenmiştir.

## Kullanılan Algoritmalar

-   **Derin Q-Ağları (DQN):** Deneyim tekrarı ve hedef ağları kullanarak değer fonksiyonunu öğrenen bir algoritma.
-   **Eşzamansız Avantaj Aktör-Eleştirmen (A3C):** Çoklu işçi ile paralel öğrenme sağlayarak politika ve değer fonksiyonunu öğrenen bir algoritma.
-   **Yakın Politika Optimizasyonu (PPO):** Politika gradyan yöntemlerini kullanarak kararlı bir öğrenme sağlayan bir algoritma.

## Ortam

-   **LunarLander-v3:** Gymnasium kütüphanesinde bulunan, bir uzay aracının yumuşak iniş yapmayı öğrendiği bir ortam. Sürekli durum ve ayrık eylem uzayına sahiptir.

## Kurulum

1.  **GitHub deposunu klonlayın:**

    ```bash
    git clone https://github.com/SinanTaskinlar/RL-Algorithms-LunarLander.git
    cd RL-Algorithms-LunarLander
    ```

2.  **Gerekli kütüphaneleri kurun:**

    ```bash
    pip install -r requirements.txt
    ```
    
    `requirements.txt` dosyasının içeriği:
    ```
    torch
    optuna
    gymnasium
    matplotlib
    numpy
    ```

## Kullanım

1.  **Proje dosyasını çalıştırın:**
    ```bash
    python main.py
    ```
    Bu komut, tüm algoritmaları eğitir ve sonuçları analiz eder. Eğitim sırasında loglar ve grafikler oluşturulur.

## Dosya Yapısı
Use code with caution.
Markdown
RL-Algorithms-LunarLander/
├── main.py # Ana çalıştırma dosyası
├── Models/ # Model dosyaları
│ ├── A3C.py # A3C algoritma modeli ve eğitim kodu
│ ├── DQN.py # DQN algoritma modeli ve eğitim kodu
│ └── PPO.py # PPO algoritma modeli ve eğitim kodu
├── Optimizer/ # Optimizasyon dosyaları
│ └── Bayes.py # Bayes optimizasyon kodu
├── Utilities/ # Yardımcı araçlar
│ └── Utils.py # Yardımcı fonksiyonlar ve sınıflar
├── SavedModels/ # Kaydedilen modeller
│ └── dqn/
│ └── a3c/
│ └── ppo/
├── requirements.txt # Gerekli kütüphaneler
└── README.md # Bu README dosyası

## Sonuçlar

Proje sonuçları, eğitim sırasında elde edilen loglar, grafikler ve model performanslarını içerir. Bayes optimizasyonu sonucunda, aşağıdaki en iyi hiperparametre değerleri bulunmuştur:

*   **DQN:** Öğrenme hızı (1.72e-05), keşif azalma oranı (0.9976), mini-batch boyutu (36), hedef ağ güncelleme sıklığı (33).
*   **A3C:** Öğrenme hızı (4.09e-05), entropi katsayısı (0.0048), genelleştirilmiş avantaj kestirim katsayısı (0.9802), ödül ölçeklendirme faktörü (0.019).
*   **PPO:** Öğrenme hızı (3.47e-05), politika güncelleme oranı (0.115), entropi katsayısı (0.0392), mini-batch boyutu (113), güncelleme adımı sayısı (5), ödül ölçeklendirme faktörü (0.048).

Eğitim süreçleri sonucunda, A3C algoritması 625.8 saniyede, PPO algoritması 3013.8 saniyede tamamlanmıştır. DQN algoritması ise 5072.0 saniyede tamamlanmıştır.
3060ti ve Macbook Air M1 kullanılmıştır.

## Katkıda Bulunma

Projenin geliştirilmesine katkıda bulunmak isterseniz, aşağıdaki adımları takip edebilirsiniz:

1.  **Fork**.
2.  **Branch**
3.  **Commit**
4.  **Push**
5.  **Pull Request**

## Lisans

-
