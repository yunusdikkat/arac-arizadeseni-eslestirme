# Araç Arıza Deseni Eşleştirme Projesi

Bu proje, kullanıcıların tarif ettiği araç arızaları için benzer geçmiş vakaları önermeyi amaçlayan bir otomotiv projesidir. Proje, 4 haftalık bir süreçte geliştirilmektedir.

## 1. Hafta: Veri Hazırlama ve Önişleme

### 1.1. Veri Toplama

Proje için gerekli veri, carcomplaints adlı siteden web scraping yöntemiyle toplanmıştır. Toplanan veri, araç arızalarına ilişkin metin açıklamalarını içermektedir. Veri seti, `pdate`, `pdate 2`, `marka`, `cheader`, `cheader 2` ve `comments` olmak üzere çeşitli sütunlar içermektedir. Özellikle `comments` sütunu, arıza tanımlarını içeren kullanıcı yorumlarıdır ve projenin temel veri kaynağını oluşturmaktadır.
* **pdate:** Arıza kaydının yayınlandığı tarih (örneğin, "Jul 20").
* **pdate 2:** Arıza kaydının yayınlandığı yıl (örneğin, "2017").
* **marka:** Arızanın bildirildiği aracın markası (örneğin, "Fiesta").
* **cheader:** Arızanın genel başlığı (örneğin, "Transmission Failed").
* **cheader 2:** Bu sütun, aracın arızalandığındaki kilometre bilgisini içermektedir.
* **comments:** Kullanıcı tarafından yazılan arıza açıklaması. Bu sütun, arızanın detaylı bir tarifini, semptomlarını ve kullanıcının yaşadığı sorunları içermektedir. Projenin temel metin verisi kaynağıdır.


### 1.2. Veri Ön İşleme

Toplanan veri, makine öğrenimi modelleri tarafından daha iyi işlenebilmesi için bir dizi ön işleme adımından geçirilmiştir. İşlemler, notebooks klasörünün içinde bulunan 02_data_preprocess.ipyn içerisinde gerçekleştirilmiştir. İşlemlere ait adımlar şunlardır:

* **Küçük harfe çevirme:** Tüm metinler küçük harfe dönüştürülerek tutarlılık sağlanmıştır.
* **Noktalama işaretlerini kaldırma:** Metinlerdeki noktalama işaretleri kaldırılarak gürültü azaltılmıştır.
* **Tokenizasyon:** Metinler, kelime veya kelime öbeklerine ayrılmıştır.
* **Stop word temizliği:** İngilizce stop word listesi kullanılarak anlamsız kelimeler metinlerden çıkarılmıştır.
* **Lemmatizasyon:** Kelimeler köklerine indirgenerek farklı çekimler aynı forma getirilmiştir.
* **Stemming:**  Kelimenin kökünü bulmak için yapılmıştır.

<br>Lemmatize ve stemming işlemlerinin sonucunda elde edilen veri setleri "preprocessed_data_lemmatized_only.csv" ve "preprocessed_data_stemmed_only.csv" olarak iki ayrı dosya olacak şekilde kaydedilmiştir.

Veri ön işleme adımları, `nltk` ve `re` kütüphaneleri kullanılarak Python'da uygulanmıştır.


### 1.3. Sonraki Adımlar

Projenin bir sonraki aşamasında, ön işlenmiş veriler kullanılarak makine öğrenimi modelleri eğitilecektir. Bu modeller, kullanıcıların tarif ettiği arızalar ile geçmiş arızalar arasında benzerlik kurmayı amaçlayacaktır.
## 2. Hafta: TF-IDF Vektörleştirme ve Word2Vec Modelleri Eğitimi

Bu hafta, ön işlenmiş metin verileri hem TF-IDF yöntemiyle vektörleştirilecek hem de Word2Vec modeli kullanılarak kelime vektörleri elde edilecektir.

### 2.1. TF-IDF Vektörleştirme

* TF-IDF (Term Frequency-Inverse Document Frequency) yöntemi, bir metin içindeki kelimelerin önemini ölçmek için kullanılan bir tekniktir. Bu adımda, her bir metin verisi, terim frekansları (TF) ve ters belge frekansı (IDF) kullanılarak bir vektöre dönüştürülür.
* `sklearn.feature_extraction.text` kütüphanesindeki `TfidfVectorizer` sınıfı, bu dönüşümü gerçekleştirmek için kullanılır.
* notebooks klaörünün içinde bulunan '03_vectorization_tf-idf' dosyasında bu işlem gerçekleştirilmiştir. Elde edilen bulgular dosya içinde bulunmaktadır.
  
### 2.2. Cosine Similarity (Kosinüs Benzerliği) Hesaplaması

* TF-IDF vektörleri elde edildikten sonra, metinler arasındaki benzerliği ölçmek için Cosine Similarity yöntemi kullanılır. Bu yöntem, iki vektör arasındaki açının kosinüsünü hesaplayarak metinlerin ne kadar benzer olduğunu belirler.
* `sklearn.metrics.pairwise` kütüphanesindeki `cosine_similarity` fonksiyonu, bu hesaplamayı yapmak için kullanılır.
*notebooks klaörünün içinde bulunan '03_vectorization_tf-idf' dosyasında bu işlem gerçekleştirilmiştir. Elde edilen bulgular dosya içinde bulunmaktadır

### 2.3. İlk Cümle için En Yüksek TF-IDF Skorlu Kelimeler

* TF-IDF vektörleştirme işleminden sonra, her metindeki en önemli kelimeler belirlenir. Bu, her metin için en yüksek TF-IDF skoruna sahip kelimelerin bulunmasıyla yapılır.
* Bu analiz, veri setindeki metinlerin anahtar temalarını ve özelliklerini anlamaya yardımcı olur.

### 2.4. Cosine Similarity Matrisi Oluşturma

* Tüm metinler arasındaki Cosine Similarity skorları bir matris içinde düzenlenir. Bu matris, hangi metinlerin birbirine daha çok benzediğini görselleştirmeyi ve analiz etmeyi kolaylaştırır.
* Bu matris, öneri sistemleri veya benzer arıza kayıtlarını bulma gibi uygulamalar için temel oluşturabilir.

### 2.5. Word2Vec Modelleri Eğitimi

* Word2Vec modeli, kelimelerin anlamlarını vektörler aracılığıyla temsil etmeyi amaçlayan bir tekniktir. Bu adımda, metin verilerinden kelime vektörleri elde edilir.
* Model eğitimi için farklı parametre kombinasyonları kullanılır. Bu parametreler, modelin performansını ve elde edilen vektörlerin kalitesini etkileyebilir.
* Model eğitimi notebooks klasörü içerisinde yer alan '04_word2vec' dosyasında gerçekleştirilmiştir.
* Seçilecek parametreler şunları içeriyor:
    * **Model tipi:** CBOW (Continuous Bag of Words) veya Skip-gram.
    * **Pencere boyutu:** Bir kelimenin bağlamını oluşturan kelime sayısı.
    * **Vektör boyutu:** Kelimelerin temsil edileceği vektörlerin boyutu.
* Eğitilen modeller, daha sonra kullanılmak üzere dosyaya kaydedilmiştir. Dosya adları, kullanılan parametreleri içerecek şekilde düzenlenmiştir (örneğin, "lemmatized_model_cbow_window2_dim100.model"). Elde edilen dosyalar, models klasörü içerisine kaydedilmiştir.

### 2.6. Model Değerlendirmesi ve Kullanımı

* Eğitilen Word2Vec modelleri, kelime benzerliği, kelime analojisi gibi görevlerde değerlendirilebilir.
* Modelin performansı ve elde edilen vektörlerin kalitesi analiz edilebilir.
* En iyi performansı gösteren modeller, proje kapsamında kullanılmak üzere seçilebilir.
1. **Gerekli Kütüphanelerin Kurulumu:**
    * Proje için gerekli Python kütüphanelerini yüklemek için `requirements.txt` dosyasını kullanın.
    * Komut satırında, `requirements.txt` dosyasının bulunduğu dizine gidin ve şu komutu çalıştırın:
        ```bash
        pip install -r requirements.txt
        ```

2.  **Veri Setinin Hazırlanması:**
    * Veri toplama adımlarını uygulamak için `notebooks/01_data_collection.ipynb` not defterini çalıştırın. Bu, veriyi toplar ve birleştirir.(İsteğe bağlıdır. Birleştirilmiş veri data klasöründe mevcuttur.)
    * Veri ön işleme adımlarını uygulamak için `notebooks/02_data_preprocess.ipynb` not defterini çalıştırın. Bu, metin verisini temizler ve işler.(İsteğe bağlıdır. İşlenmiş veri data klasöründe mevcuttur.)

3.  **Veri Analizi ve Vektörleştirme:**
    * Veri setinin özelliklerini anlamak için `notebooks/01_01_zipf_analizi.ipynb` ve `notebooks/02_01_lemmatizeandstemmed_zipf.ipynb` dosyalarını çalıştırarak Zipf Yasası analizini gerçekleştirin.
    * Metin verisini makine öğrenimi modelleri için sayısal vektörlere dönüştürmek için `notebooks/03_vectorization_tf-idf.ipynb` veya `notebooks/04_word2vec.ipynb` not defterlerini çalıştırın.

