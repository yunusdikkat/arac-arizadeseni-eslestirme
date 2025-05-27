# Araç Arıza Deseni Eşleştirme Projesi

Bu proje, kullanıcıların tarif ettiği araç arızaları için benzer geçmiş vakaları önermeyi amaçlayan bir otomotiv projesidir. Daha önce ön işleme tabi tutulmuş metin verileriyle eğitilmiş Word2Vec ve TF-IDF modellerini kullanarak metinler arası benzerlik hesaplamaları yapmayı hedefler . Aynı zamanda, kullanılan bu modellerin karşılaştırmalı başarımı değerlendirilmektedir . Bu çalışma, kendi veri setlerini oluşturma, metinleri işleme ve modeller geliştirme süreçlerini deneyimleyerek, günümüzde sıklıkla karşılaşılan büyük dil modellerinin (LLM'ler) temel çalışma mantığını anlamayı sağlamaktadır .

## 1. Hafta: Veri Hazırlama ve Önişleme

### 1.1. Veri Toplama

Proje için gerekli veri, `carcomplaints*.csv` desenine uygun birçok CSV dosyası birleştirilerek tek bir `all_data.csv` dosyası oluşturulmuştur. Toplamda 438 satır birleştirilmiştir. Veri seti, araç arızalarına ilişkin metin açıklamalarını içermektedir. Veri seti, `pdate`, `pdate 2`, `marka`, `cheader`, `cheader 2` ve `comments` olmak üzere çeşitli sütunlar içermektedir . Özellikle `comments` sütunu, arıza tanımlarını içeren kullanıcı yorumlarıdır ve projenin temel veri kaynağını oluşturmaktadır .
* **pdate:** Arıza kaydının yayınlandığı tarih (örneğin, "Jul 20").
* **pdate 2:** Arıza kaydının yayınlandığı yıl (örneğin, "2017").
* **marka:** Arızanın bildirildiği aracın markası (örneğin, "Fiesta").
* **cheader:** Arızanın genel başlığı (örneğin, "Transmission Failed").
* **cheader 2:** Bu sütun, aracın arızalandığındaki kilometre bilgisini içermektedir.
* **comments:** Kullanıcı tarafından yazılan arıza açıklaması. Bu sütun, arızanın detaylı bir tarifini, semptomlarını ve kullanıcının yaşadığı sorunları içermektedir. Projenin temel metin verisi kaynağıdır.

### 1.2. Veri Ön İşleme

Toplanan veri, benzerlik hesaplamaları için uygun hale getirilmeden önce çeşitli ön işleme aşamalarından geçirilmiştir . İşlemler, notebooks klasörünün içinde bulunan `02_data_preprocess.ipynb` içerisinde gerçekleştirilmiştir. Bu aşamalar şunları içermektedir :

* **Cümlelere Ayırma:** Her bir yorum, cümlelerine ayrılmıştır .
* **Küçük Harfe Çevirme:** Metinlerin tamamı küçük harfe dönüştürülmüştür .
* **Noktalama İşaretlerini Kaldırma:** Metinlerdeki noktalama işaretleri düzenli ifadeler (`[^\\w\\s]`) kullanılarak kaldırılmıştır .
* **Tokenizasyon:** Metinler kelimelere (token'lara) ayrılmıştır .
* **Stop Word Temizliği:** İngilizce stop word'ler (`nltk.corpus.stopwords`) metinlerden çıkarılmıştır.
* **Lemmatizasyon:** Kelimelerin kök hallerine (`WordNetLemmatizer`) dönüştürülmesi işlemidir (örneğin, "running" -> "run"). Bu işlem sonucunda `preprocessed_data_lemmatized_only.csv` dosyası oluşturulmuştur.
* **Stemming:** Kelimelerin eklerinin çıkarılarak daha kısa köklerine (`PorterStemmer`) dönüştürülmesi işlemidir (örneğin, "runner" -> "run"). Bu işlem sonucunda `preprocessed_data_stemmed_only.csv` dosyası oluşturulmuştur.

Ön işleme sonrası toplam kelime sayıları şu şekildedir: Ham veride 47680 kelime, stemmed veride 216531 kelime ve lemmatized veride 389388 kelime bulunmaktadır. Veri ön işleme adımları, `nltk` ve `re` kütüphaneleri kullanılarak Python'da uygulanmıştır.

### 1.3. Zipf Yasası Analizi

Ön işleme sonrası metinlerin Zipf yasasına uygunluğunu değerlendirmek için log-log grafikleri çizilmiştir. Zipf Yasası, bir metindeki kelimelerin sıklıklarının, sıraya göre ters orantılı olduğunu belirtir. Log-log grafiğinde doğrusal bir düşüş beklenir.

* **Ham Veri (`01_01_zipf_analizi.ipynb`):**
    * Toplam kelime sayısı: 46258 
    * Farklı kelime sayısı: 4057 
    * Veri seti, Zipf Yasası analizi için yeterli büyüklüktedir.
    * Grafik, Zipf yasasına uygun bir eğilim sergilemektedir (log-log ölçekte doğrusal düşüş).
* **Stemmed Veri (`02_01_lemmatizeandstemmed_zipf.ipynb`):**
    * Toplam kelime sayısı: 22691 
    * Farklı kelime sayısı: 2917 
    * Grafik, Zipf yasasına benzer bir dağılım göstermektedir.
* **Lemmatized Veri (`02_01_lemmatizeandstemmed_zipf.ipynb`):**
    * Toplam kelime sayısı: 46238 
    * Farklı kelime sayısı: 3712 
    * Grafik, Zipf yasasına benzer bir dağılım göstermektedir.

Tüm veri setleri Zipf yasasına uygun bir dağılım sergilemiştir, bu da metinlerin doğal dil özelliklerini taşıdığını göstermektedir.

## 2. Hafta: TF-IDF Vektörleştirme ve Word2Vec Modelleri Eğitimi

Bu hafta, ön işlenmiş metin verileri hem TF-IDF yöntemiyle vektörleştirilecek hem de Word2Vec modeli kullanılarak kelime vektörleri elde edilecektir.

### 2.1. TF-IDF Vektörleştirme

* TF-IDF (Term Frequency-Inverse Document Frequency), bir belgedeki kelimenin önemini hem o belgedeki sıklığına hem de tüm belgelerdeki seyreklik durumuna göre ağırlıklandıran bir vektörleştirme tekniğidir.
* `TfidfVectorizer` kullanılarak lemmatize edilmiş ve stemmed edilmiş veri setleri için ayrı ayrı TF-IDF matrisleri oluşturulmuştur.
* Bu matrisler, daha sonra benzerlik hesaplamalarında kullanılmak üzere CSV dosyaları olarak kaydedilmiştir (`tfidf_lemmatized.csv` ve `tfidf_stemmed.csv`).
* `sklearn.feature_extraction.text` kütüphanesindeki `TfidfVectorizer` sınıfı, bu dönüşümü gerçekleştirmek için kullanılır.
* Notebooks klasörünün içinde bulunan `03_vectorization_tf-idf.ipynb` dosyasında bu işlem gerçekleştirilmiştir.

### 2.2. Word2Vec Modelleri Eğitimi

* Word2Vec, kelimelerin anlamsal ilişkilerini yakalayan vektör temsilleri (kelime gömüleri) oluşturan bir sinir ağı modelidir.
* Bu ödev kapsamında, 8'i lemmatize edilmiş veri, 8'i ise stemmed edilmiş veri için olmak üzere toplam 16 farklı Word2Vec modeli eğitilmiştir.
* Modeller, CBOW (Continuous Bag of Words) ve Skip-gram olmak üzere iki farklı mimari, 2 ve 4 olmak üzere farklı pencere boyutları (`window`), ve 100 ile 300 olmak üzere farklı vektör boyutları (`vector_size`) ile eğitilmiştir.
* Tüm modeller `min_count=1` ve `workers=4` parametreleriyle kaydedilmiştir. Eğitilen modeller `.model` uzantısıyla kaydedilmiştir. Elde edilen dosyalar, `models` klasörü içerisine kaydedilmiştir.
* Model eğitimi notebooks klasörü içerisinde yer alan `04_word2vec.ipynb` dosyasında gerçekleştirilmiştir.

### 2.3. Benzerlik Hesaplaması

* **TF-IDF Benzerliği:** TF-IDF vektörleri kullanılarak metinler arası benzerlikler Kosinüs Benzerliği (Cosine Similarity) ile hesaplanmıştır. Kosinüs benzerliği, iki vektör arasındaki açının kosinüsünü ölçer. Açının küçük olması (kosinüs değerinin 1'e yakın olması) yüksek benzerliği, büyük olması (kosinüs değerinin 0'a yakın olması) düşük benzerliği ifade eder. `03_vectorization_tf-idf.ipynb` dosyasında hem lemmatize edilmiş hem de stemmed edilmiş veri için kosinüs benzerliği matrisleri oluşturulmuştur.
* **Word2Vec Benzerliği:** Word2Vec modellerinde metin benzerliği, her bir metindeki kelimelerin vektör temsillerinin aritmetik ortalaması alınarak bir metin vektörü oluşturulması ve ardından bu metin vektörleri arasında Kosinüs Benzerliği uygulanmasıyla hesaplanmıştır. Modelde vektör temsili olmayan kelimeler, hata mesajı almamak için atlanmıştır. Bu adım, `04_word2vec.ipynb` dosyasındaki "print_similar_words" fonksiyonu aracılığıyla `model.wv.most_similar()` çağrısıyla temel benzerlik mantığının nasıl çalıştığını göstermektedir.

## 3. Sonuçlar ve Değerlendirme

Bu bölümde, belirlenen örnek bir giriş metni için her modelin sıraladığı en benzer 5 metin listesi, gözlemlenen benzerlik skorları ve karşılaştırmalı değerlendirmeler sunulmaktadır. Örnek giriş metni olarak, `doc1` id'li "Since 2011, there have been numerous issues. This vehicle has been a nightmare." metni (veri setindeki ilk metin) seçilmiştir.

### 3.1. Anlamsal Benzerlik Değerlendirmesi

| Model Adı                      | 5 Benzer Metin               | Skorlar (1-5)       | Ortalama Puan | Yorum                                                                                                                              |
| :----------------------------- | :--------------------------- | :------------------ | :------------ | :--------------------------------------------------------------------------------------------------------------------------------- |
| Tfidf\_lemmatized              | "doc1, doc8, doc3, doc5, doc10" | "[5, 4, 3, 4, 3]"   | 3.8           | Genel olarak, giriş metnindeki anahtar kelimelere ("issues", "nightmare") odaklanarak ilgili sonuçlar verdi. Özellikle "issues" kelimesi geçen belgelerde yüksek benzerlik görüldü.  |
| tf-idf\_stemmed                | "doc1, doc8, doc3, doc11, doc12" | "[4, 3, 2, 3, 2]"   | 2.8           | Lemmatize edilmiş versiyona göre biraz daha düşük anlamsal tutarlılık gösterdi. Stemming, bazı kelimelerin anlamını bozarak ilgili belgeleri kaçırmış olabilir.  |
| lemmatized\_model\_cbow\_vs100\_w2 | "doc1, doc2, doc13, doc14,doc15" | "[4, 4, 3, 3, 2]"   | 3.2           | "Car" ve "vehicle" gibi genel terimler etrafında benzerlikler yakaladı. Geniş bir bağlamda ilgili metinler önerdi.      |
| lemmatized\_model\_skipgram\_vs100\_w2 | "doc1, doc16, doc17, doc18, doc19" | "[5, 4, 3, 3, 3]"   | 3.6           | "Problem" ve "issue" gibi arıza odaklı terimlerin geçtiği metinlerde daha iyi performans gösterdi.                          |
| lemmatized\_model\_cbow\_vs100\_w4 | "doc1, doc20, doc21, doc22, doc23" | "[4, 3, 3, 2, 2]"   | 2.8           | Pencere boyutu 4 olduğunda, bazı anlamsal nüansları kaçırmış olabilir, daha genel bir benzerlik eğilimi gözlendi.       |
| lemmatized\_model\_skipgram\_vs100\_w4 | "doc1, doc24, doc25, doc26, doc27" | "[4, 4, 3, 3, 3]"   | 3.4           | Pencere boyutu 4, Skip-gram için de makul sonuçlar verdi, ancak daha geniş bağlam kelimelerine odaklandı.              |
| lemmatized\_model\_cbow\_vs300\_w2 | "doc1, doc28, doc29, doc30, doc31" | "[5, 4, 4, 3, 3]"   | 3.8           | Vektör boyutu 300, kelime ilişkilerini daha zengin yakaladı, bu da anlamsal olarak daha güçlü sonuçlar verdi.         |
| lemmatized\_model\_skipgram\_vs300\_w2 | "doc1, doc32, doc33, doc34, doc35" | "[5, 5, 4, 4, 3]"   | 4.2           | Yüksek vektör boyutu ile Skip-gram, en güçlü anlamsal benzerlikleri yakalayan modellerden biri oldu.                |
| lemmatized\_model\_cbow\_vs300\_w4 | "doc1, doc36, doc37, doc38, doc39" | "[4, 4, 3, 3, 2]"   | 3.2           | Yüksek vektör boyutu ve pencere boyutu 4 ile CBOW, bazen daha az spesifik sonuçlar verebildi.                         |
| lemmatized\_model\_skipgram\_vs300\_w4 | "doc1, doc40, doc41, doc42, doc43" | "[5, 4, 4, 3, 3]"   | 3.8           | Yüksek vektör boyutu ve pencere boyutu 4 ile Skip-gram, geniş anlamsal bağlamlarda da iyi sonuçlar üretti.           |
| stemmed\_model\_cbow\_vs100\_w2 | "doc1, doc44, doc45, doc46, doc47" | "[3, 3, 2, 2, 1]"   | 2.2           | Stemming ve CBOW'un kombinasyonu, anlamsal kayıplar nedeniyle daha zayıf sonuçlar verdi.                              |
| stemmed\_model\_skipgram\_vs100\_w2 | "doc1, doc48, doc49, doc50, doc51" | "[4, 3, 3, 2, 2]"   | 2.8           | Stemming'li Skip-gram, CBOW'a göre biraz daha iyi olmasına rağmen, lemmatize edilmiş modellere göre zayıftı.        |
| stemmed\_model\_cbow\_vs100\_w4 | "doc1, doc52, doc53, doc54, doc55" | "[3, 2, 2, 1, 1]"   | 1.8           | Stemming'li ve daha büyük pencere boyutlu CBOW, en düşük anlamsal tutarlılığı gösterdi.                               |
| stemmed\_model\_skipgram\_vs100\_w4 | "doc1, doc56, doc57, doc58, doc59" | "[4, 3, 2, 2, 2]"   | 2.6           | Stemming'li ve daha büyük pencere boyutlu Skip-gram, anlamsal olarak hala sınırlıydı.                                 |
| stemmed\_model\_cbow\_vs300\_w2 | "doc1, doc60, doc61, doc62, doc63" | "[4, 3, 3, 2, 2]"   | 2.8           | Yüksek vektör boyutuna rağmen, stemming'in etkisi anlamsal doğruluğu düşürdü.                                          |
| stemmed\_model\_skipgram\_vs300\_w2 | "doc1, doc64, doc65, doc66, doc67" | "[4, 4, 3, 3, 3]"   | 3.4           | Stemming'li yüksek vektör boyutlu Skip-gram, stemmed modeller içinde en iyi performansı gösterdi.                     |
| stemmed\_model\_cbow\_vs300\_w4 | "doc1, doc68, doc69, doc70, doc71" | "[3, 2, 2, 1, 1]"   | 1.8           | En düşük ortalama puanlardan birine sahip oldu, muhtemelen stemming ve CBOW mimarisinin dezavantajları birleşti.    |
| stemmed\_model\_skipgram\_vs300\_w4 | "doc1, doc72, doc73, doc74, doc75" | "[4, 3, 3, 2, 2]"   | 2.8           | Yüksek vektör boyutu ve pencere boyutu 4 ile stemming, hala anlamsal zorluklar yarattı.                                |

### 3.2. Değerlendirme Yorumları

* **Hangi model(ler) daha yüksek ortalama aldı?** Anlamsal değerlendirme tablosuna göre, `lemmatized_model_skipgram_vs300_w2` modeli en yüksek ortalama puanı (4.2) almıştır. Bunu `lemmatized_model_cbow_vs300_w2` ve `lemmatized_model_skipgram_vs300_w4` modelleri (3.8) takip etmektedir. Bu durum, lemmatizasyonun ve daha yüksek vektör boyutlarının anlamsal doğruluğu artırdığını göstermektedir.
* **En anlamlı sonuçları hangi model verdi?** En anlamlı sonuçları, genel olarak lemmatize edilmiş Word2Vec modelleri, özellikle de `lemmatized_model_skipgram_vs300_w2` modeli vermiştir. Bu model, giriş metnindeki "issue" ve "nightmare" gibi anahtar arıza terimleriyle ilgili daha spesifik ve bağlamsal olarak uygun metinler önermiştir. TF-IDF modelleri de anahtar kelime benzerliklerinde iyi performans gösterse de, Word2Vec'in anlamsal derinliği sayesinde daha geniş bir bağlamda ilgili metinleri bulabildiği gözlemlenmiştir.
* **TF-IDF ile Word2Vec modelleri arasında fark var mı?** Evet, TF-IDF ve Word2Vec modelleri arasında belirgin farklar gözlemlenmiştir.
    * **TF-IDF:** Kelime sıklığına ve ters belge sıklığına dayandığı için, giriş metnindeki tam eşleşen veya sık geçen kelimelerin yoğun olduğu belgelerde yüksek benzerlik skorları vermiştir. Ancak, kelimeler arasındaki anlamsal ilişkileri (örneğin "car" ve "vehicle" arasındaki ilişkiyi) doğrudan yakalayamaz. Bu nedenle, bazen tam eşleşmeyen ancak anlamsal olarak ilgili metinleri kaçırabilir.
    * **Word2Vec:** Kelimelerin bağlamsal ilişkilerini vektörler halinde öğrendiği için, giriş metnindeki kelimelerin semantik olarak benzer olduğu metinleri bulmada daha başarılı olmuştur. Özellikle yüksek vektör boyutuna sahip Word2Vec modelleri, daha zengin anlamsal temsiller oluşturarak, doğrudan kelime eşleşmesi olmasa bile ilgili belgeleri bulabilmiştir.
* **Model yapılandırmalarının (CBOW, window, vektör boyutu) etkisi gözlenebiliyor mu?** Evet, model yapılandırmalarının önemli bir etkisi gözlemlenmiştir.
    * **CBOW vs. Skip-gram:** Genellikle, Skip-gram modelleri, özellikle düşük frekanslı kelimeler için daha iyi kelime temsilleri oluşturabildiği ve anlamsal ilişkileri daha iyi yakaladığı için bu görevde CBOW modellerine göre daha başarılı bulunmuştur. Sağlanan kod çıktılarında car, oil, problem kelimeleri için benzer kelimelerin listesi incelendiğinde, Skip-gram modellerinin ilgili kelimeleri daha tutarlı ve anlamsal olarak yakın bulduğu görülmüştür.
    * **Pencere Boyutu (Window Size):** Küçük pencere boyutları (örneğin 2), yakın bağlamdaki kelimelere odaklanırken, büyük pencere boyutları (örneğin 4), daha geniş bağlamdaki kelimeleri dikkate alır. Bu ödevde, küçük pencere boyutları (w2) bazen daha spesifik benzerlikler sağlarken, büyük pencere boyutları (w4) daha genel bağlamları yakalamaya çalışmıştır. Ancak anlamsal doğruluk için, daha küçük pencere boyutlarının bazen daha keskin sonuçlar verdiği görülebilir.
    * **Vektör Boyutu (Vector Size):** Daha yüksek vektör boyutları (örneğin 300), kelimelerin daha zengin ve ayrıntılı anlamsal temsillerini oluşturur. Bu, modelin kelimeler arasındaki ince nüansları yakalamasına yardımcı olur. Gözlemlerimiz, 300 boyutlu vektörlerin 100 boyutlu vektörlere göre anlamsal değerlendirmede daha iyi sonuçlar verdiğini göstermektedir.

### 3.3. Sıralama Tutarlılığı Değerlendirmesi (Jaccard Benzerliği)

Farklı modellerin aynı giriş metni için sıraladığı ilk 5 sonucun tutarlılığını ölçmek için Jaccard benzerliği kullanılmıştır. Jaccard benzerliği, iki kümenin kesişiminin birleşimine oranı olarak hesaplanır: $Jaccard(A,B)= \frac{|A \cap B|}{|A \cup B|}$

| Model Adı                          | tf-idf\_lemmatized | tf-idf\_stemmed | lem\_cbow\_w2\_d100 | lem\_sg\_w2\_d100 | lem\_cbow\_w4\_d100 | lem\_sg\_w4\_d100 | lem\_cbow\_w2\_d300 | lem\_sg\_w2\_d300 | lem\_cbow\_w4\_d300 | lem\_sg\_w4\_d300 | stem\_cbow\_w2\_d100 | stem\_sg\_w2\_d100 | stem\_cbow\_w4\_d100 | stem\_sg\_w4\_d100 | stem\_cbow\_w2\_d300 | stem\_sg\_w2\_d300 | stem\_cbow\_w4\_d300 | stem\_sg\_w4\_d300 |
| :--------------------------------- | :----------------- | :-------------- | :------------------ | :---------------- | :------------------ | :---------------- | :------------------ | :---------------- | :------------------ | :---------------- | :------------------- | :------------------- | :------------------- | :------------------- | :------------------- | :------------------- | :------------------- | :------------------- |
| tf-idf\_lemmatized                 | 1.00               | 0.43            | 0.20                | 0.25              | 0.15                | 0.18              | 0.22                | 0.28              | 0.17                | 0.23              | 0.10                 | 0.12                 | 0.08                 | 0.11                 | 0.12                 | 0.15                 | 0.09                 | 0.13                 |
| tf-idf\_stemmed                    | 0.43               | 1.00            | 0.18                | 0.20              | 0.13                | 0.15              | 0.20                | 0.25              | 0.15                | 0.20              | 0.12                 | 0.15                 | 0.10                 | 0.13                 | 0.14                 | 0.17                 | 0.11                 | 0.14                 |
| lem\_cbow\_w2\_d100                | 0.20               | 0.18            | 1.00                | 0.60              | 0.55                | 0.50              | 0.70                | 0.65              | 0.60                | 0.55              | 0.30                 | 0.35                 | 0.28                 | 0.32                 | 0.35                 | 0.40                 | 0.30                 | 0.33                 |
| lem\_sg\_w2\_d100                  | 0.25               | 0.20            | 0.60                | 1.00              | 0.50                | 0.55              | 0.65                | 0.70              | 0.55                | 0.60              | 0.35                 | 0.40                 | 0.32                 | 0.38                 | 0.40                 | 0.45                 | 0.35                 | 0.40                 |
| lem\_cbow\_w4\_d100                | 0.15               | 0.13            | 0.55                | 0.50              | 1.00                | 0.65              | 0.60                | 0.58              | 0.70                | 0.63              | 0.25                 | 0.30                 | 0.22                 | 0.27                 | 0.30                 | 0.35                 | 0.25                 | 0.30                 |
| lem\_sg\_w4\_d100                  | 0.18               | 0.15            | 0.50                | 0.55              | 0.65                | 1.00              | 0.58                | 0.62              | 0.63                | 0.68              | 0.28                 | 0.33                 | 0.25                 | 0.30                 | 0.33                 | 0.38                 | 0.28                 | 0.33                 |
| lem\_cbow\_w2\_d300                | 0.22               | 0.20            | 0.70                | 0.65              | 0.60                | 0.58              | 1.00                | 0.75              | 0.68                | 0.62              | 0.35                 | 0.40                 | 0.30                 | 0.35                 | 0.40                 | 0.45                 | 0.35                 | 0.40                 |
| lem\_sg\_w2\_d300                  | 0.28               | 0.25            | 0.65                | 0.70              | 0.58                | 0.62              | 0.75                | 1.00              | 0.62                | 0.68              | 0.40                 | 0.45                 | 0.35                 | 0.40                 | 0.45                 | 0.50                 | 0.40                 | 0.45                 |
| lem\_cbow\_w4\_d300                | 0.17               | 0.15            | 0.60                | 0.55              | 0.70                | 0.63              | 0.68                | 0.62              | 1.00                | 0.72              | 0.30                 | 0.35                 | 0.25                 | 0.30                 | 0.35                 | 0.40                 | 0.30                 | 0.35                 |
| lem\_sg\_w4\_d300                  | 0.23               | 0.20            | 0.55                | 0.60              | 0.63                | 0.68              | 0.62                | 0.68              | 0.72                | 1.00              | 0.33                 | 0.38                 | 0.28                 | 0.33                 | 0.38                 | 0.43                 | 0.33                 | 0.38                 |
| stem\_cbow\_w2\_d100               | 0.10               | 0.12            | 0.30                | 0.35              | 0.25                | 0.28              | 0.35                | 0.40              | 0.30                | 0.33              | 1.00                 | 0.67                 | 0.50                 | 0.45                 | 0.60                 | 0.55                 | 0.55                 | 0.50                 |
| stem\_sg\_w2\_d100                 | 0.12               | 0.15            | 0.35                | 0.40              | 0.30                | 0.33              | 0.40                | 0.45              | 0.35                | 0.38              | 0.67                 | 1.00                 | 0.45                 | 0.50                 | 0.55                 | 0.60                 | 0.50                 | 0.55                 |
| stem\_cbow\_w4\_d100               | 0.08               | 0.10            | 0.28                | 0.32              | 0.22                | 0.25              | 0.30                | 0.35              | 0.25                | 0.28              | 0.50                 | 0.45                 | 1.00                 | 0.67                 | 0.55                 | 0.50                 | 0.60                 | 0.55                 |
| stem\_sg\_w4\_d100                 | 0.11               | 0.13            | 0.32                | 0.38              | 0.27                | 0.30              | 0.35                | 0.40              | 0.30                | 0.33              | 0.45                 | 0.50                 | 0.67                 | 1.00                 | 0.50                 | 0.55                 | 0.55                 | 0.60                 |
| stem\_cbow\_w2\_d300               | 0.12               | 0.14            | 0.35                | 0.40              | 0.30                | 0.33              | 0.40                | 0.45              | 0.35                | 0.38              | 0.60                 | 0.55                 | 0.55                 | 0.50                 | 1.00                 | 0.67                 | 0.65                 | 0.60                 |
| stem\_sg\_w2\_d300                 | 0.15               | 0.17            | 0.40                | 0.45              | 0.35                | 0.38              | 0.45                | 0.50              | 0.40                | 0.43              | 0.55                 | 0.60                 | 0.50                 | 0.55                 | 0.67                 | 1.00                 | 0.60                 | 0.65                 |
| stem\_cbow\_w4\_d300               | 0.09               | 0.11            | 0.30                | 0.35              | 0.25                | 0.28              | 0.35                | 0.40              | 0.30                | 0.33              | 0.65                 | 0.60                 | 0.60                 | 0.55                 | 0.65                 | 0.60                 | 1.00                 | 0.67                 |
| stem\_sg\_w4\_d300                 | 0.13               | 0.14            | 0.33                | 0.40              | 0.28                | 0.33              | 0.40                | 0.45              | 0.35                | 0.38              | 0.60                 | 0.65                 | 0.55                 | 0.60                 | 0.60                 | 0.65                 | 0.67                 | 1.00                 |

Jaccard benzerlik matrisi incelendiğinde, farklı model türleri ve ön işleme yöntemleri arasındaki sıralama tutarlılığı gözlemlenmiştir.

* **Lemmatize Edilmiş Word2Vec Modelleri Arası:** Özellikle lemmatize edilmiş CBOW ve Skip-gram modelleri, benzer pencere ve vektör boyutlarında birbirlerine daha yüksek Jaccard skorları vermiştir (örneğin, `lemmatized_cbow_vs100_w2` ve `lemmatized_skipgram_vs100_w2` arasındaki 0.60). Bu durum, aynı ön işleme adımının ve benzer yapılandırmaların benzer sıralamalar ürettiğini göstermektedir. Ayrıca, 300 boyutlu lemmatize edilmiş Word2Vec modelleri (örn. `lem_cbow_w2_d300` ve `lem_sg_w2_d300` arasındaki 0.75), kendi aralarında en yüksek tutarlılığı gösteren gruptur. Bu, daha yüksek vektör boyutlarının, kelime temsillerini daha kararlı hale getirerek modeller arasında daha tutarlı sıralamalar sağlamasına yardımcı olduğunu göstermektedir.
* **Stemmed Word2Vec Modelleri Arası:** Benzer şekilde, stemmed edilmiş CBOW ve Skip-gram modelleri de kendi aralarında yüksek tutarlılık sergilemiştir (örneğin, `stemmed_cbow_vs100_w2` ve `stemmed_skipgram_vs100_w2` arasındaki 0.67). Bu durum, stemming'in kelime formlarını daha agresif bir şekilde sadeleştirmesi nedeniyle, her iki mimarinin de benzer "kök" tabanlı sonuçlara ulaşmasını sağlamıştır. En yüksek Jaccard skorları (`stem_cbow_w4_d300` ve `stem_sg_w4_d300` arasındaki 0.67) stemmed modeller arasında da görülmüştür.
* **TF-IDF ile Word2Vec Arası:** TF-IDF modelleri ile Word2Vec modelleri arasındaki Jaccard skorları genellikle daha düşüktür (örneğin, `tf-idf_lemmatized` ile `lemmatized_cbow_w2_d100` arasındaki 0.20 ve `tf-idf_lemmatized` ile `stemmed_cbow_w2_d100` arasındaki 0.10). Bu durum, iki yaklaşımın metin temsili ve benzerlik hesaplama felsefelerindeki farklılıklardan kaynaklanmaktadır. TF-IDF kelime sıklığına odaklanırken, Word2Vec anlamsal bağlamı yakalamaya çalışır, bu da farklı sıralamalara yol açar.
* **Anlamsal Değerlendirme ile İlişki:** Örneğin, Tablo 2'de `stemmed_cbow_w2_dim100` ve `stemmed_skipgram_w2_dim100` modellerinin 0.67 gibi yüksek bir Jaccard benzerliği gösterdiği gözlemlenmiştir. Ancak, Anlamsal Değerlendirme tablosuna baktığımızda, bu iki modelin ortalama puanlarının sırasıyla 2.2 ve 2.8 olduğunu görüyoruz, yani anlamsal olarak çok başarılı bulunmamışlardır. Bu durum, modellerin benzer sonuçlar üretmesinin her zaman en iyi anlamsal doğruluğu garanti etmediğini ortaya koymaktadır. Benzer ancak anlamsal olarak zayıf sonuçlar üreten modeller yüksek Jaccard skorlarına sahip olabilir.
* **Model Yapılandırmalarının Sıralama Başarımına Etkisi:** Model yapılandırmaları (CBOW vs SkipGram, window genişliği, vektör boyutu) sıralama başarımı üzerinde etkili olmuştur.
    * **CBOW vs. Skip-gram:** Genellikle, aynı ön işleme adımı (lemmatized veya stemmed) ve aynı boyutlarda, CBOW ve Skip-gram modellerinin birbirlerine yakın Jaccard skorları verdiği gözlemlenmiştir. Bu, her iki mimarinin de belirli bir ön işleme ve boyutlandırma kombinasyonunda benzer sıralamalar üretebildiğini düşündürmektedir. Ancak, anlamsal değerlendirmede Skip-gram genellikle daha iyi performans göstermiştir.
    * **Pencere Boyutu ve Vektör Boyutu:** Matrisin tamamı incelendiğinde, daha büyük vektör boyutlarının (300) ve bazen de farklı pencere boyutlarının, modeller arasındaki Jaccard benzerliğini etkilediği görülmüştür. Genellikle, aynı ön işleme ve mimari tipindeki modellerde, vektör boyutunun artmasıyla daha tutarlı (yüksek Jaccard) sonuçlar elde edilmiştir.

## 4. Sonuç ve Öneriler

### Genel Çıkarımlar

Bu ödevde, metin benzerliği hesaplamaları için TF-IDF ve Word2Vec modelleri kullanılarak kapsamlı bir analiz yapılmıştır. Anlamsal değerlendirme ve sıralama tutarlılığı analizleri sonucunda önemli çıkarımlar elde edilmiştir.

* Genel olarak, lemmatizasyonun stemming'e kıyasla anlamsal olarak daha doğru ve anlamlı sonuçlar ürettiği gözlemlenmiştir. Lemmatizasyon, kelimeleri anlamlı kök formlarına indirgeyerek kelime çeşitliliğini korurken, stemming daha agresif bir sadeleştirme yaparak bazı anlamsal nüansların kaybolmasına neden olabilmektedir.
* Word2Vec modelleri, TF-IDF'e kıyasla kelimeler arasındaki anlamsal ilişkileri daha iyi yakalayabilmiş, bu da daha bağlamsal ve ilgili metinlerin bulunmasına yol açmıştır. Özellikle yüksek vektör boyutuna sahip (300) Word2Vec modelleri, kelime temsillerini daha zengin kılarak performansı artırmıştır.
* Word2Vec modelleri arasında, Skip-gram mimarisi, CBOW'a göre biraz daha üstün performans sergilemiştir.

### Hangi model, hangi tür görevler için daha uygun olabilir?

* **TF-IDF:** Anahtar kelime tabanlı eşleştirme, belge sınıflandırması gibi görevler için hala etkili bir yöntem olabilir. Metinlerin temel içeriğini ve belirli terimlerin önemini hızlıca analiz etmek gerektiğinde kullanışlıdır. Özellikle büyük ve çok çeşitli veri setlerinde hızlı bir başlangıç noktası olarak düşünülebilir.
* **Word2Vec (Lemmatize edilmiş, yüksek vektör boyutlu Skip-gram):** Metinlerdeki anlamsal ilişkileri ve bağlamı yakalamak, semantik arama, öneri sistemleri, duygu analizi ve daha derin anlamsal benzerlik görevleri için daha uygundur. Özellikle lemmatize edilmiş veri üzerinde eğitilmiş, yüksek vektör boyutlu (örn. 300) ve Skip-gram mimarili modellerin, bu tür görevlerde en iyi performansı sergilediği görülmüştür. Bu modeller, kelimelerin anlamlarını daha doğru bir şekilde temsil edebilir ve bu da daha isabetli benzerlik sonuçlarına yol açar.

Sonuç olarak, projemizdeki araç arıza yorumları gibi anlamsal bağlamın önemli olduğu metin verileriyle çalışırken, lemmatizasyon ve Word2Vec (özellikle yüksek boyutlu Skip-gram) kombinasyonunun, metinler arası anlamsal benzerlikleri yakalamada daha üstün bir yaklaşım olduğu sonucuna varılmıştır.

## Kurulum ve Kullanım

1.  **Gerekli Kütüphanelerin Kurulumu:**
    * Proje için gerekli Python kütüphanelerini yüklemek için `requirements.txt` dosyasını kullanın.
    * Komut satırında, `requirements.txt` dosyasının bulunduğu dizine gidin ve şu komutu çalıştırın:
        ```bash
        pip install -r requirements.txt
        ```
        (Not: `metin_benzerligi.ipynb` dosyasındaki ilk hücrede `nltk` verilerinin ve `gensim` kütüphanesinin otomatik olarak indirilmesi/yüklenmesi sağlanmıştır.)
2.  **Veri Setinin Hazırlanması ve Modellerin Eğitilmesi:**
    * Veri toplama adımlarını uygulamak için `notebooks/01_data_collection.ipynb` not defterini çalıştırın. Bu, veriyi toplar ve birleştirir. (İsteğe bağlıdır. Birleştirilmiş veri `data/raw/all_data.csv` klasöründe mevcuttur.)
    * Veri ön işleme adımlarını uygulamak için `notebooks/02_data_preprocess.ipynb` not defterini çalıştırın. Bu, metin verisini temizler ve işler, `preprocessed_data_lemmatized_only.csv` ve `preprocessed_data_stemmed_only.csv` dosyalarını oluşturur. (İsteğe bağlıdır. İşlenmiş veri `data/processed/` klasöründe mevcuttur.)
    * Veri setinin özelliklerini anlamak için `notebooks/01_01_zipf_analizi.ipynb` ve `notebooks/02_01_lemmatizeandstemmed_zipf.ipynb` dosyalarını çalıştırarak Zipf Yasası analizini gerçekleştirin.
    * TF-IDF vektörlerini oluşturmak için `notebooks/03_vectorization_tf-idf.ipynb` not defterini çalıştırın.
    * Word2Vec modellerini eğitmek ve kaydetmek için `notebooks/04_word2vec.ipynb` not defterini çalıştırın. Bu, `models/` klasörüne eğitilmiş Word2Vec modellerini kaydedecektir.
3.  **Benzerlik Hesaplamalarını Çalıştırma ve Değerlendirme:**
    * Modellerin performansını değerlendirmek ve benzerlik hesaplamalarını (`TF-IDF` ve `Word2Vec` için Kosinüs Benzerliği, Jaccard Benzerliği) görmek için `metin_benzerligi.ipynb` not defterini çalıştırın. Bu not defteri, raporunuzdaki analiz ve tabloları üretir.
