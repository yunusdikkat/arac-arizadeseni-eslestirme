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
