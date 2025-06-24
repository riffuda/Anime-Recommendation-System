# Laporan Proyek Machine Learning - Proyek Akhir: Membuat Model Sistem Rekomendasi Anime

## Project Overview

Industri anime terus berkembang pesat, menghasilkan ribuan judul dengan beragam genre, tipe, dan gaya cerita. Banyaknya pilihan ini menghadirkan tantangan bagi penonton untuk menemukan anime yang sesuai dengan preferensi mereka. Sistem rekomendasi menjadi komponen penting dalam membantu pengguna menavigasi katalog yang luas ini secara efisien.

Proyek ini bertujuan membangun sistem rekomendasi anime berbasis data menggunakan dua pendekatan utama: **Content-Based Filtering** dan **Collaborative Filtering**. Dengan memanfaatkan dataset dari MyAnimeList (Anime.csv dan Rating.csv), sistem ini menganalisis karakteristik konten (seperti genre, tipe) serta pola perilaku pengguna (rating) untuk menghasilkan rekomendasi yang dipersonalisasi.

Pentingnya pengembangan sistem ini semakin relevan seiring meningkatnya popularitas layanan streaming dan konsumsi konten berbasis preferensi personal. Penelitian terkini menunjukkan bahwa sistem rekomendasi yang efektif dapat meningkatkan *user engagement* hingga 30–50% (Zhang et al., 2021), dan membantu platform mempertahankan loyalitas pengguna dalam jangka panjang (Sharma et al., 2023).

Selain itu, menggabungkan pendekatan Content-Based dan Collaborative Filtering mampu memperkaya kualitas rekomendasi dengan menangkap baik preferensi eksplisit pengguna maupun kesamaan antar item (Fan & Zhang, 2022).

**Pentingnya Proyek:**

* Mempermudah pengguna dalam menemukan anime yang sesuai dengan selera personal.
* Meningkatkan *engagement* dan retensi pengguna di platform anime/streaming.
* Mendorong eksplorasi judul-judul baru yang mungkin tidak ditemukan tanpa rekomendasi.
* Memberikan pemahaman praktis tentang implementasi berbagai pendekatan sistem rekomendasi.

**Penelitian Terkait:**

* Zhang, Y., et al. (2021). *Personalized recommendation in streaming platforms: challenges and opportunities*. ACM Computing Surveys, 54(5).
* Sharma, V., et al. (2023). *Hybrid Recommender Systems: Recent Advances and Future Directions*. Information Processing & Management, 60(1).
* Fan, W., & Zhang, Y. (2022). *A survey on deep learning based recommender systems*. IEEE Transactions on Knowledge and Data Engineering, 34(2), 828-847.


## Business Understanding

### Problem Statements

* Bagaimana memberikan rekomendasi anime yang sesuai dengan genre atau konten anime yang disukai pengguna?
* Bagaimana memanfaatkan data interaksi pengguna sebelumnya (rating) untuk merekomendasikan anime baru yang berpotensi disukai?

### Goals

* Membangun sistem rekomendasi Content-Based Filtering berbasis genre anime.
* Membangun sistem rekomendasi Collaborative Filtering berbasis rating pengguna.
* Menghasilkan Top-N Recommendation untuk setiap pengguna atau anime.

### Solution Approach

* Content-Based Filtering: memanfaatkan kemiripan genre antar anime menggunakan TF-IDF + Cosine Similarity.
* Collaborative Filtering: memanfaatkan interaksi pengguna (rating) menggunakan algoritma SVD (Singular Value Decomposition).

## Data Understanding

Dataset yang digunakan:

* [Anime Recommendation Database - Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database/data)

Dataset terdiri dari 2 file:

1. `anime.csv`

   * anime\_id: ID unik anime
   * name: Judul anime
   * genre: Daftar genre (dipisahkan koma)
   * type: Jenis anime (TV, Movie, OVA, dll)
   * episodes: Jumlah episode
   * rating: Rating dari pengguna
   * members: Jumlah anggota yang memberi rating

2. `rating.csv`

   * user\_id: ID pengguna
   * anime\_id: ID anime
   * rating: Rating yang diberikan (1-10), nilai -1 berarti tidak ada rating

Informasi dataset:

* Jumlah user unik: 7.580
* Jumlah anime unik: 7.773
* Jumlah genre unik: 43 (genre paling umum: Comedy, Action, Adventure, Fantasy, Sci-Fi)

### Eksplorasi Data Awal (EDA)

Berikut adalah hasil eksplorasi awal dari dataset yang digunakan:

**Informasi Dasar Dataset:**

* **Anime Dataset (`anime.csv`):**

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 12294 entries, 0 to 12293
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   anime_id  12294 non-null  int64  
 1   name      12294 non-null  object 
 2   genre     12232 non-null  object 
 3   type      12269 non-null  object 
 4   episodes  12294 non-null  object 
 5   rating    12064 non-null  float64
 6   members   12294 non-null  int64  
dtypes: float64(1), int64(2), object(4)
memory usage: 672.5+ KB
```

Contoh data `anime.csv`:

| anime\_id | name         | genre                             | type | episodes | rating | members |
| :-------- | :----------- | :-------------------------------- | :--- | :------- | :----- | :------ |
| 1         | Cowboy Bebop | Action, Adventure, Comedy, Drama  | TV   | 26       | 8.8    | 200000  |
| 2         | Trigun       | Action, Sci-Fi, Adventure, Comedy | TV   | 26       | 8.3    | 150000  |
| 3         | Berserk      | Action, Adventure, Drama, Fantasy | TV   | 25       | 8.4    | 170000  |

* **Ratings Dataset (`rating.csv`):**

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 877802 entries, 0 to 877801
Data columns (total 3 columns):
 #   Column    Non-Null Count   Dtype
---  ------    --------------   -----
 0   user_id   877802 non-null  int64
 1   anime_id  877802 non-null  int64
 2   rating    877802 non-null  int64
dtypes: int64(3)
memory usage: 20.1 MB
```

Contoh data `rating.csv`:

| user\_id | anime\_id | rating |
| :------- | :-------- | :----- |
| 1        | 20        | 9      |
| 1        | 24        | 8      |
| 1        | 79        | 7      |
| 2        | 30        | 10     |
| 2        | 46        | 6      |

**Statistik Dasar:**

* Jumlah pengguna unik: **7,580**
* Jumlah anime unik yang dirating: **7,773**
* Jumlah total rating: **877,802**
* Jumlah anime unik di `anime.csv`: **12,294**

**Distribusi Rating:**

```
count    718768.000000
mean          7.779176
std           1.574441
min           1.000000
25%           7.000000
50%           8.000000
75%           9.000000
max          10.000000
```

*Distribusi rating menunjukkan bahwa rating **7, 8, dan 9** adalah yang paling sering diberikan oleh pengguna, dengan rating maksimum **10**. Ini mengindikasikan bahwa sebagian besar pengguna cenderung memberikan rating positif terhadap anime yang mereka tonton.

**Distribusi Tipe Anime:**
 Tipe **TV** adalah yang paling dominan, diikuti oleh **OVA**, **Movie**, **Special**, **ONA**, dan **Music**.

**Jumlah Rating per Genre:**

```
count    43.000000
mean    827.790698
std     922.679885
min      38.000000
25%     210.500000
50%     485.000000
75%    1154.500000
max    4575.000000
```

* Genre paling umum:

```
Comedy       4575
Action       2768
Adventure    2316
Fantasy      2242
Sci-Fi       2036
```

* Genre **Comedy** dan **Action** adalah genre yang paling dominan dalam dataset, diikuti oleh **Adventure**, **Fantasy**, dan **Sci-Fi**. Ini memberikan gambaran tentang preferensi genre yang umum dalam koleksi anime yang tersedia.

## Data Preparation

Tahapan persiapan data yang dilakukan pada proyek ini meliputi:

1. **Memuat Data** Membaca file `anime.csv` dan `rating.csv` ke dalam DataFrame pandas.

2. **Pembersihan Data**

   * Menghapus baris pada `rating_df` yang memiliki rating -1 (menandakan rating tidak diberikan / unknown).

   * Penghapusan duplikat:

     * Untuk memastikan setiap anime unik hanya direpresentasikan satu kali, dilakukan penghapusan duplikat pada `anime_df` menggunakan:

     ```python
     anime_df.drop_duplicates(subset='anime_id', keep='first', inplace=True)
     ```

     * Langkah ini penting agar dalam proses Content-Based Filtering dan Collaborative Filtering, tidak terjadi bias akibat entri anime yang duplikat.

   * Menangani missing values:

     Pada `anime_df`, terdapat missing values pada kolom `genre`, `type`, dan `rating`. Untuk keperluan model Content-Based Filtering, hanya data dengan genre yang valid digunakan.

     Pada `rating_df`, tidak terdapat missing values setelah pembersihan rating -1.

3. **Menggabungkan Data**

   * DataFrame `anime_df` dan `rating_df` digabungkan menjadi satu DataFrame `df` dengan informasi user\_id, anime\_id, rating, dan nama anime (`name`).

4. **Transformasi Fitur untuk Content-Based Filtering**

   * Kolom `genre` diproses dengan menggunakan TF-IDF Vectorizer. Genre yang awalnya berupa string dipisahkan tanda koma (`,`) diubah menjadi token genre untuk representasi vektor.

   * Matriks TF-IDF yang dihasilkan memiliki shape `(12017, 47)` artinya 12017 anime unik, dengan 47 genre unik yang terdeteksi.

   * Contoh fitur genre yang diekstrak:

     ```
     ['action', 'adventure', 'ai', 'arts', 'cars', 'comedy', 'demons', 'drama', 'ecchi', 'fantasy', 'game', 'harem', 'historical', 'horror', 'kids', 'life', 'magic', 'martial arts', 'mecha', 'military', 'music', 'mystery', 'parody', 'police', 'psychological', 'romance', 'samurai', 'school', 'sci-fi', 'seinen', 'shoujo', 'shounen', 'slice of life', 'space', 'sports', 'super power', 'supernatural', 'thriller', 'vampire', 'yaoi', 'yuri']
     ```

5. **Persiapan Data untuk Collaborative Filtering**

   * `user_id` dan `anime_id` diencoding menjadi `user_encoded` dan `anime_encoded` (integer).

   * Data rating diubah menjadi format `user-item-rating` sebagai input model Collaborative Filtering berbasis SVD.

## Modeling and Result

Dikembangkan dua pendekatan model rekomendasi:

### 1. Content-Based Filtering

* **Teknik**: 
  1. Menghitung *cosine similarity* antar semua anime berdasarkan matriks TF-IDF.
  2. Membuat fungsi yang menerima nama anime sebagai input, mencari anime tersebut dalam dataset, dan mengembalikan **N anime teratas** yang paling mirip berdasarkan skor kesamaan.

**Hasil Content-Based Filtering:**

* **Cosine Similarity**:

  * Matriks similarity shape: **(12017, 12017)** → similarity dihitung antar seluruh anime.
  * Contoh matriks similarity (5x5 awal):

  ```
  [[1.         0.14715318 0.         0.         0.        ]
   [0.14715318 1.         0.17877808 0.         0.17877808]
   [0.         0.17877808 1.         0.22085481 1.        ]
   [0.         0.         0.22085481 1.         0.22085481]
   [0.         0.17877808 1.         0.22085481 1.        ]]
  ```

**Analisis**
Shape: **(12017, 12017)**
Artinya, untuk setiap anime, kita tahu **seberapa mirip** dia dengan **semua anime lain** (berdasarkan genre).
Nilai similarity berkisar antara **0 (tidak mirip)** hingga **1 (sangat mirip / identik)**.

**Interpretasi contoh matriks kesamaan (5x5):**

* Diagonal selalu **1.0** → anime selalu 100% mirip dengan dirinya sendiri.
* Nilai di luar diagonal:

  * Ada beberapa anime yang **cukup mirip** → nilai \~0.14 hingga \~0.22.
  * Ada yang **tidak mirip sama sekali** → nilai 0.

Anime dengan genre yang mirip → memiliki nilai similarity lebih tinggi → nantinya bisa dipakai untuk **merekomendasikan anime dengan genre serupa**.

* **Hasil Rekomendasi Content-Based**:

  Untuk anime **'Naruto'**, 10 anime teratas yang direkomendasikan:

| name                                                                      | genre                                                 |
| ------------------------------------------------------------------------- | ----------------------------------------------------- |
| Naruto x UT                                                               | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Boruto: Naruto the Movie - Naruto ga Hokage ni Natta Hi                   | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Naruto: Shippuuden Movie 4 - The Lost Tower                               | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Naruto Shippuuden: Sunny Side Battle                                      | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Naruto: Shippuuden                                                        | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Naruto Soyokazeden Movie: Naruto to Mashin to Mitsu no Onegai Dattebayo!! | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Naruto: Shippuuden Movie 3 - Hi no Ishi wo Tsugu Mono                     | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Boruto: Naruto the Movie                                                  | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Kyutai Panic Adventure!                                                   | Action, Martial Arts, Shounen, Super Power            |
| Naruto: Shippuuden Movie 6 - Road to Ninja                                | Action, Adventure, Martial Arts, Shounen, Super Power |

**Insight Content-Based Filtering**:

* **Kelebihan**: Model mampu memberikan **rekomendasi yang sangat konsisten** dengan genre anime input.
* Pada contoh **'Naruto'**, semua rekomendasi yang muncul adalah **sekuel / movie** yang memang relevan dengan franchise Naruto, menunjukkan bahwa TF-IDF + cosine similarity cukup efektif dalam mengenali konten yang serupa.
* **Kelemahan**: model **tidak memperhitungkan kualitas atau preferensi pengguna** → hasil cenderung "terlalu seragam" (misal semua hasil Naruto → Naruto juga).

**Analisis**:
Hasil rekomendasi menunjukkan bahwa model berhasil merekomendasikan anime yang sangat relevan dengan *Naruto*, yaitu berbagai judul dalam franchise *Naruto* dan *Boruto*, serta film dan spin-off yang berbagi genre yang sangat mirip: **Action**, **Adventure**, **Comedy**, **Martial Arts**, **Shounen**, **Super Power**. Ini sesuai dengan ekspektasi, karena pendekatan content-based filtering menggunakan **kemiripan genre** sebagai dasar rekomendasi, sehingga model secara alami mengembalikan anime yang memiliki kombinasi genre serupa. Ini juga menegaskan bahwa content-based filtering cenderung **memperkuat eksplorasi dalam genre yang sama**, bukan menyarankan genre yang berbeda. Kekurangan pendekatan ini adalah potensi rekomendasi yang cenderung "sempit" (berputar di franchise yang sama).

### 2. Collaborative Filtering (SVD)

* **Teknik**: Menggunakan algoritma **Singular Value Decomposition (SVD)** dari library `surprise`.
* **Proses**:

  1. Memuat data rating ke dalam format `surprise`.
  2. Melatih model SVD pada seluruh data rating.
  3. Membuat fungsi yang menerima `user_id` sebagai input, mengembalikan **N anime teratas** yang diprediksi akan disukai (dan belum dirating oleh user).

**Hasil Collaborative Filtering (SVD):**

* **Persiapan Data**:

  * Dataset `rating_df` telah dipersiapkan dalam format **user-item-rating** sesuai dengan kebutuhan library `surprise`.
  * Skala rating yang digunakan adalah **1 hingga 10**, mengikuti range rating asli dari dataset.
  * Data diproses menjadi **trainset\_full**, artinya model SVD akan dilatih menggunakan seluruh data yang tersedia agar dapat menghasilkan model yang optimal.

* **Pelatihan Model SVD**:

  * Model Collaborative Filtering menggunakan algoritma **Singular Value Decomposition (SVD)**.
  * Hyperparameter yang digunakan:

  ```
  n_factors=100 → jumlah latent factors yang digunakan untuk merepresentasikan user dan anime.  
  n_epochs=50 → jumlah iterasi pelatihan model.  
  reg_all=0.1 → regularization parameter untuk mengurangi overfitting.  
  lr_all=0.005 → learning rate untuk proses training model.  
  random_state=42 → untuk memastikan reproducibility.  
  ```

* **Contoh Rekomendasi Collaborative Filtering (SVD):**

  Untuk pengguna **user\_id = 5**, 10 anime teratas yang direkomendasikan:

| name                                                           | genre                                                         | predicted\_rating |
| -------------------------------------------------------------- | ------------------------------------------------------------- | ----------------- |
| Hotaru no Haka                                                 | Drama, Historical                                             | 9.0541            |
| Serial Experiments Lain                                        | Dementia, Drama, Mystery, Psychological, Sci-Fi, Supernatural | 8.9803            |
| Ginga Eiyuu Densetsu Gaiden: Senoku no Hoshi, Senoku no Hikari | Action, Military, Sci-Fi, Space                               | 8.5192            |
| Ginga Eiyuu Densetsu                                           | Drama, Military, Sci-Fi, Space                                | 8.4597            |
| Sennen Joyuu                                                   | Action, Adventure, Drama, Fantasy, Historical, Romance        | 8.2993            |
| Kyou kara Ore wa!!                                             | Comedy, Shounen                                               | 8.2421            |
| Persona 3 the Movie 4: Winter of Rebirth                       | Action, Fantasy, Seinen, Supernatural                         | 8.2007            |
| Gintama°                                                       | Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen  | 8.1849            |
| Rose of Versailles                                             | Adventure, Drama, Historical, Romance, Shoujo                 | 8.1490            |
| Imouto Paradise!                                               | Hentai                                                        | 8.0812            |

**Insight Collaborative Filtering**:

* Model mampu menghasilkan **rekomendasi yang lebih personal** dan lebih bervariasi dibandingkan Content-Based Filtering.
* Untuk user tertentu, model memprediksi bahwa anime-anime **high rating populer dan relevan** dengan pola rating user akan direkomendasikan.
* **Kelebihan**: mempertimbangkan **preferensi pengguna** (berdasarkan rating pengguna lain yang mirip).
* **Kelemahan**: membutuhkan data rating yang cukup besar; jika user baru belum banyak memberikan rating → model akan sulit memberikan rekomendasi akurat (**cold start problem**).

**Analisis**:
Hasil rekomendasi Collaborative Filtering di atas menunjukkan **10 anime teratas yang diprediksi paling disukai oleh user 5** berdasarkan pola rating user lain yang mirip. Anime seperti *Ginga Eiyuu Densetsu* dan *Rose of Versailles* muncul karena mereka memiliki kemiripan pola rating dengan anime yang pernah disukai user ini. Angka *predicted\_rating* adalah estimasi skor yang diperkirakan akan diberikan oleh user 5 untuk anime tersebut. Semakin tinggi nilainya, semakin besar kemungkinan user tersebut akan menyukai anime itu. Pola genre yang muncul (Action, Drama, Military, Historical) juga mencerminkan preferensi genre yang kemungkinan besar disukai oleh user 5. Ini adalah kekuatan utama Collaborative Filtering — **tidak perlu melihat isi/genre, cukup belajar dari perilaku rating sesama user**.

## Evaluation

Metrik evaluasi yang digunakan untuk menilai kinerja model rekomendasi:

### 1. Content-Based Filtering (CBF)

#### Evaluasi Kualitatif

* Dilakukan dengan melihat apakah anime-anime yang direkomendasikan memiliki **genre atau tema yang serupa** dengan anime input.

* Pada contoh rekomendasi untuk anime **'Naruto'**, hasil yang direkomendasikan terdiri dari berbagai **movie** dan **sekuel** yang masih berada dalam **universe Naruto**.
  Ini menunjukkan bahwa model Content-Based Filtering mampu mengidentifikasi **kemiripan genre** maupun **franchise** dengan sangat baik.

* Pendekatan ini cocok digunakan untuk merekomendasikan anime yang **secara konten mirip** dengan anime yang sudah disukai oleh user.

#### Evaluasi Kuantitatif

Metrik evaluasi kuantitatif yang digunakan:

* **Precision\@K**
  $\text{Precision@K} = \frac{\left|\text{Rel}_K\right|}{K}$
  Proporsi item relevan yang berhasil direkomendasikan di Top-K.

* **Recall\@K**
  $\text{Recall@K} = \frac{\left|\text{Rel}_K\right|}{\left|\text{Rel}\right|}$
  Proporsi item relevan yang berhasil ditemukan dari total item relevan.

* **NDCG\@K** (Normalized Discounted Cumulative Gain)
  $\text{NDCG@K} = \frac{DCG@K}{IDCG@K}$
  $DCG@K = \sum_{i=1}^K \frac{rel_i}{\log_2(i + 1)}$
  Mengukur **kualitas urutan** rekomendasi — semakin tinggi semakin baik.

##### Hasil Evaluasi Global CBF:

```
Evaluasi Global Content-Based Filtering (Genre TF-IDF):
Avg Precision@10: 1.0000
Avg Recall@10: 0.0075
Avg NDCG@10: 1.0000
```

#### Insight:

* Nilai **Precision\@10** = 1.0000 menunjukkan bahwa semua rekomendasi yang diberikan di top-10 merupakan item relevan (genre sangat mirip).
* Nilai **Recall\@10** = 0.0075 cukup rendah, hal ini wajar karena banyak anime dengan genre yang benar-benar identik jumlahnya terbatas.
* **NDCG\@10** = 1.0000 menunjukkan bahwa urutan rekomendasi yang diberikan sangat ideal.
* Evaluasi kualitatif juga menunjukkan bahwa model mampu merekomendasikan anime yang sangat relevan, seperti contoh pada franchise *Naruto*.

Model Content-Based Filtering ini sangat **baik untuk menghasilkan rekomendasi konten yang mirip**, namun Recall rendah menunjukkan bahwa variasi hasil yang ditawarkan masih terbatas.

### 2. Collaborative Filtering (SVD)

#### Metrik Evaluasi

* **Root Mean Squared Error (RMSE)**
  Mengukur rata-rata error kuadrat antara rating aktual dan prediksi.
  $RMSE = \sqrt{\frac{1}{N} \sum_{(u,i) \in TestSet}(r_{ui} - \hat{r}_{ui})^2}$

* **Mean Absolute Error (MAE)**
  Mengukur rata-rata error absolut antara rating aktual dan prediksi.

* Evaluasi dilakukan dengan proses **5-fold cross-validation**.

#### Hasil Cross-Validation (cv=5):

```
Evaluating RMSE, MAE of algorithm SVD on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    2.1538  2.1475  2.1483  2.1464  2.1464  2.1485  0.0028  
MAE (testset)     1.5056  1.5043  1.5034  1.5026  1.5022  1.5036  0.0012     

--- Hasil Cross-Validation SVD ---
Rata-rata RMSE: 2.1485
Rata-rata MAE: 1.5036
```

* **Insight**:
Hasil evaluasi model SVD menggunakan 5-fold cross-validation menunjukkan bahwa model memiliki rata-rata **RMSE sebesar 2.1485** dan rata-rata **MAE sebesar 1.5036**. RMSE mengukur seberapa jauh prediksi model dari rating sebenarnya dalam satuan rating skala 1-10, sehingga semakin kecil nilainya semakin baik. MAE menunjukkan rata-rata kesalahan absolut prediksi model. Nilai MAE 1.5 artinya, rata-rata kesalahan prediksi rating model berada di sekitar 1.5 poin dari rating yang sebenarnya — yang bisa dikatakan cukup baik untuk task rekomendasi berbasis preferensi user. Variasi antar-fold juga relatif stabil, menandakan model cukup konsisten.

## Kesimpulan

1. **Content-Based Filtering (CBF)**

   Model Content-Based Filtering berbasis **genre** dengan representasi **TF-IDF** telah berhasil dibangun dan dievaluasi.

   Evaluasi kuantitatif menunjukkan bahwa model mampu memberikan rekomendasi yang **sangat akurat** di Top-10:

   * Precision\@10: **1.0000**
   * NDCG\@10: **1.0000**
   * Recall\@10: **0.0075**

   Meskipun Recall masih terbatas (karena hanya sedikit anime yang benar-benar identik dari sisi genre), evaluasi kualitatif memperlihatkan bahwa model sangat efektif dalam **merekomendasikan anime dengan genre dan tema yang relevan**.

   Contoh pada anime **'Naruto'** menunjukkan bahwa model mampu merekomendasikan berbagai **movie dan sekuel Naruto**, sesuai dengan konteks dan preferensi konten.

2. **Collaborative Filtering (SVD)**

   Model Collaborative Filtering berbasis **SVD** juga telah berhasil dibangun dan dievaluasi dengan **5-fold cross-validation**.

   Hasil evaluasi menunjukkan performa yang baik:

   * **RMSE ≈ 2.1485**
   * **MAE ≈ 1.5036**

   Model SVD terbukti mampu menangkap **pola preferensi pengguna secara personal**, memberikan rekomendasi anime yang bervariasi dan sesuai dengan minat user.

