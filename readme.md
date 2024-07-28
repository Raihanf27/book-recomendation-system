# Laporan Proyek Machine Learning - Raihan Fahlevi

## Project Overview

Pada era digital ini, volume informasi yang berlimpah menimbulkan tantangan baru dalam membantu pengguna menemukan informasi yang relevan. Dalam dunia literatur, khususnya, jumlah buku yang tersedia sangat banyak, sehingga membuat pengguna kesulitan menemukan buku yang sesuai dengan minat dan preferensinya. Sistem rekomendasi menjadi solusi penting untuk mengatasi tantangan ini, dengan memberikan rekomendasi yang tepat berdasarkan preferensi individu pengguna. Proyek ini bertujuan untuk mengembangkan sistem rekomendasi buku yang memanfaatkan teknik Machine Learning Content-Based Filtering untuk memberikan rekomendasi yang relevan dan personal.

## Business Understanding

### Problem Statements

1. Bagaimana membangun sistem rekomendasi buku yang efektif menggunakan data yang tersedia?
2. Bagaimana mengukur keakuratan dan relevansi rekomendasi yang diberikan oleh sistem?

### Goals

- Membangun model rekomendasi buku menggunakan algoritma content-based filtering.
- Mengevaluasi model rekomendasi dengan metrik yang sesuai untuk memastikan keakuratan dan relevansi.

### Solution statements

- Menggunakan teknik content-based filtering untuk merekomendasikan buku berdasarkan fitur teks.
- Mengimplementasikan algoritma TF-IDF untuk mengekstraksi fitur teks dari judul, penulis, dan penerbit buku.
- Menggunakan cosine similarity untuk mengukur kemiripan antara buku dan memberikan rekomendasi.

## Data Understanding

Dataset yang digunakan terdiri dari tiga file utama yakni:

- Books.CSV: Pada dataset ini terdapat 271.360 baris dan 8 kolom yang berisi informasi terkait buku.
- Ratings.CSV: Pada dataset ini terdapat 1.149.780 baris dan 3 kolom yang berisi informasi terkait rating pada buku.
- Users.CSV: Pada dataset ini terdapat 278.858 baris dan 3 kolom yang berisi terkait pengguna, lokasi dan pengguna.

Sumber data ini diperoleh dari kaggle yang dapat diakses melalu link berikut: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

### Informasi Variabel sebagai berikut:

- Books.csv:

![Books.info](https://github.com/user-attachments/assets/458e4bfa-4fa0-4c5b-80da-61e79bab2523)

`ISBN`: Identifikasi unik buku
`Book-Title`: Judul buku
`Book-Author`: Penulis buku
`Publisher`: Penerbit buku
`Year-Of-Publication`: Tahun publikasi
`Image-URL-S`: Link Image dengan ukuran Small
`Image-URL-M`: Link Image dengan ukuran Medium
`Image-URL-L`: Link Image dengan ukuran Large


- Ratings.csv:

![Ratings.info](https://github.com/user-attachments/assets/98788983-3bc3-438c-ac30-62087724ead1)

`User-ID`: Identifikasi unik pengguna
`ISBN`: Identifikasi unik buku
`Book-Rating`: Rating buku yang diberikan oleh pengguna


- Users.csv:

![Users.info](https://github.com/user-attachments/assets/30203097-f9e3-42f0-9bb9-8a1533e04058)

`User-ID`: Identifikasi unik pengguna
`Location`: Lokasi pengguna
`Age`: Usia pengguna

### Univariate Data Analysis

Disini saya akan memeriksa data yang memilik statistik deskriptif dari data Ratings(Book-Rating) dan Users(Age):

- DataFrame Ratings:

![image](https://github.com/user-attachments/assets/df264b9a-2ca3-4212-85e0-5e0db45bb432)

Bisa dilihat pada statistik diatas bahwa Book-Rating memiliki:

- Mean: Rata-rata rating buku adalah 2.8
- Std: Standar deviasi adalah 3.8
- Min: Rating buku terkecil adalah 0
- Kuartil pertama (Q1): 0
- Kuartil Kedua (median): 0
- Kuartil ketiga (Q3): 7
- Max: Rating Buku terbesar 10

- DataFrame Users:

![statistikusers](https://github.com/user-attachments/assets/24fffdf0-71f0-420f-8a6a-b37af71ec15c)

Bisa dilihat pada statistik diatas bahwa umur memiliki:

- Mean: Rata-rata umur user adalah 34
- Std: Standar deviasi adalah 14
- Min: Umur termuda adalah 0
- Kuartil pertama (Q1): 24
- Kuartil kedua (median): 32
- Kuartil ketiga (Q3): 44
- Max: Umur tertua sebesar 244

### Data Preprocessing

- Menggabungkan DataFrame ratings dan book bedasarkan kolom ISBN.

![merge1](https://github.com/user-attachments/assets/ebfc6475-1a2d-4a26-9b25-fd3d93311cba)

- Menggabungkan data yang digabungkan tadi dengan data Users bedasarkan kolom User-ID

![merge2](https://github.com/user-attachments/assets/e6381572-021b-4415-8f0f-30205f34f74e)

## Data Preparation

**1. Menghapus Kolom yang Tidak Relevan**

Setelah dataset digabungkan, beberapa kolom tidak relevan atau tidak berguna untuk modeling. Menghapus kolom seperti Image-URL-S, Image-URL-M, Image-URL-L, Year-Of-Publication, Location, dan Age membantu menyederhanakan dataset dan fokus pada informasi yang penting.

**2. Menangani Nilai Null**

Nilai null atau missing values dapat menyebabkan masalah dalam analisis dan modeling. Dalam kasus ini, kolom Publisher memiliki nilai null yang perlu ditangani. Menghapus baris dengan nilai null memastikan data yang digunakan lengkap dan tidak menyebabkan error dalam tahap selanjutnya.

**3. Memeriksa nilai Book-Rating**

Menghapus baris yang memiliki nilai Book-Rating kurang dari 0 penting untuk memastikan bahwa hanya rating yang valid dan relevan yang dipertimbangkan dalam analisis dan modeling. Rating 0 dianggap sebagai implicit feedback dan mungkin tidak memberikan informasi yang berguna untuk model rekomendasi yang dibangun.

**4. Memeriksa nilai unik kolom**

Memeriksa nilai unik pada kolom User-ID, ISBN, dan Book-Title membantu memahami distribusi data dan memastikan bahwa tidak ada duplikasi yang tidak diinginkan. Ini juga membantu dalam validasi bahwa data telah digabungkan dengan benar.

**5. Membuat Dataframe**

Membuat kolom Features yang merupakan gabungan dari Book-Title, Book-Author, dan Publisher penting untuk model content-based filtering. Kolom ini menyatukan informasi penting dari buku yang akan digunakan untuk menghitung kemiripan (similarity) antara buku.

**6. Membuat kolom baru**

Membuat kolom baru bernama Features di DataFrame preparation. Kolom ini adalah hasil penggabungan kolom Book-Title, Book-Author, dan Publisher yang dipisahkan oleh koma.

**7. Menghapus nilai NaN**

Menghapus baris dengan nilai NaN pada kolom Features memastikan bahwa semua data yang digunakan untuk perhitungan similarity lengkap dan tidak menyebabkan error saat perhitungan dilakukan.

**8. Membuat nilai menjadi kecil semua**

Mengonversi semua teks pada kolom Features menjadi huruf kecil memastikan konsistensi dalam perhitungan similarity. Ini menghindari perbedaan antara huruf besar dan kecil yang dapat mempengaruhi hasil perhitungan.

**9. Memeriksa nilai duplicate**

Memeriksa dan menghapus nilai duplikat pada kolom Book-Title penting untuk memastikan bahwa setiap buku hanya muncul sekali dalam dataset, menghindari bias dalam rekomendasi yang dihasilkan.

**10. Konversi kolom menjadi list**

Mengonversi kolom ISBN, Book-Title, dan Features menjadi daftar (list) memudahkan akses dan manipulasi data dalam tahap-tahap komputasi berikutnya, terutama saat membangun model dan memberikan rekomendasi.

**11. Membatasi jumlah data**

Mengambil 10.000 data pada kolom ISBN, Book-Title, dan Features membantu dalam memudahkan komputasi data.

**12. Membuat dictionary**

Membuat dictionary untuk menentukan pasangan key-value pada data ISBN, Book-Title, dan Features membantu dalam akses cepat dan efisien saat mencari informasi terkait buku tertentu, yang berguna dalam proses rekomendasi.

## Modeling

Proses modelling dalam proyek ini bertujuan untuk membangun sistem rekomendasi buku berdasarkan kemiripan fitur menggunakan model content-based filtering.

Content-based filtering bekerja dengan menganalisis fitur dari item yang akan direkomendasikan dan menghitung kemiripan antara item-item tersebut. Proses ini melibatkan ekstraksi fitur, pembentukan vektor fitur, perhitungan kemiripan, dan akhirnya memberikan rekomendasi berdasarkan item yang paling mirip. Metode ini sangat efektif ketika data tentang pengguna terbatas atau tidak tersedia, karena hanya bergantung pada karakteristik item itu sendiri.

Berikut adalah langkah-langkah detail dalam proses modelling:

### **1. Pengambilan Sampel data**

Langkah pertama adalah mengambil sampel acak dari data yang telah disiapkan untuk memastikan bahwa data tersebut siap digunakan untuk modelling.

```
data = book_new
data.sample(5)
```

### **2. Mengubah Teks Menjadi Representasi Numerik dengan TF-IDF.**

TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF adalah teknik yang digunakan untuk mengubah teks menjadi representasi numerik. Ini membantu dalam memahami seberapa penting suatu kata dalam dokumen tertentu relatif terhadap seluruh kumpulan dokumen.

**a. Insialisasi TfidfVectorizer**

Menginisialisasi TfidfVectorizer dan memfitkan pada kolom Book_Features.

```
tf = TfidfVectorizer()
tf.fit(data['Book_Features'])
tf.get_feature_names_out()
```

**b. Melakukan Transformasi teks**

Melakukan transformasi teks pada kolom Book_Features menggunakan TfidfVectorizer.

```
tfidf_matrix = tf.fit_transform(data['Book_Features'])
```

**c. Menghasilkan Vektor dalam bentuk matriks**

Menghasilkan vektor TF-IDF dalam bentuk matriks, menggunakan fungsi todense().

```
tfidf_matrix.todense()
```

**d. Melihat matriks TF-IDF**
Membuat DataFrame dari matriks TF-IDF, dengan kolom sebagai nama fitur (kata-kata) dan baris sebagai judul buku dari DataFrame data. Kemudian, akan menampilkan sampel acak dari 10 baris dan 22 kolom.

```
pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names_out(),
    index=data['Book_Title']
).sample(22, axis=1).sample(10, axis=0)
```

### 3. Menghitung cosine similarity
Cosine Similarity adalah metrik yang digunakan untuk mengukur kemiripan antara dua vektor dalam ruang multidimensi. Ini digunakan untuk mengukur seberapa mirip satu buku dengan buku lainnya berdasarkan fitur yang telah diekstraksi.

```
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=data['Book_Title'], columns=data['Book_Title'])
print('Shape:', cosine_sim_df.shape)
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)
```

### 4. Membangun Fungsi Rekomendasi

Fungsi book_recommendations dibuat untuk memberikan rekomendasi buku berdasarkan kemiripan dengan buku yang diberikan. Fungsi ini bekerja sebagai berikut:

  **Parameter Input:**

  - book_title: Judul buku yang digunakan sebagai referensi untuk mencari rekomendasi.

  - similarity_data: DataFrame yang berisi nilai kemiripan kosinus antara buku-buku.

  - items: DataFrame yang berisi informasi buku (ISBN, judul buku).

  - k: Jumlah rekomendasi yang diinginkan.

  **Proses:**

Dalam proses rekomendasi menggunakan model content-based filtering, langkah pertama adalah mengambil indeks dari k nilai kemiripan tertinggi berdasarkan perhitungan cosine similarity. Ini berarti kita mencari buku-buku yang memiliki kemiripan tertinggi dengan buku yang dijadikan referensi. Setelah itu, kita mendapatkan judul buku yang paling mirip dari indeks-indeks tersebut. Namun, agar hasil rekomendasi tetap relevan, buku yang dicari tidak dimasukkan dalam daftar rekomendasi. Langkah terakhir adalah menggabungkan informasi buku yang dihasilkan, lalu mengembalikan k rekomendasi teratas kepada pengguna. Proses ini memastikan bahwa rekomendasi yang diberikan adalah buku-buku yang paling mirip dan relevan dengan preferensi pengguna.

### 5. Menghasilkan Rekomendasi
Menggunakan fungsi book_recommendations untuk mendapatkan rekomendasi buku berdasarkan buku yang diberikan. Misalnya, untuk buku "Harry Potter and the Sorcerer's Stone (Book 1)". Dan memberikan rekomendasi buku yang diberikan.

![hasil rekomendasi](https://github.com/user-attachments/assets/6fafc629-f8d6-4354-8499-c0be2530ea40)

## Evaluation

Pada proyek ini, kami menggunakan dua metrik evaluasi utama untuk mengukur kinerja sistem rekomendasi, yaitu presisi dan recall.

**Precision (Presisi):**
Presisi mengukur proporsi item yang direkomendasikan yang benar-benar relevan. Nilai presisi tinggi menunjukkan bahwa sistem tidak sering merekomendasikan item yang tidak relevan.

![Presisi](https://github.com/user-attachments/assets/f6b30f84-9d1d-4846-b9a3-75606d2c150d)

​Cara Kerja:

- Menghitung TP dan FP: Tentukan jumlah prediksi yang benar-benar positif (TP) dan prediksi yang salah dianggap positif (FP).
- Menghitung Presisi: Bagi jumlah TP dengan total TP dan FP

**Recall:**
Recall mengukur proporsi item relevan yang berhasil direkomendasikan oleh sistem. Nilai recall tinggi menunjukkan bahwa sistem berhasil merekomendasikan sebagian besar item yang relevan.

![recall](https://github.com/user-attachments/assets/60472ca7-416a-49be-aed5-fb3286865d76)

Cara Kerja:

- Menghitung TP dan FN: Tentukan jumlah prediksi yang benar-benar positif (TP) dan sampel positif yang tidak terdeteksi (FN).
- Menghitung Recall: Bagi jumlah TP dengan total TP dan FN.

**Tahapan evaluasi:**
1. Definisi Ground Truth: Ground truth didefinisikan sebagai matriks cosine similarity itu sendiri (ground_truth = cosine_sim). Dalam konteks ini, ground truth adalah matriks yang menunjukkan kemiripan antar buku berdasarkan fitur yang ada.

2. Membuat DataFrame Ground Truth: Sebuah DataFrame (ground_truth_df) dibuat untuk menampilkan beberapa nilai dari matriks cosine similarity, guna melihat beberapa contoh nilai kemiripan antar buku.

3. Mengambil Sampel: Sebagian kecil dari matriks cosine similarity dan ground truth diambil untuk evaluasi (5000x5000 pertama dari matriks asli). Ini dilakukan untuk mengurangi beban komputasi.

4. Mengonversi Matriks ke Array: Matriks cosine similarity dan ground truth diubah menjadi array satu dimensi menggunakan flatten(). Ini dilakukan agar mudah dibandingkan dalam bentuk array.

5. Menghitung Prediksi: Prediksi dibuat berdasarkan threshold yang ditentukan. Dalam hal ini, setiap nilai cosine similarity yang lebih besar atau sama dengan threshold dianggap sebagai 1 (relevan), dan nilai lainnya dianggap 0 (tidak relevan).

6. Menghitung Precision dan Recall: Precision dan recall dihitung menggunakan fungsi precision_score dan recall_score dari sklearn. Precision mengukur proporsi prediksi yang relevan dari semua prediksi yang dihasilkan, sedangkan recall mengukur proporsi prediksi yang relevan dari semua ground truth yang relevan.

Hasil:

- Precision: 1.0
  Ini menunjukkan bahwa semua prediksi yang dihasilkan oleh model adalah benar-benar relevan. Dengan kata lain, tidak ada prediksi yang salah positif.

- Recall: 1.0
  Ini menunjukkan bahwa model berhasil menemukan semua item yang relevan dalam ground truth. Dengan kata lain, tidak ada item relevan yang terlewatkan oleh model.

Model rekomendasi buku berhasil dibangun menggunakan algoritma content-based filtering. Proses modeling mencakup ekstraksi fitur dengan TF-IDF, penghitungan cosine similarity, dan implementasi fungsi rekomendasi yang efektif.

Keakuratan dan relevansi rekomendasi diukur menggunakan metrik evaluasi presisi dan recall. Hasil evaluasi menunjukkan bahwa model mencapai nilai presisi dan recall yang tinggi (1.0), yang berarti semua rekomendasi yang diberikan relevan dan semua item relevan berhasil direkomendasikan.

## Kesimpulan

Dengan berhasil menjawab problem statements, mencapai goals yang diharapkan, dan melihat dampak positif dari solusi yang direncanakan, dapat disimpulkan bahwa proyek ini berhasil dalam mengembangkan sistem rekomendasi buku yang efektif . Model content-based filtering yang dibangun memberikan rekomendasi yang relevan dan berkualitas, meningkatkan pengalaman pengguna dalam menemukan buku-buku yang sesuai dengan preferensi mereka. Hasil evaluasi yang menunjukkan presisi dan recall yang tinggi memperkuat kesimpulan bahwa sistem ini berfungsi dengan baik dan memenuhi tujuan bisnis yang telah ditetapkan.
