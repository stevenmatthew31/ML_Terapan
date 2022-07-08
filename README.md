# Laporan Proyek Machine Learning -Steven Matthew Gondowijoyo

## Deteksi Stroke
<p align="Justify">
Saat ini gangguan neurologis sangat mempengaruhi kehidupan masyarakat pada tingkat epidemi. Penyakit stroke merupakan penyakit Neurodegeneratif yang paling sering diderita oleh pasien yang berusia diatas 60 tahun. Secara khusus, stroke adalah penyakit kronis gangguan neurologis yang berhubungan dengan hemiplegia, kurangnya keseimbangan dan gaya berjalan yang abnormal. Sebanyak 83-90% penderita mengalami gejala tersebut. Meskipun stroke merupakan penyakit yang tidak menular, namun stroke merupakan penyebab kedua kematian di dunia menurut WHO. Terlebih lagi, usia adalah faktor utama yang paling penting dalam mendeteksi stroke dan fakta bahwa populasi semakin tua, angka-angka ini dapat meningkat lebih lanjut dalam waktu yang tidak terlalu lama. Umumnya, masyarakat sekitar sering menganggap sepele gejala stroke yang dapat berakibat fatal. 
</p>

- <p align="Justify"> Permasalahan tersebut harus diatasi karena seperti yang disebutkan diawal jika stroke terlambat diatasi maka dapat berakibat fatal. Maka dengan hal ini, tujuan dari adanya Proyek ini adalah untuk proses pendeteksian yang singkat, terjangkau, dan nyaman namun tetap memiliki tingkat keakuratan yang tinggi, dan memiliki kontribusi yaitu membantu pencegahan penyakit stroke akut dan dapat melakukan aktivitas dengan normal serta untuk meningkatkan kesejahteraan masyarakat dengan cara melakukan pengecekan secara mandiri oleh masing-masing individu.

## Business Understanding
### Problem Statements
- Bagaimana mendeteksi stroke secara dini?
- Apakah Stroke dapat diatasi?

### Goals
- Stroke dapat dideteksi secara dini dengan menggunakan bantuan dari Machine Learning dimana dengan menginputkan beberapa data yang diperlukan
- Stroke dapat diatasi apabila dapat dideteksi secara dini sehingga dapat dilakukan terapi

## Data Understanding

Seperti yang dijelaskan sebelumnya, alasan proyek ini yaitu deteksi stroke maka diambil sebuah dataset yaitu data sekunder yang berfungsi untuk *``testing``* dan *``training``*. Dataset yang dipilih dapat diakses di [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). Serta pada podul ini, kita akan menggunakan proporsi pembagian sebesar 90:10 (``test_size=0.1``) dengan fungsi ``train_test_split`` dari sklearn seperti pada gambar 
![Train_Test_Split](https://drive.google.com/uc?export=view&id=1kath8iol9iyFHzFIkpQpZuKNuPr2Y728)

### Variabel-variabel pada Stroke dataset adalah sebagai berikut:
- stroke_clean : merupakan variabel yang diambil secara langsung dari data yang diupload.
- stroke_data : merupakan variabel setelah menghilangkan missing variable
- dimension : merupakan variabel pengganti dari paramater/column yang memiliki korelasi yang tinggi

**Rubrik/Kriteria Tambahan (Opsional)**:
- Data Analysis yang telah dilakukan adalah sebagai berikut: 
 
  - Deskripsi Variabel
    Seperti info() dan describe()
  - Missing Variabel
    Dalam proyek ini nilai yang dihilangkan adalah data BMI
  - Univariate Analysis
    Melakukan Numerical Features dan Categorical Features serta memvisualisasikan
  - Multivariate Analysis
    Melakukan Numerical Features dan Categorical Features serta memvisualisasikan
  

## Data Preparation
Pada proyek ini terdapat data preparation sebagai berikut:
- Encoding fitur kategori.
- Reduksi dimensi dengan Principal Component Analysis (PCA).
- Pembagian dataset dengan fungsi train_test_split dari library sklearn
- Standarisasi.

--
- Dilakukan encoding dengan one hot encoder dengan tujuan untuk mengubah data atau variabel kategori menjadi data atau variabel numerik
  ![OneHot](https://drive.google.com/uc?export=view&id=1opOLg3vXC0aiEchvZHCZSAW1IIL0e06F)
  Sehingga, variabel yang dilakukan proses encoding adalah 
  - ``gender``
  - ``ever_married``
  - ``work_type``
  - ``Residence_type``
  - ``smoking_status``
- Reduksi dengan PCA dilakukan untuk mengurangi sejumlah fitur agar menjadi 3 komponen PC. Fitur yang dikurangi adalah sebagai berikut
  ![PCA](https://drive.google.com/uc?export=view&id=1PBttmt7wK7alzEg2i5B713hO0QaN8Y6f)
 
  Sehingga, variabel yang dilakukan pengurangan oleh proses PCA adalah 
  - ``age``
  - ``hypertension``
  - ``heart_disease``
  - ``avg_glucose_level``
 Dan digantikan oleh variabel bernama ``Dimension``
- Tujuan dari pembagian dataset agar kita tidak mengotori data uji dengan informasi yang kita dapat dari data latih. 
- Standarisasi untuk mengubah nilai rata-rata (mean) menjadi 0 dan nilai standar deviasi menjadi 1 agar siap dilatih

## Modeling
Pada tahap ini, saya mengembangkan model machine learning dengan tiga algoritma yaitu 
- K-Nearest Neighbor
  Kita menggunakan k = 10 tetangga dan metric Euclidean untuk mengukur jarak antara titik. Pada tahap ini kita hanya melatih data training dan menyimpan data testing untuk tahap evaluasi yang akan dibahas di Modul Evaluasi Model.
- Random Forest
  Mengimpor RandomForestRegressor dari library scikit-learn. Anda juga mengimpor mean_squared_error sebagai metrik untuk mengevaluasi performa model. Selanjutnya, Anda membuat variabel RF dan memanggil RandomForestRegressor dengan beberapa nilai parameter. Berikut adalah parameter-parameter yang digunakan:
  - n_estimator: jumlah trees (pohon) di forest. Di sini kita set n_estimator=50.
  - max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.
  - random_state: digunakan untuk mengontrol random number generator yang digunakan. 
  - n_jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.
- Boosting Algorithm
  Algoritma boosting terdiri dari dua metode yaitu Adaptive boosting dan Gradient boosting
  
  Parameter-parameter yang digunakan pada kode proyek saya adalah
  - learning_rate: bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting
  - random_state: digunakan untuk mengontrol random number generator yang digunakan.

--
- Kekurangan dari KNN adalah jika dihadapkan pada jumlah fitur atau dimensi yang besar
- Kelebihan dari Random Forest adalah algoritma yang cukup sederhana tetapi memiliki stabilitas yang mumpuni. 
- Algoritma boosting sering mengungguli model yang lebih sederhana seperti logistic regression dan random forest.
- Kemudian, jika dilihat dalam proyek saya juga terdapat perbandingan dari MSE pada 3 algoritma yang digunakan. Dimana yang memiliki model terbaik adalah **_Random Forest_**

## Evaluation
Pada tahap ini metrik evaluasi yang digunakan dari ketiga model tersebut adalah MSE (*Mean Squared Error*)

Sebelumnya juga dilakukan proses scaling fitur agar skala antara data latih dan data uji sama dan kita bisa melakukan evaluasi.

Dihadirkan juga pengujian pada model yang telah digunakan dalam proyek ini sebagai berikut
| | train | test |
| ----------- | :---------: | ----------: |
| **KNN** | 0.000036 | 0.000051 |
| **RF** | 0.000008 | 0.00005 |
| **Boosting** | 0.000043 | 0.000048 |

Serta, untuk pengvisualisasian adalah sebagai berikut
  
 ![MSE](https://drive.google.com/uc?export=view&id=1EvwptvV5xRwdzVOj30qwtgd1G0XR_cZP)

Sehingga, dari tabel dan gambar diatas yang memiliki model terbaik adalah ___**Random Forest**___


--
- MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.
