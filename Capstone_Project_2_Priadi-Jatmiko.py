#!/usr/bin/env python
# coding: utf-8

# # Latar Belakang
# 
# Airbnb merupakan salah satu platform penyedia akomodasi terbesar di dunia yang menghubungkan pemilik properti dengan wisatawan atau penyewa jangka pendek. Kehadiran Airbnb telah mengubah pola industri perhotelan dengan menghadirkan pilihan akomodasi yang lebih beragam, mulai dari kamar pribadi hingga hunian lengkap dengan harga yang bervariasi.
# 
# Bangkok sebagai ibu kota Thailand dikenal sebagai salah satu destinasi wisata internasional dengan jumlah kunjungan wisatawan yang sangat tinggi setiap tahunnya. Kota ini menawarkan daya tarik berupa budaya, kuliner, hiburan malam, hingga pusat perbelanjaan. Dengan tingginya arus wisatawan, permintaan terhadap akomodasi di Bangkok juga meningkat pesat, baik berupa hotel maupun alternatif lain seperti Airbnb.
# 
# Data Airbnb Listings Bangkok yang digunakan dalam proyek ini berisi informasi ribuan listing akomodasi di Bangkok, mencakup detail seperti harga sewa, tipe kamar, lokasi (neighbourhood), tingkat ketersediaan (availability), hingga jumlah ulasan. Melalui analisis data ini, kita dapat memahami pola harga, faktor-faktor yang memengaruhi ketersediaan dan popularitas listing, serta perbedaan karakteristik antar lokasi dan tipe kamar.
# 

# # Pernyataan Masalah
# 
# Tujuannya adalah memberikan insight kepada Airbnb host/manajemen untuk:
# 1. Mengetahui faktor apa saja yang memengaruhi harga listing.
# 2. Mengetahui persebaran listing berdasarkan lokasi.
# 3. Mengetahui preferensi pasar (room type, minimum nights, dll).
# 4. Memberikan rekomendasi strategi pricing dan pengembangan bisnis.
# 
# 
# # Data
# 
# Untuk menjawab pertanyaan di atas, kita akan menganalisa data Airbnb Listings di Bangkok yang sudah dikumpulkan oleh perusahaan. Dataset dapat diakses [di sini](https://drive.google.com/file/d/1tjHTQKOyNIxlb-rSmG4VzkxHlq-v8mf9/view?usp=drive_link).
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap


# Dataset ini berisi informasi terkait identitas listing, host, lokasi, tipe kamar, harga, ketersediaan, dan aktivitas review/ulasan yang terdiri dari 16 kolom, yaitu:
# 1. **id** : Kode unik Airbnb untuk daftar properti.
# 2. **name** : Nama daftar properti.
# 3. **host_id** : Kode unik Airbnb untuk pemilik/pengelola properti.
# 4. **host_name** : Nama dari pemilik/pengelola properti. Biasanya hanya nama depan.
# 5. **neighbourhood** : Lingkungan sekitar dipetakan secara geografis menggunakan koordinat lintang dan bujur terhadap lingkungan yang didefinisikan oleh file shapefile digital terbuka atau publik.
# 6. **latitude** : Menggunakan proyeksi Sistem Geodetik Dunia (WGS84) untuk lintang dan bujur.
# 7. **longitude** : Menggunakan proyeksi Sistem Geodetik Dunia (WGS84) untuk lintang dan bujur.
# 8. **room_type** : [Rumah/apartemen utuh | Kamar pribadi | Kamar bersama | Hotel]
#    Semua properti dibagi menjadi tiga jenis kamar berikut:
# 
#    **Rumah Utuh**
#    adalah pilihan terbaik jika Anda mencari tempat tinggal yang nyaman seperti di rumah sendiri. Dengan rumah utuh, Anda akan memiliki seluruh ruang untuk diri sendiri. Biasanya termasuk kamar tidur, kamar mandi, dapur, dan pintu masuk terpisah yang khusus. Tuan rumah harus mencantumkan dalam deskripsi apakah mereka akan berada di properti atau tidak (contoh: “Tuan rumah menempati lantai pertama rumah”) dan memberikan detail tambahan di daftar.
#    
#    **Kamar Pribadi**
#    cocok jika Anda menginginkan privasi tambahan tetapi tetap menghargai koneksi lokal. Saat memesan kamar pribadi, Anda akan memiliki kamar tidur pribadi dan mungkin berbagi beberapa ruang dengan orang lain. Anda mungkin perlu melewati ruang dalam yang ditempati oleh tuan rumah atau tamu lain untuk mencapai kamar Anda.
# 
#    **Kamar Bersama**
#    cocok untuk Anda yang tidak keberatan berbagi ruang dengan orang lain. Saat memesan kamar bersama, Anda akan tidur di ruang yang dibagikan dengan orang lain dan berbagi seluruh ruang dengan orang lain. Kamar bersama populer di kalangan pelancong fleksibel yang mencari teman baru dan penginapan yang ramah anggaran.
# 9. **price** : Harga harian dalam mata uang lokal. Catatan: Tanda $ mungkin digunakan meskipun sesuai dengan pengaturan wilayah.
# 10. **minimum_nights** : Jumlah malam minimum yang harus diinap untuk daftar ini (aturan kalender mungkin berbeda).
# 11. **number_of_reviews** : Jumlah ulasan yang dimiliki oleh daftar tersebut.
# 12. **last_review** : Tanggal ulasan terakhir/terbaru.
# 13. **review_per_month** : Rata-rata jumlah ulasan yang diterima per bulan.
# 14. **calculated_host_listings_count** : Jumlah daftar yang dimiliki pemilik/pengelola properti dalam pengambilan data saat ini di wilayah kota/daerah tersebut.
# 15. **availability_365** : Ketersediaan_x. Kalender menentukan ketersediaan daftar properti x hari ke depan. Perhatikan bahwa sebuah daftar properti mungkin tersedia karena telah dipesan oleh tamu atau diblokir oleh pemilik/pengelola properti.
# 16. **number_of_reviews_ltm** : Jumlah ulasan yang dimiliki oleh daftar tersebut (dalam 12 bulan terakhir).
# 
# Berikut adalah 5 data teratas dan 5 data terbawah dari dataset tersebut:

# In[111]:


# Membaca Dataset 'Airbnb Listings Bangkok.csv'
df = pd.read_csv('Airbnb Listings Bangkok.csv', index_col=0)

# Menampilkan 5 Baris Teratas dan 5 Baris Terbawah
display(df.head(), df.tail())


# # Info Dataset
# 
# Memahami data yang akan dilakukan analisis, dengan melakukan pengecekan terlebih dahulu apakah ada anomali-anomali pada dataset tersebut yang perlu ditangani dalam tahapan data cleaning.
# 

# In[112]:


# Menampilkan Informasi DataFrame

print(f'Jumlah baris dan kolom pada dataset tersebut adalah {df.shape}')
df.info()


# 
# **Keterangan:**
# 1. Jumlah Baris Data : 15.854 baris data
# 2. Data Kosong (Missing Values) yaitu pada Kolom :
#    - **name** : 8 baris
#    - **host_name** : 1 baris
#    - **last_review** : 5.790 baris
#    - **reviews_per_month** : 5.790 baris
# 
# 
# # Cek Missing Values
# 

# In[113]:


# Cek Missing Value

listItem = []
for col in df.columns :
    listItem.append([col, df[col].dtype, df[col].isna().sum(),df[col].nunique(), list(df[col].drop_duplicates().sample(2).values)]);

dfDesc = pd.DataFrame(columns=['Column Name', 'DataType', 'Null', 'Number of Unique', 'Unique Sample'],data=listItem)
print(df.shape)
dfDesc


# # Pembersihan Data
# 
# - Isi data kosong **reviews_per_month** dengan angka 0.
# - Batasi **minimum_nights** ke 365.
# - Buang outlier yang memiliki harga ekstrem (price <= 0 atau > 50000
#   

# In[114]:


# Data Cleaning dan Isi Data Kosong

df_clean = df.copy()
df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)
df_clean['minimum_nights'] = df_clean['minimum_nights'].clip(upper=365)
df_clean = df_clean[(df_clean['price'] > 0) & (df_clean['price'] <= 50000)].reset_index(drop=True)
print('After cleaning shape:', df_clean.shape)
df_clean[['price','minimum_nights','reviews_per_month']].describe()


# **Keterangan:**
# * Jumlah baris : 15796 (berkurang 58 baris)
# * Jumlah kolom : 16

# # Visualisasi Analisis
# 

# In[115]:


# Distribusi Harga

plt.figure(figsize=(16,8))  
ax = sns.histplot(df_clean['price'], bins=100, kde=True)  
plt.xlim(0, 5000)
plt.title('Distribusi Harga', fontsize=20)
plt.xlabel("Harga", fontsize=14)
plt.ylabel("Jumlah Listing", fontsize=14)
plt.grid()

plt.show()


# In[116]:


# Distribusi Room Type

plt.figure(figsize=(6,4))
sns.countplot(x='room_type', data=df_clean, order=df_clean['room_type'].value_counts().index)
plt.title('Distribusi Room Type')
plt.grid()
plt.show()


# In[117]:


# Lokasi dengan Listing Terbanyak

plt.figure(figsize=(10,5))
df_clean['neighbourhood'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Distrik dengan Listing Terbanyak')
plt.ylabel('Jumlah Listing')
plt.grid()
plt.show()


# In[ ]:


# Sebaran Lokasi Listing/Properti dengan Marker Cluster

# Pusat peta Bangkok
mapsBangkok = folium.Map(location=[13.7563, 100.5018], zoom_start=11, tiles="CartoDB positron")

marker_cluster = MarkerCluster().add_to(mapsBangkok)

# Warna per room_type
color_dict = {
    "Entire home/apt": "red",
    "Private room": "blue",
    "Shared room": "green",
    "Hotel room": "purple"
}

for _, row in df_clean.iterrows():  # pakai subset biar gak berat
    room = row['room_type']
    price = row['price']
    name = row['name'] if 'name' in df_clean.columns else "No Name"
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color=color_dict.get(room, "gray"),
        fill=True,
        fill_color=color_dict.get(room, "gray"),
        fill_opacity=0.6,
        popup=f"<b>{name}</b><br>Type: {room}<br>Price: ฿{price}"
    ).add_to(marker_cluster)

mapsBangkok



# In[ ]:


# Sebaran Lokasi Listing/Properti dengan HeatMap

# Peta Bangkok
mapsBangkok = folium.Map(location=[13.7563, 100.5018], zoom_start=11, tiles="CartoDB positron") #dark_matter

heat_data = df_clean[['latitude', 'longitude', 'price']].dropna().values.tolist()

# Heatmap (harga semakin mahal makin merah)
HeatMap(heat_data, 
        radius=8, 
        blur=1,  # 15 
        max_zoom=13).add_to(mapsBangkok)

mapsBangkok


# # Analisis Faktor Harga
# 

# In[120]:


# Korelasi Numerik dengan Harga

plt.figure(figsize=(8,6))
corr = df[['price', 'minimum_nights', 'number_of_reviews', 
           'reviews_per_month', 'calculated_host_listings_count', 
           'availability_365', 'number_of_reviews_ltm']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Korelasi Variabel Numerik dengan Harga")
plt.show()


# 
# **Korelasi Numerik** : kemungkinan **reviews_per_month** dan **availability_365** ada pengaruh ke harga, tapi tidak besar (biasanya harga lebih banyak dipengaruhi oleh lokasi & tipe kamar).
# 

# In[124]:


# Perbandingan Harga per Jenis Kamar/Room Type

plt.figure(figsize=(8,5))
sns.boxplot(x='room_type', y='price', data=df)
plt.ylim(0, 5000)  # batasi agar outlier ga bikin kacau
plt.title("Perbandingan Harga per Jenis Kamar")
plt.show()


# 
# **Perbandingan Harga berdasarkan Tipe Kamar** : **Entire home/apt** paling mahal, **Shared room** paling murah.
# 

# In[127]:


# Harga rata-rata per Neighbourhood (Top 10 Distrik)

top_neigh = df['neighbourhood'].value_counts().head(10).index
plt.figure(figsize=(10,5))
sns.barplot(x='neighbourhood', y='price', data=df[df['neighbourhood'].isin(top_neigh)], estimator=lambda x: x.mean())
plt.xticks(rotation=45)
plt.title("Harga Rata-rata per Distrik (Top 10)")
plt.ylabel("Harga (Baht)")
plt.show()


# **Neighbourhood** : Beberapa distrik premium jauh lebih mahal.

# 
# # Kesimpulan
# - Mayoritas listing harganya < 2000 Baht (sekitar Rp 2 juta).
# - Ada outlier dengan harga sangat tinggi, mungkin luxury/villa premium.
# - **Entire home/apt** : paling mahal.
# - **Shared room** : paling murah.
# - **Sesuai ekspektasi** : semakin privat, semakin mahal.
# - Beberapa distrik (misalnya dekat pusat kota) mendominasi listing.
# - Distrik dengan listing terbanyak biasanya area wisata/akses transportasi mudah.
# - Listing terkonsentrasi di pusat kota Bangkok.
# - Warna menunjukkan jenis kamar, terlihat persebaran entire home/apt vs private room.
# 
# # Rekomendasi
# - **Optimalkan room type**. Jika memungkinkan, ubah listing dari shared/private room ke entire home/apt untuk bisa meningkatkan harga dan permintaan.
# - **Strategi lokasi**. Properti dekat pusat kota bisa dipasarkan dengan harga premium. Properti jauh, dapat fokus pada keunikan (villa, resort, view bagus) agar tetap menarik.
# - **Kelola review**. Listing dengan review lebih banyak cenderung punya harga lebih stabil. Menghimbau tamu untuk meninggalkan review, sehingga bisa meningkatkan persepsi kualitas.
# 
# 

# In[ ]:




