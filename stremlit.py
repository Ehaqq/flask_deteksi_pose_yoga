import streamlit as st
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Fungsi untuk memuat data dari MongoDB
@st.cache_data
def load_data():
    client = MongoClient('mongodb+srv://sulapsempurna:sempurna0011@yoga.re1gige.mongodb.net/')
    db = client['v4yoga']  # Ganti dengan nama database Anda
    collection = db['deteksi']  # Ganti dengan nama koleksi Anda
    data = list(collection.find())
    df = pd.DataFrame(data)
    # Mengonversi ObjectId menjadi string untuk menghindari masalah dengan Arrow
    df['_id'] = df['_id'].astype(str)
    return df

# Fungsi untuk menghitung nilai statistik berdasarkan pose yoga dan probabilitas
def calculate_metrics(df):
    # Menentukan nilai hipotetis untuk setiap pose yoga
    pose_metrics = {
        'Hanumanasana': {'blood_pressure': 120, 'heart_rate': 75, 'calories': 50},
        'Kapotasana': {'blood_pressure': 115, 'heart_rate': 78, 'calories': 55},
        'Makara Adho Mukha Svanasana': {'blood_pressure': 110, 'heart_rate': 80, 'calories': 60},
        'Janu Sirsana': {'blood_pressure': 125, 'heart_rate': 72, 'calories': 45},
        'Baddha Konasana': {'blood_pressure': 118, 'heart_rate': 76, 'calories': 48},
        'Chakravakasana': {'blood_pressure': 122, 'heart_rate': 74, 'calories': 52},
        'Anjaneyasana': {'blood_pressure': 117, 'heart_rate': 79, 'calories': 57},
        'Dhanurasana': {'blood_pressure': 121, 'heart_rate': 77, 'calories': 53},
    }
    
    metrics = {}
    for pose, values in pose_metrics.items():
        pose_df = df[df['class'] == pose]
        probability_mean = pose_df['probability'].mean()
        metrics[pose] = {
            'blood_pressure': values['blood_pressure'] * probability_mean,
            'heart_rate': values['heart_rate'] * probability_mean,
            'calories': values['calories'] * probability_mean
        }
    return metrics

# Memilih halaman
option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home', 'Dataframe')
)

if option == 'Home' or option == '':
    st.write("# Halaman Utama")
elif option == 'Dataframe':
    st.write("## Dataframe")

    # Memuat data
    df = load_data()

    # Menambahkan input tanggal
    selected_date = st.date_input("Pilih Tanggal", value=datetime.today())
    selected_date_str = selected_date.strftime("%Y-%m-%d")

    # Memastikan ada kolom 'tanggal'
    if 'tanggal' not in df.columns:
        st.error("Data tidak memiliki kolom tanggal.")
        st.stop()

    # Memfilter data berdasarkan tanggal
    df['tanggal'] = pd.to_datetime(df['tanggal']).dt.date
    filtered_df = df[df['tanggal'] == selected_date]

    if filtered_df.empty:
        st.write("Tidak ada data untuk tanggal yang dipilih.")
    else:
        st.write(f"## Data untuk tanggal: {selected_date_str}")

        # Menghitung metrik
        metrics = calculate_metrics(filtered_df)

        # Visualisasi untuk blood_pressure, heart_rate, dan calories
        bp_data = pd.DataFrame(metrics).T

        st.write("## Tekanan Darah pergerakan perpose")
        fig, ax = plt.subplots()
        sns.barplot(x=bp_data.index, y=bp_data['blood_pressure'], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.write(bp_data[['blood_pressure']])

        st.write("## Heart Rate per Pose")
        fig, ax = plt.subplots()
        sns.barplot(x=bp_data.index, y=bp_data['heart_rate'], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.write(bp_data[['heart_rate']])

        st.write("## Calories per Pose")
        fig, ax = plt.subplots()
        sns.barplot(x=bp_data.index, y=bp_data['calories'], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.write(bp_data[['calories']])

        st.write("## Dataframe")
        st.write(filtered_df)

    # Menambahkan input bulan dan tahun
    current_year = datetime.today().year
    current_month = datetime.today().month

    selected_year = st.selectbox("Pilih Tahun", range(current_year - 10, current_year + 11), index=10)
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    selected_month = st.selectbox("Pilih Bulan", months, index=current_month - 1)

    selected_month_str = f"{selected_year}-{months.index(selected_month) + 1:02}"
    st.write("Selected Month:", selected_month_str)

    # Memfilter data berdasarkan bulan dan tahun
    df['year_month'] = df['tanggal'].apply(lambda x: x.strftime('%Y-%m'))
    filtered_month_df = df[df['year_month'] == selected_month_str]

    if filtered_month_df.empty:
        st.write("Tidak ada data untuk bulan yang dipilih.")
    else:
        st.write(f"## Data untuk bulan: {selected_month_str}")

        # Menghitung metrik
        monthly_metrics = calculate_metrics(filtered_month_df)

        # Visualisasi untuk blood_pressure, heart_rate, dan calories
        monthly_bp_data = pd.DataFrame(monthly_metrics).T

        st.write("## Blood Pressure per Pose for the Selected Month")
        fig, ax = plt.subplots()
        sns.barplot(x=monthly_bp_data.index, y=monthly_bp_data['blood_pressure'], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.write(monthly_bp_data[['blood_pressure']])

        st.write("## Heart Rate per Pose for the Selected Month")
        fig, ax = plt.subplots()
        sns.barplot(x=monthly_bp_data.index, y=monthly_bp_data['heart_rate'], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.write(monthly_bp_data[['heart_rate']])

        st.write("## Calories per Pose for the Selected Month")
        fig, ax = plt.subplots()
        sns.barplot(x=monthly_bp_data.index, y=monthly_bp_data['calories'], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.write(monthly_bp_data[['calories']])
