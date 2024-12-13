import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn import metrics
import seaborn as sns
import re
from random import choice
from collections import Counter


# 1. Read data
# Đọc dữ liệu từ file
file_path = 'processed_data.csv'
df_products = pd.read_csv(file_path)

# Sử dụng toàn bộ file để phân tích
if 'full_products' not in st.session_state:  # Đảm bảo chỉ khởi tạo một lần
    st.session_state.full_products = df_products

#6. Load models 
# Import KNN Model:
#file = 'new_data.csv'
#with open('KNN.pkl', 'rb') as file:  
#    KNN = pickle.load(file)

# Import TF_IDF Model:
#with open('TF_IDF_Model.pkl', 'rb') as file:  
#    TF_IDF_Model = pickle.load(file)


# Hàm để tải danh sách từ từ file
def load_words(file_path):
    try:
        # Đọc file và trả về tập hợp các từ
        with open(file_path, "r", encoding="utf-8") as file:
            return set(file.read().splitlines())
    except FileNotFoundError:
        # Thông báo lỗi nếu file không tồn tại
        st.error(f"Không tìm thấy file: {file_path}.")
        return set()    
    
# Hàm kiểm tra từ khóa chính xác bằng regex
def contains_keywords(comment, keywords):
    pattern = r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b'
    return re.findall(pattern, comment.lower())

# Hàm dự đoán cảm xúc
def predict_sentiment(comment, positive_words, negative_words, neutral_words):
    # Kiểm tra dữ liệu null
    if pd.isnull(comment):
        return "Neutral"

    # Tìm các từ tích cực, tiêu cực và trung lập
    positive_matches = contains_keywords(comment, positive_words)
    negative_matches = contains_keywords(comment, negative_words)
    neutral_matches = contains_keywords(comment, neutral_words)

    # Đếm số từ trong từng danh sách
    positive_count = len(positive_matches)
    negative_count = len(negative_matches)
    neutral_count = len(neutral_matches)

    # Logic xác định cảm xúc
    if negative_count > 0:  # Ưu tiên tiêu cực
        return "Negative"
    elif neutral_count > 0 and positive_count == 0:  # Nếu chỉ có từ trung lập
        return "Neutral"
    elif positive_count > 0:  # Nếu chỉ có từ tích cực
        return "Positive"
    else:
        return "Neutral"

# Hàm chính để phân tích cảm xúc của file
def analyze_sentiment(file, positive_words, negative_words, neutral_words):
    # Xác định loại file
    if file.name.endswith('.txt'):
        df = pd.read_csv(file, sep='\t', header=None, names=["Text_contents"], encoding='utf-8')
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file, encoding='utf-8')
    else:
        raise ValueError("File không được hỗ trợ. Vui lòng sử dụng file .txt hoặc .csv")

    # Xử lý cột "Text_contents" nếu cần
    if 'Text_contents' not in df.columns:
        df.columns = ["Text_contents"]

    # Loại bỏ các dòng trống hoặc giá trị không hợp lệ
    df = df.dropna(subset=["Text_contents"]).reset_index(drop=True)

    # Thêm cột "Prediction" cho dự đoán cảm xúc
    df['Prediction'] = df['Text_contents'].apply(
        lambda x: predict_sentiment(x, positive_words, negative_words, neutral_words)
    )

    return df

def count_sentiments(data, product_code):
    product_data = data[data['ma_san_pham'] == product_code]
    if product_data.empty:
        st.write(f"Không tìm thấy sản phẩm với mã: {product_code}")
        return None  # Trả về None khi không có dữ liệu

    if 'cam_xuc' not in data.columns:
        data['cam_xuc'] = data['so_sao'].apply(lambda x: 'Negative' if x <= 2 else 'Neutral' if x == 3 else 'Positive')

    sentiment_counts = product_data['cam_xuc'].value_counts()
    return {
        "Positive": sentiment_counts.get('Positive', 0),
        "Neutral": sentiment_counts.get('Neutral', 0),
        "Negative": sentiment_counts.get('Negative', 0),
    }


# Hàm hiển thị số lượng nhận xét theo cảm xúc
def display_sentiment_counts(data, product_code):
    sentiments = count_sentiments(data, product_code)
    if sentiments:
        print("### Số lượng nhận xét:")
        print(f"- Positive: {sentiments['Positive']}")
        print(f"- Neutral: {sentiments['Neutral']}")
        print(f"- Negative: {sentiments['Negative']}")
    else:
        print("Không có dữ liệu nhận xét cho sản phẩm này.")

# Hàm hiển thị thông tin sản phẩm
def display_product_info(data, product_code):
    product_info = data[data['ma_san_pham'] == product_code]
    if product_info.empty:
        print(f"Không tìm thấy mã sản phẩm: {product_code}")
        return

    product_name = product_info['ten_san_pham'].iloc[0]
    sentiment_counts = count_sentiments(data, product_code)

    print("Thông tin sản phẩm:")
    print(f"Mã sản phẩm: {product_code}")
    print(f"Tên sản phẩm: {product_name}")
    print("Số lượng nhận xét:")
    print(f"Positive: {sentiment_counts['Positive']}")
    print(f"Neutral: {sentiment_counts['Neutral']}")
    print(f"Negative: {sentiment_counts['Negative']}")

def generate_keywords(data, product_code, sentiment, top_n=10):
    """
    Trích xuất các từ khóa quan trọng từ nội dung bình luận theo cảm xúc.

    Args:
    - data (DataFrame): Dữ liệu sản phẩm.
    - product_code (int): Mã sản phẩm cần phân tích.
    - sentiment (str): Loại cảm xúc (Positive, Neutral, Negative).
    - top_n (int): Số lượng từ khóa quan trọng cần trích xuất.

    Returns:
    - List[str]: Danh sách các từ khóa quan trọng.
    """
    # Lọc dữ liệu sản phẩm và cảm xúc
    product_data = data[data['ma_san_pham'] == product_code]
    reviews = product_data[product_data['cam_xuc'] == sentiment]['noi_dung_binh_luan']
    text = ' '.join(reviews.dropna())

    if not text.strip():
        return []

    # Tách các từ và đếm tần suất
    words = text.split()
    word_counts = Counter(words)
    keywords = [word for word, _ in word_counts.most_common(top_n)]
    return keywords

def generate_wordcloud(data, product_code, sentiment):
    """
    Tạo WordCloud từ nội dung bình luận theo cảm xúc.

    Args:
    - data (DataFrame): Dữ liệu sản phẩm.
    - product_code (int): Mã sản phẩm cần phân tích.
    - sentiment (str): Loại cảm xúc (Positive, Neutral, Negative).

    Returns:
    - None: Hiển thị WordCloud.
    """
        # Lọc dữ liệu sản phẩm và cảm xúc
    product_data = data[data['ma_san_pham'] == product_code]
    reviews = product_data[product_data['cam_xuc'] == sentiment]['noi_dung_binh_luan']
    text = ' '.join(reviews.dropna())

    if not text.strip():
        st.write(f"Không có dữ liệu để vẽ WordCloud cho {sentiment}.")
        return

    # Tạo WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))  # Tạo figure để hiển thị trong Streamlit
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"WordCloud - {sentiment}", fontsize=16)
    st.pyplot(fig)  # Sử dụng st.pyplot để hiển thị biểu đồ

def analyze_emotion_distribution(data, ma_san_pham):
    """
    Phân tích và hiển thị phân bố cảm xúc (theo số sao) trên giao diện Streamlit.

    Args:
    - data (DataFrame): Dữ liệu sản phẩm.
    - ma_san_pham (int): Mã sản phẩm cần phân tích.

    Returns:
    - None: Hiển thị biểu đồ trên giao diện Streamlit.
    """
    # Lọc dữ liệu theo mã sản phẩm
    product_data = data[data['ma_san_pham'] == ma_san_pham]

    if product_data.empty:
        st.write(f"Không tìm thấy dữ liệu cho mã sản phẩm: {ma_san_pham}")
        return

    # Tính toán phân bố cảm xúc
    emotion_counts = product_data['cam_xuc'].value_counts()

    # Trực quan hóa phân bố cảm xúc
    fig, ax = plt.subplots(figsize=(8, 6))
    emotion_counts.plot(kind='bar', color=['green', 'orange', 'red'], ax=ax)
    ax.set_title(f"Phân bố cảm xúc cho sản phẩm {ma_san_pham}", fontsize=16)
    ax.set_xlabel('Cảm xúc', fontsize=12)
    ax.set_ylabel('Số lượng nhận xét', fontsize=12)
    ax.set_xticks(range(len(emotion_counts.index)))
    ax.set_xticklabels(emotion_counts.index, rotation=0)
    st.pyplot(fig)

def visualize_comment_frequency_by_code(data, product_code):
    """
    Trực quan hóa tần suất bình luận theo giờ, tháng, và năm cho một sản phẩm cụ thể dựa trên mã sản phẩm.

    Args:
    - data (DataFrame): Tệp dữ liệu chứa thông tin sản phẩm và bình luận.
    - product_code (str): Mã sản phẩm cần phân tích.

    Returns:
    - None: Hiển thị biểu đồ trên giao diện Streamlit.
    """
    # Lọc dữ liệu cho sản phẩm cụ thể dựa trên mã sản phẩm
    product_data = data[data['ma_san_pham'] == product_code]

    if product_data.empty:
        st.write(f"Không tìm thấy mã sản phẩm: {product_code}")
        return

    # Xử lý dữ liệu thời gian
    product_data['ngay_binh_luan'] = pd.to_datetime(product_data['ngay_binh_luan'], errors='coerce')
    product_data['gio_binh_luan'] = product_data['gio_binh_luan'].str.extract(r'(\d{1,2})').astype(float)
    product_data['thang'] = product_data['ngay_binh_luan'].dt.month
    product_data['nam'] = product_data['ngay_binh_luan'].dt.year

    # Phân tích tần suất theo giờ
    hourly_counts = product_data['gio_binh_luan'].value_counts().reindex(range(24), fill_value=0).sort_index()
    fig_hour, ax_hour = plt.subplots(figsize=(10, 5))
    hourly_counts.plot(kind='bar', color='#1f77b4', ax=ax_hour)
    ax_hour.set_title(f'Số lượng bình luận theo giờ - Mã sản phẩm: {product_code}', fontsize=16)
    ax_hour.set_xlabel('Giờ', fontsize=12)
    ax_hour.set_ylabel('Số lượng', fontsize=12)
    ax_hour.set_xticks(range(24))
    ax_hour.set_xticklabels(range(24), rotation=0)
    st.pyplot(fig_hour)

    # Phân tích tần suất theo tháng
    monthly_counts = product_data['thang'].value_counts().reindex(range(1, 13), fill_value=0).sort_index()
    fig_month, ax_month = plt.subplots(figsize=(10, 5))
    monthly_counts.plot(kind='bar', color='#1f77b4', ax=ax_month)
    ax_month.set_title(f'Số lượng bình luận theo tháng - Mã sản phẩm: {product_code}', fontsize=16)
    ax_month.set_xlabel('Tháng', fontsize=12)
    ax_month.set_ylabel('Số lượng', fontsize=12)
    ax_month.set_xticks(range(1, 13))
    ax_month.set_xticklabels(range(1, 13), rotation=0)
    st.pyplot(fig_month)

    # Phân tích tần suất theo năm
    yearly_counts = product_data['nam'].value_counts().sort_index()
    fig_year, ax_year = plt.subplots(figsize=(10, 5))
    yearly_counts.plot(kind='bar', color='#1f77b4', ax=ax_year)
    ax_year.set_title(f'Số lượng bình luận theo năm - Mã sản phẩm: {product_code}', fontsize=16)
    ax_year.set_xlabel('Năm', fontsize=12)
    ax_year.set_ylabel('Số lượng', fontsize=12)
    st.pyplot(fig_year)


# Hàm hiển thị thông tin và xử lý phân tích sản phẩm
def display_product_analysis(product_code):
    """
    Phân tích toàn diện sản phẩm và hiển thị trên giao diện Streamlit.

    Args:
    - product_code (int): Mã sản phẩm cần phân tích.

    Returns:
    - None: Hiển thị kết quả phân tích trên giao diện Streamlit.
    """
    # Lấy thông tin sản phẩm
    selected_product_data = df_products[df_products['ma_san_pham'] == product_code]

    # Kiểm tra nếu không tìm thấy sản phẩm
    if selected_product_data.empty:
        st.write(f"Không tìm thấy sản phẩm với mã: {product_code}")
        return  # Dừng hàm

    # Nếu tìm thấy sản phẩm, hiển thị thông tin
    st.write("### Thông tin sản phẩm:")
    st.write(f"**Mã sản phẩm:** {product_code}")
    st.write(f"**Tên sản phẩm:** {selected_product_data['ten_san_pham'].values[0]}")
    st.write(f"**Điểm trung bình:** {selected_product_data['diem_trung_binh'].values[0]}")

    # Đếm số lượng nhận xét theo cảm xúc
    sentiments = count_sentiments(df_products, product_code)
    if sentiments:
        st.write("### Số lượng nhận xét:")
        st.write(f"- Positive: {sentiments['Positive']}")
        st.write(f"- Neutral: {sentiments['Neutral']}")
        st.write(f"- Negative: {sentiments['Negative']}")

        # Hiển thị từ khóa quan trọng
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            keywords = generate_keywords(df_products, product_code, sentiment)
            if keywords:
                st.write(f"- Các từ khóa {sentiment} quan trọng: {', '.join(keywords)}")

    else:
        st.write("Không có dữ liệu nhận xét cho sản phẩm này.")

    # Tạo WordCloud cho từng loại cảm xúc
    st.write("### Wordcloud các nhận xét")
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        st.write(f"#### WordCloud cho cảm xúc {sentiment}:")
        generate_wordcloud(df_products, product_code, sentiment)

        
# Hiển thị đề xuất ra bảng
def display_recommended_products(recommended_products, cols=5):
    for i in range(0, len(recommended_products), cols):
        cols = st.columns(cols)
        for j, col in enumerate(cols):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:   
                    st.write(product['ten_san_pham'])                    
                    expander = st.expander(f"Bình luận")
                    product_description = product['noi_dung_binh_luan']
                    truncated_description = ' '.join(product_description.split()[:100]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")           

# THIẾT KẾ SIDEBAR:
# Tạo menu
menu = ["Business Objective", "Build Model","New Prediction","Product Analysis"]
choice = st.sidebar.selectbox('Menu', menu)

# Giáo viên hướng dẫn:
st.sidebar.write("""#### Giảng viên hướng dẫn:
                    Khuất Thùy Phương""")
# st.sidebar.write("**Khuất Thùy Phương**")

# Ngày bảo vệ:
st.sidebar.write("""#### Thời gian bảo vệ: 
                 16/12/2024""")

# Hiển thị thông tin thành viên trong sidebar
st.sidebar.write("#### Thành viên thực hiện:")

# Thành viên 1: Nguyễn Văn Thông
st.sidebar.image("Thongnv.jpg", width=150)
st.sidebar.write("**Nguyễn Văn Thông**")

# Thành viên 2: Vũ Trần Ngọc Lan
st.sidebar.image("Lan.jpg", width=150)
st.sidebar.write("**Vũ Trần Ngọc Lan**")

###### Giao diện Streamlit ######
st.image('Sentiment_Analysis.jpg', use_column_width=True)


# GUI
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ##### HASAKI.VN là hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên sâu với hệ thống cửa hàng trải dài trên toàn quốc; và hiện đang là đối tác phân phối chiến lược tại thị trường Việt Nam của hàng loạt thương hiệu lớn.
    ##### Từ những đánh giá của khách hàng, vấn đề được đưa ra là làm sao để các nhãn hàng hiểu khách hàng rõ hơn, biết họ đánh giá gì về sản phẩm, từ đó có thể cải thiện chất lượng sản phẩm cũng như các dịch vụ đi kèm.         
    """)  
    st.image("sentiment-analytis.jpg")
    st.write("""##### => Problem/ Requirement: Xây dựng hệ thống dựa trên lịch sử những đánh giá của khách hàng đã có trước đó. Dữ liệu được thu thập từ phần bình luận và đánh giá của khách hàng ở Hasaki.vn.""")
    st.write("""##### => Problem/ Requirement: Xây dựng hệ thống dựa trên lịch sử những đánh giá của khách hàng đã có trước đó. Dữ liệu được thu thập từ phần bình luận và đánh giá của khách hàng ở Hasaki.vn.""")
    

elif choice == 'Build Model':
    st.subheader("Hasaki - Build Model đánh giá cảm xúc khách hàng Positive, Neutral,Negative dựa vào số sao.")
    st.write("##### 1. Some data")
    st.image("5dong.jpg")
    st.write("##### 2. Trực quan hóa dữ liệu")
    st.write("#####  Wordcloud bình luận")
    st.image("wordcloud_ndbinhluan.jpg")
    st.write("##### Kiểm tra sự cân bằng dữ liệu")
    st.image("bieudo_camxuc.jpg")
    st.write("##### Từ biểu đồ ta thấy dữ liệu mất cân bằng rõ rệt. Sử dụng SMOTE để cân bằng dữ liệu")
    st.image("bieudo_Smote.jpg")
    st.write("###### Phân phối dữ liệu trước SMOTE:'Positive': 14146, 'Negative': 830, 'Neutral': 747")
    st.write("###### Phân phối dữ liệu sau SMOTE:'Positive': 14146, 'Neutral': 2829, 'Negative': 1414")
    st.write("##### 3. Build model...")
    st.write("###### Xây dựng một mô hình sử dụng đa dạng các thuật toán gồm Naive Bayes, KNN và Logistic Regression. Các mô hình được huấn luyện trên các đánh giá của khách hàng về sản phẩm để phân loại thành các mức độ cảm xúc.")
    st.write("##### 4. So sánh kết quả training của 3 mô hình:")
    st.image("sosanh.jpg")
    st.write("##### Confusion Matrix,ROC AUC")
    st.image("matran.jpg")
    st.image("ROC.jpg")
    st.write("##### 5. Kết luận ")
    st.write("###### Mô hình KNN phù hợp nhất đối với Sentiment Analysis của tập dữ liệu của Hasaki.vn.")

elif choice == 'New Prediction':
    st.subheader("Dự đoán cảm xúc của khách hàng cho một sản phẩm thuộc loại nào: Positive, Neutral,Negative?")
    flag = False
    lines = None
   
    # Giao diện Streamlit
    st.title("Phân tích cảm xúc từ bình luận")
    
    # Đọc danh sách từ khóa từ file
    positive_words_file = "positive_words.txt"
    negative_words_file = "negative_words.txt"
    neutral_words_file = "neutral_words.txt"

    positive_words = load_words(positive_words_file)
    negative_words = load_words(negative_words_file)
    neutral_words = load_words(neutral_words_file)

    # Nút radio để chọn kiểu dữ liệu
    type = st.radio("Chọn kiểu dữ liệu để nhập?", options=("Load file", "Input"))

    # Giao diện "Load file"
    if type == "Load file":
    
        uploaded_file = st.file_uploader("Chọn file để upload (txt hoặc csv):", type=["txt", "csv"])
        if uploaded_file:
            try:
                # Xác định loại file và đọc vào DataFrame
                if uploaded_file.name.endswith('.txt'):
                    df = pd.read_csv(uploaded_file, sep='\t', header=None, names=["Text_contents"], encoding='utf-8')
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, encoding='utf-8')

                    # Yêu cầu người dùng chọn cột chứa bình luận nếu file CSV có nhiều cột
                    st.subheader("Chọn cột chứa nội dung bình luận:")
                    column_name = st.selectbox("Chọn cột", df.columns)

                    # Lấy dữ liệu từ cột được chọn
                    df = df[[column_name]].rename(columns={column_name: "Text_contents"})
                else:
                    st.error("File không hợp lệ. Vui lòng tải file định dạng .txt hoặc .csv")
                    df = None
            
                if df is not None:
                    # Loại bỏ các dòng trống hoặc giá trị không hợp lệ
                    df = df.dropna(subset=["Text_contents"]).reset_index(drop=True)

                    # Hiển thị dữ liệu đã tải lên trước
                    st.subheader("Dữ liệu tải lên:")
                    st.dataframe(df)

                    # Dự đoán cảm xúc
                    df['Prediction'] = df['Text_contents'].apply(
                        lambda x: predict_sentiment(x, positive_words, negative_words, neutral_words)
                    )

                # Hiển thị bảng dữ liệu với dự đoán
                st.subheader("Bảng dự đoán cảm xúc:")
                st.dataframe(df)

                # Phân phối cảm xúc
                st.subheader("Phân phối Sentiment:")
                sentiment_counts = df['Prediction'].value_counts()

                # Biểu đồ
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='bar', color=['skyblue', 'orange', 'green'], ax=ax)
                ax.set_title("Thống Kê Số Lượng Sentiment")
                ax.set_xlabel("Sentiment")
                ax.set_ylabel("Số Lượng")
                for i, count in enumerate(sentiment_counts):
                     ax.text(i, count, str(count), ha='center', va='bottom')
                st.pyplot(fig)

                # Lưu kết quả và cung cấp tùy chọn tải về
                st.subheader("Lưu kết quả:")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Tải về CSV",
                    data=csv,
                    file_name='sentiment_analysis_results.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Lỗi khi xử lý file: {e}")

    # Giao diện "Input"
    elif type == "Input":
        st.subheader("Nhập bình luận:")
        user_input = st.text_area("Nhập một hoặc nhiều bình luận (mỗi bình luận trên một dòng):", height=150)

        if st.button("Dự đoán"):
            if user_input.strip():
                # Chuyển dữ liệu đầu vào thành DataFrame
                lines = user_input.strip().split("\n")
                df = pd.DataFrame(lines, columns=["Text_contents"])
                df = df.dropna(subset=["Text_contents"]).reset_index(drop=True)

                # Dự đoán cảm xúc
                df['Prediction'] = df['Text_contents'].apply(
                    lambda x: predict_sentiment(x, positive_words, negative_words, neutral_words)
                )

                # Hiển thị bảng kết quả
                st.subheader("Bảng với dự đoán cảm xúc:")
                st.dataframe(df)

                # Phân phối cảm xúc
                st.subheader("Phân phối Sentiment:")
                sentiment_counts = df['Prediction'].value_counts()

                # Biểu đồ
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='bar', color=['skyblue', 'orange', 'green'], ax=ax)
                ax.set_title("Thống Kê Số Lượng Sentiment")
                ax.set_xlabel("Sentiment")
                ax.set_ylabel("Số Lượng")
                for i, count in enumerate(sentiment_counts):
                    ax.text(i, count, str(count), ha='center', va='bottom')
                st.pyplot(fig)

                # Lưu kết quả và cung cấp tùy chọn tải về
                st.subheader("Lưu kết quả:")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Tải về CSV",
                    data=csv,
                    file_name='sentiment_analysis_results.csv',
                    mime='text/csv',
                )
            else:
                st.warning("Vui lòng nhập ít nhất một bình luận!")


    
    #if flag:
        # Giả lập việc dự đoán cảm xúc
        #st.write("**Kết quả dự báo:**")
        # Mô phỏng dự đoán 
        #from sklearn.feature_extraction.text import TfidfVectorizer
        #from sklearn.neighbors import KNeighborsClassifier

        

        #if len(lines)>0:
            #st.code(lines)        
           # x_new = TF_IDF_Model.transform(lines)        
            #y_pred_new = KNN.predict(x_new)       
            #st.code("Kết quả dự báo (0: Positive, 1: Neutral, 2: Negative): " + str(y_pred_new)) 

elif choice == 'Product Analysis':
    st.subheader("Dự đoán cảm xúc của khách hàng theo số sao cho một sản phẩm thuộc loại nào: Positive, Neutral,Negative?")
    
    # Giao diện Streamlit
    st.title("Product Report")

    # Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
    if 'selected_ma_san_pham' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
        st.session_state.selected_ma_san_pham = None

    # Theo cách cho người dùng chọn sản phẩm từ dropdown
    # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    product_options = [(row['ten_san_pham'], row['ma_san_pham']) for index,
             row in st.session_state.full_products.iterrows()]
    st.session_state.full_products

    # Tạo một dropdown với options là các tuple này
    selected_product = st.selectbox(
        "Chọn sản phẩm",
        options=product_options,
        format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
    )

    # Display the selected product
    st.write("Bạn đã chọn:", selected_product)

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_ma_san_pham = selected_product[1]

    if st.session_state.selected_ma_san_pham:
        st.write("ma_san_pham: ", st.session_state.selected_ma_san_pham)
        # Hiển thị thông tin sản phẩm được chọn
        selected_product = df_products[df_products['ma_san_pham'] == st.session_state.selected_ma_san_pham]

        if not selected_product.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_product['ten_san_pham'].values[0])
            
        # Hiển thị nút "Phân tích sản phẩm"
        if st.button("Phân tích sản phẩm"):
            display_product_analysis(st.session_state.selected_ma_san_pham)
           
            # Phân tích và hiển thị phân bố cảm xúc
            st.write("### Phân bố cảm xúc")
            analyze_emotion_distribution(df_products, st.session_state.selected_ma_san_pham)
            visualize_comment_frequency_by_code(df_products, st.session_state.selected_ma_san_pham)
        

# streamlit run streamlit_Project_1.py