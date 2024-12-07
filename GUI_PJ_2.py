import streamlit as st
import pandas as pd
import pickle
import base64
import numpy as np
from surprise import KNNBaseline
import gzip

# Các Def cần thiết
def get_top_3_similar_products(ma_sp, sp, corr_mat):
    # Tìm các sản phẩm chung hệ, đảm bảo là danh sách phẳng (flat list)
    sp_cung_he = sp[sp['ma_san_pham'] == ma_sp]['san_pham_cung_he'].tolist()[0]  # Lấy danh sách sản phẩm tương ứng

    # Tìm kiếm dòng có ma_san_pham = ma_sp và trả về index
    index_can_tim = sp[sp['ma_san_pham'] == ma_sp].index.tolist()[0]

    # Tạo result_df từ ma trận tương tự
    result_df = pd.DataFrame(corr_mat.iloc[index_can_tim])
    result_df.rename(columns={result_df.columns[0]: 'similarity'}, inplace=True)

    # Tạo ánh xạ ma_san_pham và diem_trung_binh
    mapping = dict(enumerate(sp['ma_san_pham']))
    diem_trung_binh_mapping = dict(enumerate(sp['diem_trung_binh']))
    sp_cung_he_mapping = dict(enumerate(sp['san_pham_cung_he']))
    gia_ban_mapping = dict(enumerate(sp['gia_ban']))

    # Ánh xạ ma_san_pham và diem_trung_binh vào result_df theo index
    result_df['ma_san_pham'] = result_df.index.map(mapping)
    result_df['diem_trung_binh'] = result_df.index.map(diem_trung_binh_mapping)
    result_df['san_pham_cung_he'] = result_df.index.map(sp_cung_he_mapping).astype('str')
    result_df['gia_ban'] = result_df.index.map(gia_ban_mapping)

    # Sắp xếp theo 'san_pham_cung_he', 'diem_trung_binh' (giảm dần), và 'gia_ban' (tăng dần)
    result_df = result_df.sort_values(by=['san_pham_cung_he', 'diem_trung_binh', 'gia_ban'], ascending=[True, False, True])

    # Lọc bỏ các dòng trùng 'san_pham_cung_he', giữ lại dòng đầu tiên trong mỗi nhóm
    result_df = result_df.drop_duplicates(subset='san_pham_cung_he', keep='first')

    # Sử dụng apply để kiểm tra sự tồn tại của 'ma_san_pham' trong 'san_pham_cung_he'
    top3_similarity = result_df[(result_df['diem_trung_binh'] >= 4) & (~result_df['ma_san_pham'].apply(lambda x: str(x) in sp_cung_he)) & (result_df['similarity'] < 0.9)].sort_values(by='similarity', ascending=False).head(3)

    # Thêm sản phẩm của ma_sp vào kết quả
    top3_similarity = pd.concat([
        sp[sp['ma_san_pham'] == ma_sp][['ma_san_pham']],
        top3_similarity
    ], ignore_index=True)

    # Trả về kết quả sau khi merge với thông tin sản phẩm
    result = pd.merge(top3_similarity, sp[['ma_san_pham', 'ten_san_pham', 'mo_ta', 'gia_goc', 'phan_loai']], on='ma_san_pham', how='left')

    return result
def recommend_userid(userId,algorithm,df,num):
    # df_select = df[(df['ma_khach_hang'] == userId) & (df['so_sao'] >=num)]
    # df_select = df_select.set_index('ma_san_pham')
    # df_select.head(df_select.shape[0])
    df_score = df[["ma_san_pham",'diem_trung_binh','ten_san_pham','gia_ban','san_pham_cung_danh_gia']]
    df_score['EstimateScore'] = df_score['ma_san_pham'].apply(lambda x: algorithm.predict(userId, x).est) # est: get EstimateScore
    df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)
    df_score = df_score.drop_duplicates(subset=['ma_san_pham','ten_san_pham'])
    idx_min = df_score.groupby('san_pham_cung_danh_gia')['gia_ban'].idxmin()
    df_final = df_score.loc[idx_min, ['san_pham_cung_danh_gia','ma_san_pham', 'gia_ban', 'ten_san_pham','diem_trung_binh','EstimateScore']].sort_values(by='EstimateScore', ascending=False)
    f=df_final[0:num]
    return f

#Streamlit
#Main_bacground
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://i.pinimg.com/736x/fe/3d/89/fe3d893b8315d7b0745ce62846f6d4f7.jpg  ');
        background-size: 70%;
        background-position: center;
    }
    </style>
    """, 
    unsafe_allow_html=True
)
#Sidebar background
#Ảnh background sidebar
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

background_image_path = "back_sidebar.jpg"
background_image = get_base64_of_bin_file(background_image_path)

st.markdown(f"""
<style>
[data-testid="stSidebar"] {{
    background-image: url("data:image/jpeg;base64,{background_image}");
    background-size: 80%;
    background-position: right bottom;
    background-repeat: no-repeat;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #C0C0C0;
    }
</style>
""", unsafe_allow_html=True)

#Side bar
menu = ["Tổng Quan", "Đề xuất dựa trên mã sản phẩm", "Đề xuất dựa trên UserID"]
choice = st.sidebar.selectbox('#### **Danh Mục**', menu)
st.sidebar.write("""#### *Thành viên thực hiện:*
                 Nguyễn Quang Khải 
    Trang Thư Đình """)
st.sidebar.write("""#### *Giảng viên hướng dẫn:*
                Khuất Thùy Phương""")
st.sidebar.write("""#### Thời gian thực hiện: 12/2024""")


st.title("Data Science Project:")
st.write("## ***Recommender System***")

#Chi tiết từng page
if choice == 'Tổng Quan':
    # CSS tùy chỉnh
    st.markdown(
        """
        <style>
        .main-title {
            text-align: center;
            font-size: 3rem; /* Tăng kích thước tiêu đề chính */
            font-weight: bold;
            color: #FF6F61;
        }
        .sub-title {
            text-align: center;
            font-size: 1.8rem; /* Tăng kích thước tiêu đề phụ */
            color: #666;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 2rem; /* Tăng kích thước tiêu đề các phần */
            color: #333;
            margin-top: 20px;
            margin-bottom: 10px;
            border-bottom: 2px solid #FF6F61;
        }
        .content-text {
            font-size: 1.2rem; /* Tăng kích thước nội dung văn bản */
            line-height: 1.8; /* Tăng khoảng cách dòng để dễ đọc hơn */
            color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header chính
    st.markdown('<h1 class="main-title">💄 HASAKI.VN Recommender System</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-title">Tăng cường trải nghiệm mua sắm với hệ thống gợi ý cá nhân hóa</h2>', unsafe_allow_html=True)

    # Hình ảnh minh họa
    st.image(
        "hasaki_logo.jpg", 
        caption="Hệ thống mỹ phẩm và chăm sóc sắc đẹp HASAKI.VN", 
        use_container_width=True,
    )

    # Tabs cho từng phần nội dung
    tab1, tab2, tab3 = st.tabs(["Giới thiệu", "Mục tiêu kinh doanh", "Tìm hiểu thêm"])

    with tab1:
        st.markdown('<h2 class="section-title">Giới thiệu về Recommender System</h2>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="content-text">
            Recommender System (Hệ thống gợi ý) là một hệ thống lọc thông tin nhằm dự đoán các mục mà người dùng có thể quan tâm. 
            Chúng thường được áp dụng trong các ngành thương mại như:
            <ul>
                <li>🎵 Gợi ý danh sách phát nhạc/video: Netflix, YouTube, Spotify.</li>
                <li>🛍️ Đề xuất sản phẩm: Amazon, Tiki.</li>
                <li>📚 Gợi ý khóa học: Coursera, Udemy.</li>
                <li>📱 Đề xuất nội dung: Facebook, Twitter.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown('<h2 class="section-title">Mục tiêu kinh doanh tại HASAKI.VN</h2>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="content-text">
            **HASAKI.VN** là hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên sâu với hệ thống trải dài trên toàn quốc. 

            Mục tiêu của hệ thống gợi ý:
            - Đề xuất sản phẩm mỹ phẩm phù hợp nhất cho từng khách hàng.
            - Tăng trải nghiệm người dùng.
            - Tối ưu hóa doanh thu thông qua các gợi ý cá nhân hóa.
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Chi tiết về HASAKI.VN"):
            st.markdown(
                """
                <div class="content-container">
                <div class="content-text">
                HASAKI là đối tác chiến lược tại Việt Nam của nhiều thương hiệu lớn. 
                Khách hàng có thể:
                <ul>
                    <li>Tìm kiếm và lựa chọn sản phẩm phù hợp.</li>
                    <li>Xem các đánh giá và nhận xét từ người dùng khác.</li>
                    <li>Đặt mua sản phẩm một cách nhanh chóng và tiện lợi.</li>
                </ul>
                </div>
                </div>
                """,
                unsafe_allow_html=True, 
            )

    with tab3:
        st.markdown('<h2 class="section-title">Tìm hiểu thêm về Hệ thống Gợi ý</h2>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="content-text">
            Có hai loại Recommender System phổ biến nhất:
            <ul>
                <li><b>Collaborative Filtering (CF):</b> Sử dụng thông tin từ cộng đồng người dùng để đề xuất.</li>
                <li><b>Content-Based Filtering:</b> Dựa vào đặc điểm của sản phẩm và sở thích của người dùng.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("🎯 HASAKI.VN đang tích hợp cả hai phương pháp này để mang lại gợi ý chính xác nhất cho bạn!")


    st.write("---")
    st.success("### 👉 Hãy khám phá cách hệ thống gợi ý của HASAKI.VN có thể nâng cao trải nghiệm của bạn!")
elif choice == 'Đề xuất dựa trên mã sản phẩm':
    st.subheader('Đề xuất dựa trên mã sản phẩm')
    #Đọc file
    sp = sp = pd.read_csv('All_San_pham_clean.csv',sep=';')
    # cosine_matrix = np.load('cosine_similarity.npy', allow_pickle=True)
    with gzip.GzipFile('cosine_similarity.npy.gz', 'rb') as f:
        cosine_matrix = np.load(f)
    df_cosine = pd.DataFrame(cosine_matrix)
    sp_sample = sp.sample(n=15, random_state=40)

    # Tạo select tra theo 2 hướng
    search_mode = st.radio("Bạn muốn tra cứu theo:", ["Tên sản phẩm", "Mã sản phẩm"])

    if search_mode == "Tên sản phẩm":
        # Kết hợp text_input và selectbox
        user_input = st.text_input("Nhập tên sản phẩm:")
        selected_name = st.selectbox(
            "Hoặc chọn từ gợi ý:",
            options=[""] + sp_sample["ten_san_pham"].tolist(),
            help="Bạn có thể nhập tên tự do hoặc chọn từ danh sách gợi ý."
        )
        
        # Ưu tiên giá trị nhập từ text_input
        if user_input:
            search_term = user_input
        else:
            search_term = selected_name

        # Hiển thị kết quả nếu có đầu vào
        if search_term:
           
            matching_products = sp[sp["ten_san_pham"].str.contains(search_term, case=False, na=False)]
            if not matching_products.empty:
                st.write("Kết quả tìm kiếm:")

                product_code = matching_products["ma_san_pham"].iloc[0]
                st.write(f"**Tên sản phẩm:** {search_term}")
                st.write(f"**Mã sản phẩm tương ứng:** {product_code}")
                product_description = matching_products["mo_ta"].iloc[0]
                truncated_description = ' '.join(product_description.split()[:100])
                st.write('##### Thông tin:')
                st.write(truncated_description, '...')
                st.write('#### Sản phẩm gợi ý tương tự:')
                a=product_code
                result_df = get_top_3_similar_products(a, sp, df_cosine)
                # Tạo một container để chứa các sản phẩm
                product_container = st.container()

               
                # Duyệt qua từng hàng trong DataFrame
                for index, row in result_df.iloc[1:].iterrows():
                    with product_container.expander(f"Mã sản phẩm: {row['ma_san_pham']} - {row['ten_san_pham']}"):
                        st.write(f"**Điểm trung bình:** {row['diem_trung_binh']}")
                        st.write(f"**Giá bán:** {row['gia_ban']}")
                        st.write(f"**Mô tả đầy đủ:** {row['mo_ta']}")
            else:
                st.warning("Không tìm thấy tên sản phẩm phù hợp.")
    else:
        # Kết hợp text_input và selectbox
        user_input = st.text_input("Nhập mã sản phẩm:")
        selected_code = st.selectbox(
            "Hoặc chọn từ gợi ý:",
            options=[""] + sp_sample["ma_san_pham"].astype(str).tolist(),
            help="Bạn có thể nhập mã tự do hoặc chọn từ danh sách gợi ý."
        )
        
        # Ưu tiên giá trị nhập từ text_input
        search_term = user_input if user_input else selected_code

        # Hiển thị kết quả nếu có đầu vào
        if search_term:
            # Ensure ma_san_pham is string type
            sp["ma_san_pham"] = sp["ma_san_pham"].astype(str)
            # Search
            matching_products = sp[sp["ma_san_pham"].str.contains(search_term, case=False, na=False)]
            if not matching_products.empty:
                # Get the first product if there's a match
                product_name = matching_products["ten_san_pham"].iloc[0]
                st.write("Kết quả tìm kiếm:")
                st.write(f"**Tên sản phẩm tương ứng:** {product_name}")
                st.write(f"**Mã sản phẩm:** {search_term}")
                product_description = matching_products["mo_ta"].iloc[0]
                truncated_description = ' '.join(product_description.split()[:100])
                st.write('**Thông tin:**')
                st.write(truncated_description, '...')
                st.write('#### Sản phẩm gợi ý tương tự:')
                a=search_term
                result_df = get_top_3_similar_products(a, sp, df_cosine)
                # Tạo một container để chứa các sản phẩm
                product_container = st.container()

                # Tùy chỉnh độ dài mô tả ngắn
                # max_chars_short_description = 100
               
                # Duyệt qua từng hàng trong DataFrame
                for index, row in result_df.iloc[1:].iterrows():
                    with product_container.expander(f"Mã sản phẩm: {row['ma_san_pham']} - {row['ten_san_pham']}"):
                        st.write(f"**Điểm trung bình:** {row['diem_trung_binh']}")
                        st.write(f"**Giá bán:** {row['gia_ban']}")
                        st.write(f"**Mô tả đầy đủ:** {row['mo_ta']}")
                        # # Hiển thị mô tả ngắn với nút "Xem thêm"
                        # mo_ta_ngan = row['mo_ta'][:max_chars_short_description]
                        # if len(row['mo_ta']) > max_chars_short_description:
                        #     mo_ta_ngan += "..."
                        # st.write(f"**Mô tả:** {mo_ta_ngan}")

                        # # Hiển thị đầy đủ mô tả khi click vào nút "Xem thêm"
                        # if st.button("Xem thêm", key=f"button_{index}"):
                            

            else:
                st.warning("Không tìm thấy mã sản phẩm phù hợp.")
elif choice == 'Đề xuất dựa trên UserID':
    df = pd.read_csv('df_data_user.csv')

    with gzip.open('knn_model_compressed.pkl.gz', 'rb') as f:
        algorithm = pickle.load(f)
    
    st.subheader('Đề xuất dựa trên UserID')
    user_input = st.text_input("Nhập mã khách hàng của bạn:")
    

    if id:
        # Hiện lịch sử mua hàng
        try:
            id = int(user_input)
            df_mua_hang=df[df['ma_khach_hang'] == id]
            ten_khach_hang=df_mua_hang['ho_ten_x'].iloc[0]
            st.write("Kết quả tìm kiếm:")
            st.write(f"**Tên khách hàng:** {ten_khach_hang}")
            st.write(f"**Mã khác hàng:** {id}")
            st.write('**Lịch sử mua hàng:**')
            his_b=df_mua_hang[['ma_san_pham', 'ten_san_pham', 'gia_ban', 'diem_trung_binh']]
            st.table(his_b)
            
            # Đề xuất mua hàng:
            result_id= recommend_userid(id,algorithm,df,5)
            x = result_id[['ma_san_pham', 'ten_san_pham', 'gia_ban', 'diem_trung_binh', 'EstimateScore']]
            st.write('#### Sản phẩm gợi ý cho khách hàng:')
            st.table(x)

        except Exception:
            st.error("Mã khách hàng không hợp lệ hoặc không tồn tại. Vui lòng nhập ID là một số nguyên.")
    else:
        st.warning("Bạn chưa nhập ID sản phẩm.")




