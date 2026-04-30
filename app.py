import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Thiết lập cấu hình trang web Streamlit (Nên đặt ở dòng đầu tiên sau khi import)
st.set_page_config(page_title="VN30 PCA Dashboard", page_icon="📈", layout="wide")

# ---------------------------------------------------------
# PHẦN 1: GIAO DIỆN HEADER VÀ THÔNG TIN ĐỒ ÁN
# ---------------------------------------------------------
st.title("📊 ĐỒ ÁN: PHÂN TÍCH CẤU TRÚC THỊ TRƯỜNG VN30 BẰNG PCA")

st.markdown("""
**Thực hiện bởi:** Trần Thị Trúc Xinh (TX) và Nhóm nghiên cứu.
**Học phần:** Phân tích dữ liệu cho tài chính | **Trường Đại học Ngân hàng TP.HCM (HUB)**

Ứng dụng này trực quan hóa quá trình Phân tích Thành phần Chính (PCA) được xây dựng hoàn toàn từ các phép toán Đại số tuyến tính thuần túy (from scratch) trên dữ liệu giá đóng cửa của rổ cổ phiếu VN30.
""")

st.divider() # Đường kẻ ngang phân cách

# ---------------------------------------------------------
# PHẦN 2: SIDEBAR (THANH ĐIỀU HƯỚNG BÊN TRÁI)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Bảng Điều Khiển")
st.sidebar.info("Trạng thái: Đang khởi tạo hệ thống...")

# Tạm thời hiển thị một thông báo thành công để kiểm tra máy chủ
st.success("Khởi tạo thành công máy chủ Localhost cho Streamlit!")
# ---------------------------------------------------------
# PHẦN 3: NẠP VÀ TIỀN XỬ LÝ DỮ LIỆU (SỬ DỤNG CACHE)
# ---------------------------------------------------------
st.header("1. Tổng quan Dữ liệu Thị trường (Data Overview)")
st.markdown("""
Dữ liệu bao gồm các mã cổ phiếu tiêu biểu trong rổ VN30. 
Quá trình tiền xử lý sử dụng `inner join` để đồng bộ hóa các ngày giao dịch hợp lệ và chuyển đổi mức giá đóng cửa thành **Tỷ suất sinh lợi logarit (Daily Log Returns)** nhằm đảm bảo tính dừng (stationarity) cho chuỗi thời gian.
""")

@st.cache_data
def load_and_preprocess_data():
    # Tự động quét 30 file cổ phiếu trong thư mục hiện tại
    all_files = os.listdir('.')
    stock_files = [f for f in all_files if f.startswith("Dữ liệu Lịch sử") and f.endswith(".csv") and "VN 30" not in f]
    vn30_file = "Dữ liệu Lịch sử VN 30.csv"

    def read_clean(file_name, col_name):
        df = pd.read_csv(file_name)
        col_date, col_close = 'Ngày', 'Lần cuối'
        
        # Chỉ giữ lại Ngày và Giá đóng cửa
        df = df[[col_date, col_close]].copy()
        df.rename(columns={col_close: col_name, col_date: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # --- BƯỚC LÀM SẠCH DỮ LIỆU TỐI THƯỢNG ---
        # 1. Ép toàn bộ cột về chuỗi (string)
        s = df[col_name].astype(str)
        # 2. Xóa khoảng trắng và dấu phẩy ngăn cách phần ngàn (ví dụ: 12,345.67 -> 12345.67)
        s = s.str.strip().str.replace(',', '', regex=False)
        # 3. Ép kiểu về số thực (float). Tham số errors='coerce' sẽ biến mọi ký tự rác thành NaN
        df[col_name] = pd.to_numeric(s, errors='coerce')
        
        return df

    # Khởi tạo ma trận gốc với VN30_Index
    df_merged = read_clean(vn30_file, 'VN30_Index')

    # Gộp 30 mã cổ phiếu
    for file in stock_files:
        ticker = file.replace("Dữ liệu Lịch sử ", "").replace(".csv", "")
        df_temp = read_clean(file, ticker)
        df_merged = pd.merge(df_merged, df_temp, on='Date', how='inner')

    # Sắp xếp thời gian và loại bỏ các dòng chứa NaN (do lỗi định dạng đã bị ép thành NaN ở bước trên)
    df_merged = df_merged.sort_values('Date').set_index('Date')
    df_merged = df_merged.dropna()
    
    # Tính Log Returns bằng toán học mảng NumPy để đảm bảo an toàn tuyệt đối
    # Ép kiểu ma trận về float64 trước khi tính toán
    df_merged_float = df_merged.astype('float64')
    df_returns = np.log(df_merged_float / df_merged_float.shift(1)).dropna()
    
    return df_merged_float, df_returns

# Khởi chạy hàm nạp dữ liệu
try:
    df_merged, df_returns = load_and_preprocess_data()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ma trận Giá đóng cửa (Raw Prices)")
        st.dataframe(df_merged.tail())
    with col2:
        st.subheader("Ma trận Tỷ suất sinh lợi (Log Returns)")
        st.dataframe(df_returns.tail())
        
    st.success(f"Đã nạp và làm sạch thành công {len(df_returns.columns)-1} mã cổ phiếu! Kích thước ma trận lợi suất: {df_returns.shape[0]} ngày, {df_returns.shape[1]} biến.")
    
except Exception as e:
    st.error(f"Lỗi hệ thống: {e}")
# ---------------------------------------------------------
# PHẦN 4: THUẬT TOÁN PCA TỪ ĐẦU (FROM SCRATCH) TRÊN STREAMLIT
# ---------------------------------------------------------
st.divider()
st.header("2. Phân tích Thành phần Chính (PCA from scratch)")
st.markdown("""
Khối tính toán này không sử dụng thư viện `sklearn`. Thuật toán được xây dựng hoàn toàn bằng Đại số tuyến tính (`numpy`) thông qua Phân rã giá trị riêng (Eigen Decomposition) của Ma trận Hiệp phương sai.
""")

# Chỉ chạy PCA nếu dữ liệu đã được nạp thành công
if 'df_returns' in locals() or 'df_returns' in globals():
    try:
        # Tách dữ liệu: Bỏ VN30_Index ra khỏi rổ tính toán
        vn30_col = 'VN30_Index' if 'VN30_Index' in df_returns.columns else 'VN30'
        X_returns = df_returns.drop(columns=[vn30_col], errors='ignore')
        
        # 1. Chuẩn hóa dữ liệu (Standardization)
        X_mean = X_returns.mean()
        X_std = X_returns.std()
        X_scaled = (X_returns - X_mean) / X_std
        X_matrix = X_scaled.to_numpy()
        
        # 2. Ma trận hiệp phương sai (Covariance Matrix)
        cov_matrix = np.cov(X_matrix.T)
        
        # 3. Phân rã Eigen (Eigen Decomposition)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 4. Sắp xếp Trị riêng giảm dần
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues_sorted = eigenvalues[sorted_indices]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]
        
        # Tính tỷ lệ phương sai giải thích
        explained_variance = (eigenvalues_sorted / np.sum(eigenvalues_sorted)) * 100
        
        # --- TRỰC QUAN HÓA KẾT QUẢ LÊN DASHBOARD ---
        col_pca1, col_pca2 = st.columns([2, 1])
        
        with col_pca1:
            st.subheader("Top 5 Thành phần chính (Principal Components)")
            df_eigen = pd.DataFrame({
                'Thành phần': [f'PC{i+1}' for i in range(5)],
                'Trị riêng (Lambda)': eigenvalues_sorted[:5].real, # Lấy phần thực
                '% Phương sai giải thích': explained_variance[:5].real
            })
            # Hiển thị bảng định dạng đẹp
            st.dataframe(df_eigen.style.format({
                'Trị riêng (Lambda)': '{:.4f}', 
                '% Phương sai giải thích': '{:.2f}%'
            }), use_container_width=True)
            
        with col_pca2:
            st.subheader("Mức độ giải thích của PC1")
            st.metric(label="Tỷ lệ phương sai của PC1", value=f"{explained_variance[0].real:.2f}%")
            st.info("💡 **Insight:** Mức độ giải thích này đại diện cho Rủi ro hệ thống (Systematic Risk) tác động chung lên toàn bộ rổ VN30.")
            
    except Exception as e:
        st.error(f"Lỗi trong quá trình tính toán PCA: {e}")
# ---------------------------------------------------------
# PHẦN 5: TÁI TẠO PC1 VÀ SO SÁNH VỚI CHỈ SỐ VN30-INDEX
# ---------------------------------------------------------
st.divider()
st.header("3. Trực quan hóa Biến động PC1 và VN30-Index")

# Kiểm tra xem các biến tính toán PCA đã tồn tại chưa
if 'X_scaled' in locals() and 'eigenvectors_sorted' in locals():
    try:
        # 1. Trích xuất vector riêng của PC1 (cột đầu tiên)
        pc1_eigenvector = np.real(eigenvectors_sorted[:, 0])
        
        # 2. Xây dựng chuỗi thời gian PC1 bằng phép nhân ma trận (Dot product)
        # PC1_Returns = X_scaled * Vector_Riêng_PC1
        pc1_returns = np.dot(X_scaled, pc1_eigenvector)
        
        # 3. Kết hợp PC1 và VN30-Index vào cùng 1 DataFrame để phân tích
        df_analysis = pd.DataFrame(index=X_returns.index)
        df_analysis['PC1_Returns'] = pc1_returns
        df_analysis['VN30_Returns'] = df_returns[vn30_col].values
        
        # Tính hệ số tương quan Pearson
        correlation = df_analysis['PC1_Returns'].corr(df_analysis['VN30_Returns'])
        
        # Xử lý vấn đề đảo chiều (Sign Ambiguity)
        if correlation < 0:
            df_analysis['PC1_Returns'] = -df_analysis['PC1_Returns']
            pc1_eigenvector = -pc1_eigenvector  # Đảo chiều vector riêng để dùng cho bước sau
            correlation = -correlation
            
        # 4. Tính Tỷ suất sinh lợi tích lũy (Cumulative Returns)
        df_cumulative = np.exp(df_analysis.cumsum()) - 1
        
        # --- TRỰC QUAN HÓA LÊN DASHBOARD ---
        col_chart, col_metric = st.columns([3, 1])
        
        with col_chart:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_cumulative.index, df_cumulative['PC1_Returns'], label='PC1 (Danh mục nhân tạo)', color='#1f77b4', linewidth=2)
            ax.plot(df_cumulative.index, df_cumulative['VN30_Returns'], label='VN30-Index (Thực tế)', color='#d62728', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_title('SO SÁNH LỢI SUẤT TÍCH LŨY: PC1 vs VN30-INDEX', fontsize=12, fontweight='bold')
            ax.set_ylabel('Lợi suất tích lũy')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend()
            st.pyplot(fig) # Lệnh đặc biệt của Streamlit để in biểu đồ Matplotlib
            
        with col_metric:
            st.metric(label="Hệ số tương quan (Pearson)", value=f"{correlation:.4f}")
            st.markdown("""
            **💡 Insight Tài chính:**
            Hệ số tương quan tiệm cận 1 cho thấy PC1 mô phỏng gần như hoàn hảo nhịp đập của VN30. 
            Mặc dù PC1 được tạo ra hoàn toàn "mù" (không biết công thức tính trọng số vốn hóa của sàn HOSE), nó vẫn tự động hội tụ về **Xu hướng chung của thị trường (Market Trend)**.
            """)
            
    except Exception as e:
        st.error(f"Lỗi trực quan hóa PC1: {e}")

# ---------------------------------------------------------
# PHẦN 6: PHÂN TÍCH TRỌNG SỐ CỔ PHIẾU TRONG PC1 (FACTOR LOADINGS)
# ---------------------------------------------------------
st.divider()
st.header("4. Phân tích Cấu trúc Dẫn dắt Thị trường (Factor Loadings)")

# Đảm bảo các biến từ phần trước đã tồn tại
if 'X_scaled' in locals() and 'eigenvectors_sorted' in locals() and 'pc1_eigenvector' in locals():
    try:
        # 1. Lấy danh sách tên các mã cổ phiếu (bỏ VN30_Index)
        stock_names = X_returns.columns.tolist()

        # 2. Tạo DataFrame chứa trọng số
        df_weights = pd.DataFrame({
            'Mã Cổ Phiếu': stock_names,
            'Trọng số (Loading)': pc1_eigenvector
        })

        # Sắp xếp để vẽ biểu đồ từ thấp lên cao (barh sẽ in từ dưới lên)
        df_weights_sorted = df_weights.sort_values(by='Trọng số (Loading)', ascending=True)

        # --- TRỰC QUAN HÓA LÊN DASHBOARD ---
        col_insight, col_plot = st.columns([1, 2])

        with col_insight:
            st.markdown("""
            ### 💡 Ý nghĩa Tài chính (Insights)
            Biểu đồ bên cạnh thể hiện **Hệ số tải (Factor Loadings)** của từng cổ phiếu trong Thành phần chính đầu tiên (PC1).
            
            *   **Tính đồng biến:** Nếu hầu hết các trọng số đều cùng một dấu (dương hoặc âm), điều này tuân thủ định lý Arbitrage Pricing Theory (APT). Nó chứng tỏ thị trường đang bị chi phối bởi một rủi ro hệ thống chung (ví dụ: tin tức vĩ mô, biến động tỷ giá).
            *   **Nhóm dẫn dắt:** Những mã có giá trị tuyệt đối lớn nhất chính là những "trụ cột" dẫn dắt thị trường. Sự lên xuống của các mã này tác động mạnh mẽ nhất đến PC1.
            *   **Tài sản phòng vệ (Hedging):** Nếu có cổ phiếu đi ngược dấu với phần còn lại, đó có thể là tài sản mang tính phòng thủ trong giai đoạn thị trường hoảng loạn.
            """)
            
            st.subheader("Top 5 Mã chi phối PC1 mạnh nhất")
            # Hiển thị Top 5 mã có giá trị tuyệt đối của trọng số lớn nhất
            top_5 = df_weights.reindex(df_weights['Trọng số (Loading)'].abs().sort_values(ascending=False).index).head(5)
            st.dataframe(top_5.reset_index(drop=True), use_container_width=True)

        with col_plot:
            fig_weights, ax_weights = plt.subplots(figsize=(8, 10))
            
            # Tạo màu: Xanh cho dương (cùng chiều), Đỏ cho âm (ngược chiều)
            colors = ['#1f77b4' if x > 0 else '#d62728' for x in df_weights_sorted['Trọng số (Loading)']]
            
            bars = ax_weights.barh(df_weights_sorted['Mã Cổ Phiếu'], df_weights_sorted['Trọng số (Loading)'], color=colors)
            ax_weights.set_title('TRỌNG SỐ ĐÓNG GÓP CỦA CÁC CỔ PHIẾU VÀO PC1', fontweight='bold', fontsize=14)
            ax_weights.set_xlabel('Hệ số tải (Eigenvector Loadings)', fontsize=12)
            ax_weights.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Gắn nhãn giá trị trực tiếp lên thanh biểu đồ
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 0.005 if width > 0 else width - 0.025
                ax_weights.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                         va='center', fontsize=9)
                         
            st.pyplot(fig_weights)

    except Exception as e:
        st.error(f"Lỗi khi vẽ biểu đồ trọng số: {e}")
# ---------------------------------------------------------
# PHẦN 7: PHÂN TÍCH CHUYÊN SÂU (ADVANCED ANALYTICS)
# ---------------------------------------------------------
st.divider()
st.header("5. Câu hỏi phân tích chuyên sâu")

# Tạo danh sách câu hỏi để người dùng lựa chọn
questions = [
    "--- Chọn câu hỏi để phân tích ---",
    "Q1: Sự tương quan giữa PC1 và VN30-Index mạnh đến mức nào?",
    "Q2: Nhóm cổ phiếu nào chi phối rủi ro hệ thống (PC1) nhiều nhất?",
    "Q3: PC1 giải thích bao nhiêu % biến động của thị trường?",
    "Q4: Có sự phân kỳ (Divergence) nào giữa danh mục PC1 và VN30 không?",
    "Q5: Biến động hằng ngày của PC1 so với VN30-Index ra sao?"
]

selected_q = st.selectbox("Khám phá dữ liệu qua các câu hỏi nghiên cứu:", questions)

if selected_q == questions[1]:
    st.info(f"Hệ số tương quan hiện tại: {correlation:.4f}")
    st.write("Giải thích: Hệ số này cho thấy PC1 đóng vai trò là nhân tố thị trường chung. Nếu > 0.8, PC1 gần như đại diện hoàn toàn cho VN30.")

elif selected_q == questions[2]:
    st.write("Bảng xếp hạng 5 cổ phiếu chi phối nhất:")
    top_5_stocks = df_weights.reindex(df_weights['Trọng số (Loading)'].abs().sort_values(ascending=False).index).head(5)
    st.table(top_5_stocks)

# Tạm thời để các câu hỏi khác chờ xử lý ở bước tiếp theo
# Tiếp nối phần logic Selectbox ở bước trước...

elif selected_q == questions[3]:
    st.subheader("Mức độ giải thích phương sai của các Thành phần chính")
    
    # Vẽ biểu đồ Scree Plot (Biểu đồ dốc)
    fig_scree, ax_scree = plt.subplots(figsize=(10, 5))
    components = [f'PC{i+1}' for i in range(len(explained_variance[:10]))]
    
    # Vẽ cột cho từng PC
    ax_scree.bar(components, explained_variance[:10].real, color='#3498db', alpha=0.7, label='Phương sai riêng lẻ')
    
    # Vẽ đường tích lũy
    cumulative_variance = np.cumsum(explained_variance[:10].real)
    ax_scree.plot(components, cumulative_variance, marker='o', linestyle='-', color='#e74c3c', label='Phương sai tích lũy')
    
    ax_scree.set_ylim(0, 110)
    ax_scree.set_ylabel('% Phương sai giải thích')
    ax_scree.set_title('Scree Plot: Khả năng giải thích thị trường của các PC')
    ax_scree.legend()
    st.pyplot(fig_scree)
    
    st.write(f"**Nhận xét:** PC1 giải thích được {explained_variance[0].real:.2f}% biến động. "
             f"Thông thường trong tài chính, PC1 đại diện cho nhân tố thị trường (Market Factor).")

elif selected_q == questions[4]:
    st.subheader("Phân tích Sự phân kỳ (Divergence)")
    
    # Tính hiệu số giữa PC1 và VN30-Index (đã chuẩn hóa)
    # Chúng ta so sánh lợi suất hằng ngày để tìm điểm bất thường
    diff = df_analysis['PC1_Returns'] - df_analysis['VN30_Returns']
    
    fig_diff, ax_diff = plt.subplots(figsize=(10, 4))
    ax_diff.fill_between(df_analysis.index, diff, color='purple', alpha=0.3)
    ax_diff.plot(df_analysis.index, diff, color='purple', linewidth=1)
    ax_diff.axhline(0, color='black', linestyle='--')
    ax_diff.set_title("Chênh lệch lợi suất hằng ngày (PC1 vs VN30-Index)")
    st.pyplot(fig_diff)
    
    st.warning("💡 **Insight:** Những vùng có chênh lệch lớn (spike) cho thấy cấu trúc trọng số của PC1 đang lệch pha với cách tính vốn hóa của VN30. "
               "Đây có thể là cơ hội để tìm kiếm các cổ phiếu đang bị định giá sai so với rủi ro hệ thống.")
    
# Tiếp nối phần logic Selectbox ở bước trước...

elif selected_q == questions[5]:
    st.subheader("Biến động hằng ngày: PC1 Returns vs VN30 Returns")
    
    # Thiết lập biểu đồ so sánh biến động hằng ngày (Daily Volatility)
    fig_vol, ax_vol = plt.subplots(figsize=(10, 5))
    
    ax_vol.scatter(df_analysis['VN30_Returns'], df_analysis['PC1_Returns'], alpha=0.5, color='#2ecc71')
    
    # Vẽ đường 45 độ (Đường tham chiếu y=x)
    lims = [
        np.min([ax_vol.get_xlim(), ax_vol.get_ylim()]),
        np.max([ax_vol.get_xlim(), ax_vol.get_ylim()]),
    ]
    ax_vol.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Đường tương quan hoàn hảo')
    
    ax_vol.set_xlabel('VN30 daily returns')
    ax_vol.set_ylabel('PC1 daily returns')
    ax_vol.set_title('Scatter Plot: Sự đồng nhất trong biến động hằng ngày')
    ax_vol.legend()
    ax_vol.grid(True, linestyle=':', alpha=0.6)
    
    st.pyplot(fig_vol)
    
    st.write("""
    **💡 Insight Tài chính:**
    Biểu đồ phân tán (Scatter Plot) này cho thấy mức độ tập trung của các điểm dữ liệu quanh đường tham chiếu. 
    *   Các điểm nằm sát đường đỏ: PC1 mô phỏng rất tốt biến động thực tế.
    *   Các điểm nằm xa: Đại diện cho những ngày thị trường có biến động bất thường mà các nhân tố hệ thống không giải thích hết được (Idiosyncratic events).
    """)

# --- PHẦN KẾT THÚC DASHBOARD ---
st.sidebar.success("✅ Hệ thống đã sẵn sàng!")
st.sidebar.markdown("---")
st.sidebar.write("© 2026 - Đồ án Tài chính - HUB")