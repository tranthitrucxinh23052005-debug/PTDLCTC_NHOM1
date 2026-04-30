import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="VN30 PCA Analysis | HUB Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Nghiêm túc, chuyên nghiệp) ---
st.markdown("""
    <style>
    /* Tổng thể */
    .main { background-color: #f4f7f9; }
    .stApp { color: #2c3e50; }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] { font-size: 28px; color: #1e3d59; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #555; }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 5px solid #1e3d59;
    }
    
    /* Footer */
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #ffffff; color: #7f8c8d; text-align: center;
        padding: 8px; font-size: 12px; border-top: 1px solid #dee2e6;
        z-index: 100;
    }
    
    /* Header & Sidebar */
    h1, h2, h3 { color: #1e3d59; font-family: 'Segoe UI', sans-serif; }
    .stSidebar { background-color: #f8f9fa; border-right: 1px solid #eee; }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #1e3d59 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #1e3d59;'>🏛️</h1>", unsafe_allow_html=True)
    st.title("Hệ Thống Phân Tích")
    
    with st.expander("ℹ️ Thông tin Đồ án", expanded=True):
        st.markdown(f"""
        **Sinh viên thực hiện:**
        - Trần Thị Trúc Xinh 
        - Đào Việt Anh
        - Nguyễn Phan Quỳnh Thy 
        - Nguyễn Thị Nhã Phương

        **Học phần:**
        Phân tích dữ liệu cho tài chính
                    
        **Trường:** ĐH Ngân hàng TP.HCM (HUB)
        """)
    
    st.divider()
    st.markdown("### 🛠️ Trạng thái hệ thống")
    st.success("Máy chủ: **Sẵn sàng**")
    
    if st.button("🔄 Làm mới dữ liệu (Clear Cache)", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# --- HEADER ---
st.title("📊 PHÂN TÍCH CẤU TRÚC THỊ TRƯỜNG VN30")
st.caption("Ứng dụng Phân tích Thành phần Chính (PCA) để bóc tách rủi ro hệ thống và mức độ nhạy cảm của rổ chỉ số VN30.")

# --- NẠP VÀ TIỀN XỬ LÝ DỮ LIỆU ---
@st.cache_data
def load_and_process_data():
    all_files = os.listdir('.')
    stock_files = [f for f in all_files if f.startswith("Dữ liệu Lịch sử") and f.endswith(".csv") and "VN 30" not in f]
    vn30_file = "Dữ liệu Lịch sử VN 30.csv"

    if not os.path.exists(vn30_file):
        return None, None

    def read_clean(file_name, col_name):
        try:
            df = pd.read_csv(file_name)
            col_date, col_close = 'Ngày', 'Lần cuối'
            df = df[[col_date, col_close]].copy()
            df.rename(columns={col_close: col_name, col_date: 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True) 
            s = df[col_name].astype(str).str.strip().str.replace(',', '', regex=False)
            df[col_name] = pd.to_numeric(s, errors='coerce')
            return df
        except:
            return None

    try:
        df_merged = read_clean(vn30_file, 'VN30_Index')
        if df_merged is None: return None, None

        for file in stock_files:
            ticker = file.replace("Dữ liệu Lịch sử ", "").replace(".csv", "")
            df_temp = read_clean(file, ticker)
            if df_temp is not None:
                df_merged = pd.merge(df_merged, df_temp, on='Date', how='inner')

        df_merged = df_merged.sort_values('Date').set_index('Date').dropna()
        df_merged_float = df_merged.astype('float64')
        df_returns = np.log(df_merged_float / df_merged_float.shift(1)).dropna()
        return df_merged_float, df_returns
    except:
        return None, None

# --- KHỞI CHẠY LOGIC ---
df_merged, df_returns = load_and_process_data()

if df_merged is not None:
    # 1. TÍNH TOÁN PCA (Linear Algebra)
    vn30_col = 'VN30_Index'
    X_returns = df_returns.drop(columns=[vn30_col], errors='ignore')
    
    # Chuẩn hóa (Z-score)
    X_scaled = (X_returns - X_returns.mean()) / X_returns.std()
    
    # Ma trận hiệp phương sai & Trị riêng
    cov_matrix = np.cov(X_scaled.to_numpy().T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sắp xếp trị riêng
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = np.real(eigenvalues[sorted_indices])
    eigenvectors_sorted = np.real(eigenvectors[:, sorted_indices])
    
    # Tỷ lệ giải thích phương sai
    explained_variance = (eigenvalues_sorted / np.sum(eigenvalues_sorted)) * 100
    
    # Tính toán PC1
    pc1_eigenvector = eigenvectors_sorted[:, 0]
    pc1_returns = np.dot(X_scaled, pc1_eigenvector)
    correlation = pd.Series(pc1_returns).corr(df_returns[vn30_col].reset_index(drop=True))
    
    if correlation < 0:
        pc1_returns = -pc1_returns
        pc1_eigenvector = -pc1_eigenvector
        correlation = -correlation
    
    df_cumulative = np.exp(pd.DataFrame({
        'PC1_Returns': pc1_returns,
        'VN30_Returns': df_returns[vn30_col].values
    }, index=X_returns.index).cumsum()) - 1

    # --- BỐ CỤC TABS ---
    tab1, tab2, tab3 = st.tabs(["📋 TỔNG QUAN DỮ LIỆU", "🧬 KẾT QUẢ PCA", "🔍 PHÂN TÍCH CHUYÊN SÂU"])

    with tab1:
        st.subheader("1. Tổng quan Dữ liệu Thị trường")
        st.markdown("""
        Dữ liệu được trích xuất từ các tệp CSV lịch sử, đồng bộ hóa theo thời gian và chuyển đổi sang **Log Returns** 
        để đảm bảo tính phân phối chuẩn trong phân tích tài chính.
        """)
        
        col_info1, col_info2 = st.columns([1, 1])
        with col_info1:
            st.markdown("**📊 Ma trận Giá đóng cửa (Raw Data)**")
            st.dataframe(df_merged.tail(10), use_container_width=True)
        with col_info2:
            st.markdown("**📈 Ma trận Tỷ suất sinh lợi (Log Returns)**")
            st.dataframe(df_returns.tail(10), use_container_width=True)
            
        st.info(f"✅ **Thống kê:** Đã xử lý {len(df_returns.columns)-1} mã cổ phiếu thành phần và {len(df_returns)} phiên giao dịch khớp lệnh.")

    with tab2:
        st.subheader("2. Hiệu năng Thành phần Chính (Principal Components)")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Giải thích bởi PC1", f"{explained_variance[0]:.2f}%")
        m2.metric("Tương quan PC1 vs VN30", f"{correlation:.4f}")
        
        cum_var = np.cumsum(explained_variance)
        num_pc_90 = np.argmax(cum_var > 90) + 1
        m3.metric("Số PC đạt >90% Var", f"{num_pc_90}")

        st.divider()
        
        c_p1, c_p2 = st.columns([1.6, 1])
        with c_p1:
            st.markdown("**So sánh Lợi suất Tích lũy: PC1 (Danh mục ảo) vs VN30-Index**")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_cumulative.index, df_cumulative['PC1_Returns'], label='PC1 (Artificial Portfolio)', color='#1e3d59', linewidth=2)
            ax.plot(df_cumulative.index, df_cumulative['VN30_Returns'], label='VN30-Index (Actual)', color='#e74c3c', linestyle='--', linewidth=1.5)
            ax.set_ylabel('Lợi suất tích lũy', fontsize=10)
            ax.set_xlabel('Thời gian', fontsize=10)
            ax.legend(frameon=True)
            ax.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig)
            
        with c_p2:
            st.markdown("**Bảng giá trị Trị riêng (Eigenvalues)**")
            df_eigen = pd.DataFrame({
                'Thành phần': [f'PC{i+1}' for i in range(10)],
                'Trị riêng': eigenvalues_sorted[:10],
                '% Giải thích': explained_variance[:10]
            })
            st.table(df_eigen.style.format({'Trị riêng': '{:.4f}', '% Giải thích': '{:.2f}%'}))
            st.caption("Ghi chú: PC1 thường đại diện cho xu hướng chung (Market Risk).")

    with tab3:
        st.subheader("3. Cấu trúc dẫn dắt & Hệ số tải (Loadings)")
        
        c_i1, c_i2 = st.columns([1, 1.2]) 
        with c_i1:
            st.markdown("""
            ### 🧬 Phân tích Factor Loadings
            Biểu đồ bên cạnh thể hiện mức độ đóng góp của từng cổ phiếu vào **Thành phần chính số 1 (PC1)**. 
            
            - **Trọng số dương cao:** Các cổ phiếu mang tính dẫn dắt thị trường (Market Leaders).
            - **Tính đồng nhất:** Nếu tất cả các mã đều có trọng số cùng dấu, thị trường có tính tương quan hệ thống rất cao.
            """)
            
            st.markdown("**Top 5 Mã chi phối PC1**")
            df_weights = pd.DataFrame({'Mã': X_returns.columns, 'Trọng số': pc1_eigenvector})
            top_5 = df_weights.reindex(df_weights['Trọng số'].abs().sort_values(ascending=False).index).head(5)
            st.dataframe(top_5.style.background_gradient(cmap='Blues'), use_container_width=True)
            
        with c_i2:
            fig_w, ax_w = plt.subplots(figsize=(8, 10))
            df_w_sorted = df_weights.sort_values(by='Trọng số')
            colors = ['#3498db' if x > 0 else '#e74c3c' for x in df_w_sorted['Trọng số']]
            ax_w.barh(df_w_sorted['Mã'], df_w_sorted['Trọng số'], color=colors, alpha=0.8)
            ax_w.set_title('VN30 FACTOR LOADINGS (PC1)', fontweight='bold', pad=20)
            ax_w.axvline(0, color='black', linewidth=0.8)
            ax_w.grid(axis='x', linestyle='--', alpha=0.4)
            st.pyplot(fig_w)

        st.divider()
        
        # --- 5 CÂU HỎI NGHIÊN CỨU CHI TIẾT ---
        st.markdown("### 🔍 Khám phá sâu qua các câu hỏi nghiên cứu")
        questions = [
            "--- Chọn câu hỏi nghiên cứu ---",
            "Q1: PC1 có thực sự đại diện cho chỉ số VN30 không?",
            "Q2: Những cổ phiếu nào đóng vai trò 'đầu tàu' rủi ro hệ thống mạnh nhất?",
            "Q3: Cần bao nhiêu nhân tố để giải thích 90% biến động của thị trường?",
            "Q4: PC2 và PC3 đại diện cho yếu tố gì (Ngành hay đặc thù)?",
            "Q5: Ứng dụng kết quả PCA này vào quản trị danh mục như thế nào?"
        ]
        selected_q = st.selectbox("Lựa chọn khía cạnh cần giải trình:", questions)
        
        if selected_q == questions[1]:
            st.success(f"**Kết luận:** Hệ số tương quan đạt {correlation:.4f}. Vì PC1 giải thích đến {explained_variance[0]:.2f}% phương sai và có tương quan cực cao với VN30, ta có thể khẳng định PC1 chính là 'Nhân tố thị trường' (Market Factor).")
        
        elif selected_q == questions[2]:
            st.info(f"**Phân tích:** Các cổ phiếu đứng đầu bảng trọng số như **{top_5['Mã'].iloc[0]}** và **{top_5['Mã'].iloc[1]}** là những mã nhạy cảm nhất. Khi thị trường chung biến động, các mã này sẽ phản ứng mạnh nhất, đóng vai trò dẫn dắt rủi ro hệ thống.")
        
        elif selected_q == questions[3]:
            st.warning(f"**Phân tích:** Theo biểu đồ Scree Plot, chúng ta cần tổng cộng **{num_pc_90}** thành phần chính để vượt ngưỡng 90% phương sai. Điều này cho thấy rổ VN30 có tính tập trung cao.")
            fig_s, ax_s = plt.subplots(figsize=(10, 4))
            pcs_labels = [f'PC{i+1}' for i in range(10)]
            ax_s.bar(pcs_labels, explained_variance[:10], color='#1e3d59', alpha=0.7, label='Phương sai riêng lẻ')
            ax_s.plot(pcs_labels, np.cumsum(explained_variance[:10]), marker='o', color='#e74c3c', label='Phương sai tích lũy')
            ax_s.set_title("Scree Plot: Mức độ giải thích của các PC")
            ax_s.set_ylabel("% Phương sai")
            ax_s.legend()
            st.pyplot(fig_s)

        elif selected_q == questions[4]:
            st.write(f"**Phân tích:** PC1 đã chiếm đa số phương sai. PC2 ({explained_variance[1]:.2f}%) và PC3 ({explained_variance[2]:.2f}%) thường đại diện cho sự đối lập giữa các nhóm ngành (ví dụ: dòng tiền rút từ Bất động sản sang Ngân hàng) hoặc các cú sốc chỉ ảnh hưởng đến một nhóm cổ phiếu cụ thể.")

        elif selected_q == questions[5]:
            st.markdown("""
            **Ứng dụng thực tiễn:**
            1. **Xây dựng danh mục tối ưu:** Tập trung vào các mã có Loading PC1 thấp nếu muốn giảm thiểu rủi ro thị trường.
            2. **Phòng vệ rủi ro (Hedging):** Sử dụng PC1 như một chỉ báo sớm để điều chỉnh tỷ trọng danh mục trước các biến động lớn.
            3. **Đa dạng hóa:** PCA giúp nhận diện các cổ phiếu không đi cùng hướng với đám đông (Loadings thấp hoặc âm).
            """)

else:
    st.error("❌ **Lỗi hệ thống:** Không tìm thấy dữ liệu hoặc cấu trúc file không đúng.")
    st.warning("Vui lòng kiểm tra các file 'Dữ liệu Lịch sử [Mã CK].csv' và 'Dữ liệu Lịch sử VN 30.csv' trong thư mục gốc.")

st.markdown('<div class="footer">Đồ án Phân tích dữ liệu Tài chính - HUB | Nhóm 1 | © 2026</div>', unsafe_allow_html=True)
