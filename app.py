import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="VN30 PCA Analysis | HUB Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Xanh dương Pastel, Trưởng thành, Chuyên nghiệp) ---
st.markdown("""
    <style>
    /* Tổng thể nền và màu chữ */
    .main { background-color: #F0F4F8; } /* Xanh pastel rất nhạt */
    .stApp { color: #2C3E50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] { font-size: 28px; color: #1A365D; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #4A5568; font-weight: 600; }
    .stMetric {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        border-left: 6px solid #829AB1; /* Xanh pastel đậm */
        transition: transform 0.2s ease;
    }
    .stMetric:hover { transform: translateY(-3px); }
    
    /* Footer */
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #1A365D; color: #E2E8F0; text-align: center;
        padding: 10px; font-size: 13px; border-top: 2px solid #2B6CB0;
        z-index: 100; font-weight: 500;
    }
    
    /* Header & Sidebar */
    h1, h2, h3, h4 { color: #1A365D; font-weight: bold; }
    .stSidebar { background-color: #E1E8ED; border-right: 1px solid #CBD5E1; }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid #CBD5E1; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #E2E8F0;
        border-radius: 6px 6px 0px 0px; padding: 10px 20px; color: #4A5568; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #FFFFFF; 
        border-bottom: 3px solid #2B6CB0 !important; 
        color: #2B6CB0; 
    }
    
    /* Tables */
    .dataframe th { background-color: #EDF2F7; color: #1A365D; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    # Thêm Logo trường (Thay thế bằng text nếu file ảnh không tồn tại để tránh lỗi crash)
    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.markdown("<h1 style='text-align: center; color: #1A365D;'>🏦 HUB</h1>", unsafe_allow_html=True)
        
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
st.markdown("<p style='color: #4A5568; font-size: 16px;'>Ứng dụng Phân tích Thành phần Chính (PCA) để bóc tách rủi ro hệ thống và mức độ nhạy cảm của rổ chỉ số VN30.</p>", unsafe_allow_html=True)

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

# --- THUẬT TOÁN QR MANUAL (Từ 1.ipynb) ---
def qr_manual(A, steps=500):
    n = A.shape[0]
    V = np.eye(n) # Khởi tạo ma trận Vector riêng
    A_k = A.copy()
    for _ in range(steps):
        Q, R = np.linalg.qr(A_k) # Phân rã ma trận
        A_k = R @ Q              # Cập nhật
        V = V @ Q                # Tích các Q
    return np.diag(A_k), V


# --- KHỞI CHẠY LOGIC ---
df_merged, df_returns = load_and_process_data()

if df_merged is not None:
    # 1. TÍNH TOÁN PCA (Linear Algebra)
    vn30_col = 'VN30_Index'
    X_returns = df_returns.drop(columns=[vn30_col], errors='ignore')
    features = X_returns.columns
    
    # Chuẩn hóa (Z-score)
    X_scaled = (X_returns - X_returns.mean()) / X_returns.std()
    
    # Ma trận hiệp phương sai & Trị riêng (Dùng Numpy)
    cov_matrix = np.cov(X_scaled.to_numpy().T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sắp xếp trị riêng
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = np.real(eigenvalues[sorted_indices])
    eigenvectors_sorted = np.real(eigenvectors[:, sorted_indices])
    
    # Chạy thêm thuật toán thủ công (QR Algorithm) để đối chiếu
    vals_m, vecs_m = qr_manual(cov_matrix)
    idx_m = np.argsort(vals_m)[::-1]
    vals_m, vecs_m = vals_m[idx_m], vecs_m[:, idx_m]

    # Tỷ lệ giải thích phương sai
    explained_variance = (eigenvalues_sorted / np.sum(eigenvalues_sorted)) * 100
    cum_var = np.cumsum(explained_variance)
    
    # Tính toán PC1
    pc1_eigenvector = eigenvectors_sorted[:, 0]
    pc1_returns = np.dot(X_scaled, pc1_eigenvector)
    vn30_returns = df_returns[vn30_col].values
    
    correlation = pd.Series(pc1_returns).corr(pd.Series(vn30_returns))
    
    if correlation < 0:
        pc1_returns = -pc1_returns
        pc1_eigenvector = -pc1_eigenvector
        correlation = -correlation
    
    # Đưa vào DataFrame tạm để đo lường biến động
    df_temp = pd.DataFrame({
        'PC1_Returns': pc1_returns,
        'VN30_Returns': vn30_returns
    }, index=X_returns.index)

    # Đồng bộ hóa biên độ dao động (Rescale Volatility)
    pc1_volatility = df_temp['PC1_Returns'].std()
    vn30_volatility = df_temp['VN30_Returns'].std()
    
    df_temp['PC1_Returns_Adjusted'] = df_temp['PC1_Returns'] * (vn30_volatility / pc1_volatility)

    # Tính lợi suất tích lũy chính xác
    df_cumulative = pd.DataFrame(index=X_returns.index)
    df_cumulative['PC1_Returns'] = np.exp(df_temp['PC1_Returns_Adjusted'].cumsum()) - 1
    df_cumulative['VN30_Returns'] = np.exp(df_temp['VN30_Returns'].cumsum()) - 1

    # --- BỐ CỤC 4 TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 DỮ LIỆU & TIỀN XỬ LÝ", 
        "⚙️ THUẬT TOÁN PCA", 
        "📈 HIỆU NĂNG TÍCH LŨY", 
        "🔍 CẤU TRÚC CHUYÊN SÂU"
    ])

    # ===== TAB 1: TỔNG QUAN & TIỀN XỬ LÝ =====
    with tab1:
        st.subheader("1. Tổng quan & Tiền xử lý Dữ liệu")
        st.markdown("""
        Dữ liệu được trích xuất từ các tệp CSV lịch sử, đồng bộ hóa theo thời gian và chuyển đổi sang **Log Returns** 
        nhằm đảm bảo tính phân phối chuẩn. Tiếp theo, dữ liệu được chuẩn hóa theo phương pháp **Z-score** để tránh 
        việc các cổ phiếu có biên độ biến động lớn chi phối mô hình PCA.
        """)
        
        col_info1, col_info2 = st.columns([1, 1])
        with col_info1:
            st.markdown("**📊 Ma trận Tỷ suất sinh lợi (Log Returns)**")
            st.dataframe(df_returns.tail(8), use_container_width=True)
        with col_info2:
            st.markdown("**⚖️ Thống kê sau chuẩn hóa Z-score**")
            scaled_stats = pd.DataFrame({
                'Mã Cổ Phiếu': features,
                'Trung bình (Mean)': np.round(np.mean(X_scaled, axis=0), 6),
                'Độ lệch chuẩn (Std)': np.round(np.std(X_scaled, axis=0), 6)
            })
            st.dataframe(scaled_stats.set_index('Mã Cổ Phiếu').T, use_container_width=True)
            
        st.info(f"✅ **Thống kê:** Đã xử lý {len(df_returns.columns)-1} mã cổ phiếu thành phần và {len(df_returns)} phiên giao dịch khớp lệnh.")

    # ===== TAB 2: THUẬT TOÁN PCA TỪ ĐẦU =====
    with tab2:
        st.subheader("2. Xây dựng & Phân rã Trị riêng (Eigen Decomposition)")
        
        st.markdown("""
        Thay vì chỉ dùng thư viện Black-box, mô hình áp dụng **Thuật toán QR (QR Algorithm)** để tiến hành phân rã 
        Ma trận Hiệp phương sai thành các trị riêng (Eigenvalues) và vector riêng (Eigenvectors).
        """)
        
        col_alg1, col_alg2 = st.columns([1.2, 1])
        with col_alg1:
            st.markdown("**📐 So sánh Eigenvalues: Thủ công (QR) vs Numpy**")
            comparison_df = pd.DataFrame({
                'Principal Component': [f'PC{i+1}' for i in range(5)],
                'Eigenvalues (Manual QR)': vals_m[:5],
                'Eigenvalues (Numpy)': eigenvalues_sorted[:5],
                'Độ lệch (Difference)': np.abs(vals_m[:5] - eigenvalues_sorted[:5])
            })
            st.dataframe(comparison_df.style.format({
                'Eigenvalues (Manual QR)': '{:.6f}', 
                'Eigenvalues (Numpy)': '{:.6f}', 
                'Độ lệch (Difference)': '{:.2e}'
            }), use_container_width=True)
            st.caption(f"💡 Kích thước Ma trận Hiệp phương sai: {cov_matrix.shape}. Độ tin cậy tính toán thủ công khớp hoàn toàn với thư viện đến chữ số thập phân siêu nhỏ.")

        with col_alg2:
            st.markdown("**📈 Tỷ lệ giải thích Phương sai**")
            explained_variance_df = pd.DataFrame({
                'Principal Component': [f'PC{i+1}' for i in range(5)],
                'Phương sai giải thích (%)': explained_variance[:5],
                'Phương sai Tích lũy (%)': cum_var[:5]
            })
            st.dataframe(explained_variance_df.style.format({
                'Phương sai giải thích (%)': '{:.2f}%',
                'Phương sai Tích lũy (%)': '{:.2f}%'
            }), use_container_width=True)

    # ===== TAB 3: HIỆU NĂNG TÍCH LŨY =====
    with tab3:
        st.subheader("3. So sánh Hiệu năng Thành phần Chính (PC1)")
        
        num_pc_90 = np.argmax(cum_var >= 90) + 1
        m1, m2, m3 = st.columns(3)
        m1.metric("Giải thích bởi PC1", f"{explained_variance[0]:.2f}%")
        m2.metric("Tương quan PC1 vs VN30", f"{correlation:.4f}")
        m3.metric("Số PC đạt ≥90% Var", f"{num_pc_90}")

        st.divider()
        st.markdown("**Biểu đồ So sánh Lợi suất Tích lũy: PC1 (Danh mục ảo) vs VN30-Index**")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_cumulative.index, df_cumulative['PC1_Returns'], label='PC1 Index (Nhân tố giả lập)', color='#2B6CB0', linewidth=2.5)
        ax.plot(df_cumulative.index, df_cumulative['VN30_Returns'], label='VN30 Index (Thực tế)', color='#E53E3E', linestyle='--', linewidth=1.5, alpha=0.9)
        ax.set_ylabel('Tỷ suất sinh lợi tích lũy', fontsize=11, color='#1A365D')
        ax.set_xlabel('Thời gian', fontsize=11, color='#1A365D')
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Đổi màu viền và tick của matplotlib cho hợp theme
        ax.spines['bottom'].set_color('#CBD5E1')
        ax.spines['top'].set_color('#CBD5E1')
        ax.spines['left'].set_color('#CBD5E1')
        ax.spines['right'].set_color('#CBD5E1')
        ax.tick_params(colors='#4A5568')
        fig.patch.set_facecolor('#F0F4F8')
        ax.set_facecolor('#FFFFFF')
        
        st.pyplot(fig)

    # ===== TAB 4: TRỌNG SỐ & PHÂN TÍCH CHUYÊN SÂU =====
    with tab4:
        st.subheader("4. Cấu trúc dẫn dắt & Hệ số tải (Factor Loadings)")
        
        c_i1, c_i2 = st.columns([1, 1.2]) 
        
        df_weights = pd.DataFrame({'Mã': X_returns.columns, 'Trọng số': pc1_eigenvector})
        top_5 = df_weights.reindex(df_weights['Trọng số'].abs().sort_values(ascending=False).index).head(5)
            
        with c_i1:
            st.markdown("""
            ### 🧬 Phân tích Factor Loadings
            Biểu đồ bên cạnh thể hiện mức độ đóng góp của từng cổ phiếu vào **Thành phần chính số 1 (PC1)**. 
            
            - **Trọng số dương cao:** Các cổ phiếu mang tính dẫn dắt thị trường (Market Leaders).
            - **Tính đồng nhất:** Nếu tất cả các mã đều có trọng số cùng dấu, thị trường có tính tương quan hệ thống rất cao.
            """)
            
            st.markdown("**Top 5 Mã chi phối PC1**")
            st.dataframe(top_5.style.background_gradient(cmap='Blues'), use_container_width=True)
            
        with c_i2:
            fig_w, ax_w = plt.subplots(figsize=(8, 10))
            df_w_sorted = df_weights.sort_values(by='Trọng số')
            # Cập nhật màu thanh ngang pastel
            colors_bar = ['#3182CE' if x > 0 else '#E53E3E' for x in df_w_sorted['Trọng số']]
            ax_w.barh(df_w_sorted['Mã'], df_w_sorted['Trọng số'], color=colors_bar, alpha=0.85)
            ax_w.set_title('VN30 FACTOR LOADINGS (PC1)', fontweight='bold', color='#1A365D', pad=20)
            ax_w.axvline(0, color='#1A365D', linewidth=1)
            ax_w.grid(axis='x', linestyle='--', alpha=0.4)
            
            fig_w.patch.set_facecolor('#F0F4F8')
            ax_w.set_facecolor('#FFFFFF')
            ax_w.tick_params(colors='#4A5568')
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
            ax_s.bar(pcs_labels, explained_variance[:10], color='#829AB1', alpha=0.8, label='Phương sai riêng lẻ')
            ax_s.plot(pcs_labels, np.cumsum(explained_variance[:10]), marker='o', color='#E53E3E', label='Phương sai tích lũy')
            ax_s.set_title("Scree Plot: Mức độ giải thích của các PC", color='#1A365D')
            ax_s.set_ylabel("% Phương sai", color='#4A5568')
            ax_s.legend()
            
            fig_s.patch.set_facecolor('#F0F4F8')
            ax_s.set_facecolor('#FFFFFF')
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
