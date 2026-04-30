import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="VN30 PCA Dashboard | HUB Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ĐỂ LÀM ĐẸP UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border-radius: 10px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: grey;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        border-top: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: THÔNG TIN TÁC GIẢ & CẤU HÌNH ---
with st.sidebar:
    st.image("https://hub.edu.vn/Images/logo-hub.png", width=150) # Link logo HUB nếu có
    st.title("⚙️ Cấu hình & Thông tin")
    
    st.info("""
    **Sinh viên thực hiện:**
    - Dần Thị Trúc Xinh
    - Nhóm nghiên cứu HUB
    """)
    
    st.divider()
    st.markdown("### 📁 Quản lý dữ liệu")
    # Mặc định là "." để đọc ngay tại thư mục gốc nếu bạn không tạo folder data
    data_path = st.text_input("Đường dẫn thư mục CSV", value=".")
    
    if st.button("🔄 Làm mới dữ liệu"):
        st.cache_data.clear()
        st.success("Đã xóa bộ nhớ đệm!")

# --- HÀM XỬ LÝ DỮ LIỆU (MÔ PHỎNG PCA FROM SCRATCH) ---
@st.cache_data
def load_and_process_data(folder_path):
    if not os.path.exists(folder_path):
        return None, None
        
    # Lấy danh sách file, loại bỏ app.py hoặc các file csv không phải dữ liệu nếu có
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f != 'requirements.txt']
    
    if not all_files:
        return None, None
    
    prices = pd.DataFrame()
    for file in all_files:
        ticker = file.replace('.csv', '')
        try:
            # Đọc file, giả định cột đầu là ngày (index)
            df = pd.read_csv(os.path.join(folder_path, file), index_col=0, parse_dates=True)
            if 'Close' in df.columns:
                prices[ticker] = df['Close']
        except Exception:
            continue
    
    if prices.empty:
        return None, None
        
    prices = prices.ffill().dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return prices, log_returns

def perform_pca_from_scratch(returns_df):
    # 1. Chuẩn hóa dữ liệu (Standardize)
    std_returns = (returns_df - returns_df.mean()) / returns_df.std()
    
    # 2. Ma trận hiệp phương sai
    cov_matrix = np.cov(std_returns.T)
    
    # 3. Phân rã trị riêng (Eigen Decomposition)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    
    # 4. Sắp xếp
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    
    # Tính Variance Explained (Phần thực để tránh lỗi số phức nếu có nhiễu)
    explained_variance = np.real(eigen_values) / np.sum(np.real(eigen_values))
    
    # PC1 Scores
    pc1_loadings = np.real(eigen_vectors[:, 0])
    pc1_scores = std_returns @ pc1_loadings
    
    return explained_variance, pc1_loadings, pc1_scores

# --- GIAO DIỆN CHÍNH ---
st.title("📈 Đồ án: Phân tích Cấu trúc VN30 bằng PCA")
st.markdown("---")

# Tải dữ liệu
try:
    prices, returns = load_and_process_data(data_path)
    if prices is not None and not prices.empty:
        # Thực hiện tính toán
        var_exp, loadings, pc1_scores = perform_pca_from_scratch(returns)
        
        # --- ROW 1: METRIC CARDS ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Số lượng mã cổ phiếu", f"{len(returns.columns)} Mã")
        col2.metric("Mức độ giải thích PC1", f"{var_exp*100:.2f}%", help="Đại diện cho rủi ro hệ thống")
        
        # So sánh với trung bình thị trường (Mô phỏng VN30)
        market_returns = returns.mean(axis=1)
        vn30_sim = market_returns.cumsum() 
        pc1_cum = pc1_scores.cumsum()
        
        # Lấy giá trị tương quan (trích xuất giá trị từ ma trận)
        correlation = np.corrcoef(pc1_scores, market_returns)
        
        col3.metric("Tương quan PC1 vs VN30", f"{correlation:.4f}")
        col4.metric("Trạng thái hệ thống", "Ổn định", delta="Sẵn sàng")

        # --- TABS PHÂN TÍCH ---
        tab1, tab2, tab3 = st.tabs(["📊 Tổng quan Dữ liệu", "🧬 Kết quả PCA", "🔍 Phân tích Sâu"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Ma trận Giá đóng cửa (Raw)")
                st.dataframe(prices.tail(5), use_container_width=True)
            with c2:
                st.subheader("Ma trận Tỷ suất sinh lợi (Log)")
                st.dataframe(returns.tail(5), use_container_width=True)
            
            # Chuẩn hóa giá về gốc 100 để so sánh
            norm_prices = (prices / prices.iloc * 100)
            fig_prices = px.line(norm_prices, title="Diễn biến giá VN30 (Quy đổi về gốc 100)", labels={"value": "Chỉ số", "index": "Ngày"})
            fig_prices.update_layout(showlegend=False)
            st.plotly_chart(fig_prices, use_container_width=True)

        with tab2:
            col_a, col_b = st.columns()
            
            with col_a:
                st.subheader("So sánh Lợi suất Tích lũy: PC1 vs VN30-Index")
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=pc1_cum.index, y=pc1_cum, name="PC1 (Danh mục PCA)", line=dict(color='firebrick', width=2)))
                fig_comp.add_trace(go.Scatter(x=vn30_sim.index, y=vn30_sim, name="VN30-Index (Mô phỏng)", line=dict(color='royalblue', width=2, dash='dot')))
                fig_comp.update_layout(hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                st.plotly_chart(fig_comp, use_container_width=True)
            
            with col_b:
                st.subheader("Tỷ lệ giải thích phương sai")
                # Hiển thị 10 PC đầu tiên
                num_pc = min(10, len(var_exp))
                exp_df = pd.DataFrame({'PC': [f'PC{i+1}' for i in range(num_pc)], 'Variance': var_exp[:num_pc]})
                fig_var = px.bar(exp_df, x='PC', y='Variance', text_auto='.2%', color='Variance', color_continuous_scale='Blues')
                st.plotly_chart(fig_var, use_container_width=True)

        with tab3:
            st.subheader("Cấu trúc dẫn dắt thị trường (Factor Loadings)")
            loadings_df = pd.DataFrame({'Ticker': returns.columns, 'Weight': loadings}).sort_values('Weight', ascending=False)
            
            fig_loadings = px.bar(loadings_df, x='Weight', y='Ticker', orientation='h', 
                                 title="Trọng số đóng góp của các cổ phiếu vào PC1",
                                 color='Weight', color_continuous_scale='RdBu_r', height=800)
            fig_loadings.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_loadings, use_container_width=True)
            
            with st.expander("💡 Giải thích ý nghĩa tài chính"):
                st.write("""
                - **Hệ số tải (Loadings):** Cho biết mức độ nhạy cảm của từng cổ phiếu với biến động chung của thị trường.
                - **Nhóm dẫn dắt:** Các mã nằm ở phía trên (trọng số cao) là những đầu tàu kéo chỉ số.
                - **Tính đồng biến:** Nếu tất cả các mã đều có trọng số dương, thị trường đang có tính liên kết cực kỳ chặt chẽ.
                """)

    else:
        st.warning("⚠️ Không tìm thấy file dữ liệu CSV. Nếu bạn upload lên GitHub thư mục gốc, hãy để trống ô 'Đường dẫn thư mục CSV' hoặc nhập '.'")

except Exception as e:
    st.error(f"❌ Lỗi hệ thống: {e}")
    st.info("Mẹo: Đảm bảo các file CSV có cột 'Close' và định dạng ngày tháng đúng.")

# --- FOOTER ---
st.markdown("""
    <div class="footer">
        Đồ án Phân tích dữ liệu Tài chính - Đại học Ngân hàng TP.HCM (HUB) | © 2024
    </div>
    """, unsafe_allow_html=True)
