"""
Bài 6 — TOPSIS xếp hạng 6 vùng đầu tư phát triển AI
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Shared helpers ────────────────────────────────────────────────────────────
def section(title, icon="📌"):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(0,212,255,.07),rgba(99,102,241,.05));
         border:1px solid rgba(0,212,255,.14);border-radius:10px;padding:.7rem 1.2rem;margin:.8rem 0 .5rem;">
      <span style="color:#00D4FF;font-size:1.05rem;font-weight:700;">{icon} {title}</span>
    </div>""", unsafe_allow_html=True)

def math_box(txt):
    st.markdown(f'<div class="math-box">{txt}</div>', unsafe_allow_html=True)

def ai_agent(text):
    state_key = f"ai_agent_visible_{__name__.replace('.', '_')}"
    if st.button("🤖 Phân tích bằng AI Agent", key=f"{state_key}_button", use_container_width=True):
        st.session_state[state_key] = True

    if not st.session_state.get(state_key, False):
        st.markdown("""
        <div class="agent-teaser">
          <div class="agent-title">🤖 AI Agent sẵn sàng phân tích</div>
          <div class="agent-body">Bấm nút <b>Phân tích bằng AI Agent</b> để hiển thị nhận xét chính sách mô phỏng cho kết quả hiện tại.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    with st.spinner("AI Agent đang phân tích kết quả mô hình..."):
        import time
        time.sleep(0.35)

    st.markdown(f"""
    <div class="ai-agent-card">
      <div class="agent-title">🤖 AI Policy Agent — Phân tích kết quả</div>
      <div class="agent-body">{text}</div>
    </div>
    """, unsafe_allow_html=True)
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,32,0.6)",
    font=dict(color="#94A3B8", family="Inter"),
    xaxis=dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.1)"),
    margin=dict(l=40, r=20, t=50, b=40),
)

COLORS = ["#00D4FF","#6366F1","#10B981","#F59E0B","#EC4899","#8B5CF6","#3B82F6"]

# ── Raw Data ──────────────────────────────────────────────────────────────────
regions = [
    "Trung du miền núi phía Bắc",
    "Đồng bằng sông Hồng",
    "Bắc Trung Bộ + DH Trung Bộ",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long"
]

df_raw = pd.DataFrame({
    "region_name_vi": regions,
    "grdp_per_capita_million_VND": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
    "fdi_registered_billion_USD": [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
    "digital_index_0_100": [38, 78, 55, 32, 82, 48],
    "ai_readiness_0_100": [22, 68, 40, 18, 75, 30],
    "trained_labor_pct": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
    "rd_intensity_pct": [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
    "internet_penetration_pct": [72, 92, 84, 68, 94, 78],
    "gini_coef": [0.405, 0.358, 0.372, 0.412, 0.385, 0.392]
})

criteria = [
    "grdp_per_capita_million_VND", "fdi_registered_billion_USD", "digital_index_0_100",
    "ai_readiness_0_100", "trained_labor_pct", "rd_intensity_pct", "internet_penetration_pct", "gini_coef"
]
criteria_vi = [
    "GRDP/người", "FDI đăng ký", "Digital Index", "AI Readiness", "Lao động qua đào tạo", "Cường độ R&D", "Phổ cập Internet", "Hệ số Gini"
]
is_benefit = np.array([True, True, True, True, True, True, True, False])

# ── TOPSIS Algorithm ──────────────────────────────────────────────────────────
def run_topsis(X, weights, is_benefit):
    # Vector Normalization
    norm_denom = np.sqrt((X ** 2).sum(axis=0))
    R = X / np.where(norm_denom == 0, 1.0, norm_denom)
    
    # Weighted normalized decision matrix
    V = R * weights
    
    # Identify positive & negative ideal solutions
    A_star = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    
    # Calculate Euclidean distances
    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    
    # Relative closeness to ideal
    C_star = S_neg / (S_star + S_neg)
    
    return C_star, S_star, S_neg

# ── Entropy Weights Calculation ───────────────────────────────────────────────
def run_entropy(X):
    # Sum normalization
    P = X / X.sum(axis=0)
    
    # Entropy calculation
    n = X.shape[0]
    k = 1.0 / np.log(n)
    P = np.where(P == 0, 1e-12, P)  # avoid log(0)
    E = -k * (P * np.log(P)).sum(axis=0)
    
    # Dispersion degree
    D_disp = 1.0 - E
    
    # Entropy weights
    w_ent = D_disp / D_disp.sum()
    return w_ent

# ── AHP Weights Model ────────────────────────────────────────────────────────
def run_ahp(preference_ai_vs_labor):
    # Simple 2-criteria priority simulation scaled to 8 indicators
    # 3 stands for AI & Labor high, others low.
    w_ahp = np.array([0.08, 0.08, 0.12, 0.20, 0.16, 0.16, 0.08, 0.12])
    
    # Adjust AI vs Labor weight based on slider
    # pref: 1 stands for equal (0.20, 0.16), 5 stands for AI high, -5 for Labor high
    if preference_ai_vs_labor > 0:
        w_ahp[3] += 0.02 * preference_ai_vs_labor
        w_ahp[4] -= 0.015 * preference_ai_vs_labor
    elif preference_ai_vs_labor < 0:
        w_ahp[3] -= 0.015 * abs(preference_ai_vs_labor)
        w_ahp[4] += 0.02 * abs(preference_ai_vs_labor)
        
    # Re-normalize
    w_ahp = np.clip(w_ahp, 0.01, 1.0)
    return w_ahp / w_ahp.sum()

def show():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
      <div style="display:flex;align-items:center;gap:1rem;">
        <span style="font-size:2rem;">🌍</span>
        <div>
          <div style="color:#00D4FF;font-size:.75rem;font-weight:600;letter-spacing:1px;">BÀI 6</div>
          <div style="color:#E2E8F0;font-size:1.5rem;font-weight:800;">TOPSIS xếp hạng 6 Vùng kinh tế ưu tiên đầu tư phát triển AI</div>
          <div style="margin-top:.3rem;">
            <span class="badge b-blue">Multi-Criteria TOPSIS</span>
            <span class="badge b-purple">Entropy Weighting</span>
            <span class="badge b-green">Analytic Hierarchy Process (AHP)</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📖 Bối cảnh & Mô hình", "🎛️ Tham số", "📊 Kết quả", "📈 Phân tích chính sách", "🤖 AI Agent"])

    # ══ Tab 1: Bối cảnh & Mô hình ══════════════════════════════════════════
    with tabs[0]:
        section("Bối cảnh bài toán", "🌏")
        st.markdown("""
        Theo **Chiến lược quốc gia về nghiên cứu, phát triển và ứng dụng AI đến năm 2030**, Việt Nam đặt mục tiêu trở thành trung tâm 
        đổi mới sáng tạo hàng đầu của khu vực ASEAN.
        
        Tuy nhiên, do nguồn lực phân bổ ngân sách hạ tầng và dữ liệu quốc gia cho AI là hữu hạn, việc xác định vùng nào nên được 
        ưu tiên triển khai trung tâm AI và sandbox dữ liệu trước là tối quan trọng.
        
        Bài toán này sử dụng phương pháp **TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)** 
        để xếp hạng 6 vùng kinh tế - xã hội dựa trên 8 chỉ tiêu phát triển đa chiều.
        """)

        section("Lý thuyết phương pháp TOPSIS", "📐")
        math_box("""
<b>Bước 1. Chuẩn hóa ma trận quyết định bằng Vector Normalization:</b>
  r<sub>ij</sub> = x<sub>ij</sub> / √(Σ<sub>i=1..m</sub> x<sub>ij</sub><sup>2</sup>)

<b>Bước 2. Tính ma trận chuẩn hóa có trọng số:</b>
  v<sub>ij</sub> = w<sub>j</sub> · r<sub>ij</sub>   (Với w<sub>j</sub> là trọng số tiêu chí j)

<b>Bước 3. Xác định Phương án Lý tưởng Dương (A*) và Lý tưởng Âm (A-):</b>
  • Tiêu chí lợi ích:   A<sub>j</sub>* = max<sub>i</sub>(v<sub>ij</sub>),   A<sub>j</sub>- = min<sub>i</sub>(v<sub>ij</sub>)
  • Tiêu chí chi phí (Gini): A<sub>j</sub>* = min<sub>i</sub>(v<sub>ij</sub>),   A<sub>j</sub>- = max<sub>i</sub>(v<sub>ij</sub>)

<b>Bước 4. Tính khoảng cách Euclide tới các cực lý tưởng:</b>
  S<sub>i</sub>* = √(Σ<sub>j</sub> (v<sub>ij</sub> - A<sub>j</sub>*)<sup>2</sup>)   và   S<sub>i</sub>- = √(Σ<sub>j</sub> (v<sub>ij</sub> - A<sub>j</sub>-)<sup>2</sup>)

<b>Bước 5. Tính hệ số gần gũi lý tưởng C<sub>i</sub>*:</b>
  C<sub>i</sub>* = S<sub>i</sub>- / (S<sub>i</sub>* + S<sub>i</sub>-)     (0 ≤ C<sub>i</sub>* ≤ 1)   → Điểm C<sub>i</sub>* càng cao, thứ hạng vùng càng ưu tiên.
        """)

    # ══ Tab 2: Xác định Trọng số ══════════════════════════════════════════
    with tabs[1]:
        section("Lựa chọn phương pháp xác định trọng số w_j", "🎛️")
        w_method = st.radio("Phương pháp:", [
            "1. Trọng số Chuyên gia (Expert Weights)",
            "2. Trọng số Entropy tự động (Entropy Weights)",
            "3. Trọng số AHP điều chỉnh cặp (AHP Weights)"
        ], key="b6_wm")

        X = df_raw[criteria].values.astype(float)
        
        if "Chuyên gia" in w_method:
            st.markdown("**Bộ trọng số Chuyên gia tự thiết kế qua sliders** *(tự quy chuẩn về 1)*")
            col1, col2 = st.columns(2)
            with col1:
                w_grdp = st.slider("GRDP/người", 0.05, 0.30, 0.10, 0.01, key="b6_w1")
                w_fdi = st.slider("FDI đăng ký", 0.05, 0.30, 0.10, 0.01, key="b6_w2")
                w_dig = st.slider("Digital Index", 0.05, 0.30, 0.15, 0.01, key="b6_w3")
                w_ai = st.slider("AI Readiness", 0.05, 0.30, 0.20, 0.01, key="b6_w4")
            with col2:
                w_lab = st.slider("LĐ qua đào tạo", 0.05, 0.30, 0.15, 0.01, key="b6_w5")
                w_rd = st.slider("Cường độ R&D", 0.05, 0.30, 0.15, 0.01, key="b6_w6")
                w_net = st.slider("Phổ cập Internet", 0.01, 0.15, 0.05, 0.01, key="b6_w7")
                w_gini = st.slider("Chỉ số Gini (Cost)", 0.05, 0.25, 0.10, 0.01, key="b6_w8")
                
            w_arr = np.array([w_grdp, w_fdi, w_dig, w_ai, w_lab, w_rd, w_net, w_gini])
            w_final = w_arr / w_arr.sum()
            
        elif "Entropy" in w_method:
            w_final = run_entropy(X)
            st.success("🤖 **Trọng số Entropy tự động tính toán thành công** dựa trên mức độ biến động (độ hỗn loạn) của dữ liệu thực tế!")
            
        else:
            st.markdown("**Trọng số AHP: Đánh giá tầm quan trọng tương đối giữa AI Readiness và Lao động**")
            ahp_pref = st.slider("Độ ưu tiên: AI Readiness cực lớn (-5) ↔ Cân bằng (0) ↔ Lao động cực lớn (5)", -5, 5, 0, 1, key="b6_ahp_p")
            w_final = run_ahp(ahp_pref)
            
        # Display weights
        section("Trọng số cuối cùng áp dụng cho TOPSIS", "⚖️")
        w_df = pd.DataFrame({"Tiêu chí": criteria_vi, "Trọng số (w_j)": w_final, "Định hướng": ["Lợi ích" if b else "Chi phí (Gini)" for b in is_benefit]})
        st.dataframe(w_df.style.format({"Trọng số (w_j)": "{:.4f}"}), use_container_width=True)

        # Plotly chart of weights
        fig_w = px.bar(w_df, x="Tiêu chí", y="Trọng số (w_j)", color="Định hướng", color_discrete_sequence=[COLORS[0], COLORS[4]])
        fig_w.update_layout(**PLOTLY_LAYOUT, height=350)
        st.plotly_chart(fig_w, use_container_width=True)

    # ══ Tab 3: Kết quả xếp hạng ════════════════════════════════════════════
    with tabs[2]:
        C_star, S_star, S_neg = run_topsis(X, w_final, is_benefit)
        
        df_res = df_raw.copy()
        df_res["TOPSIS Score (C*)"] = C_star
        df_res["S+ (Ideal dist)"] = S_star
        df_res["S- (Anti-ideal dist)"] = S_neg
        df_res = df_res.sort_values("TOPSIS Score (C*)", ascending=False).reset_index(drop=True)
        df_res["Xếp hạng"] = df_res.index + 1
        
        c1, c2 = st.columns(2)
        with c1:
            section("Câu 6.4.1 — TOPSIS với trọng số chuyên gia/người dùng", "🏆")
            st.dataframe(df_res[["Xếp hạng", "region_name_vi", "TOPSIS Score (C*)", "S+ (Ideal dist)", "S- (Anti-ideal dist)"]].style.format({
                "TOPSIS Score (C*)": "{:.4f}",
                "S+ (Ideal dist)": "{:.4f}",
                "S- (Anti-ideal dist)": "{:.4f}"
            }), use_container_width=True)
        with c2:
            section("Biểu đồ điểm TOPSIS C*", "📈")
            fig_score = go.Figure(go.Bar(
                x=df_res["TOPSIS Score (C*)"][::-1],
                y=df_res["region_name_vi"][::-1],
                orientation='h',
                marker_color=COLORS[1],
                text=df_res["TOPSIS Score (C*)"][::-1].round(3).astype(str),
                textposition="outside"
            ))
            fig_score.update_layout(**PLOTLY_LAYOUT, height=380)
            st.plotly_chart(fig_score, use_container_width=True)

        section("Ma trận dữ liệu 6 vùng kinh tế gốc", "📋")
        st.dataframe(df_raw.style.format({
            "grdp_per_capita_million_VND": "{:.1f}",
            "fdi_registered_billion_USD": "{:.2f}",
            "digital_index_0_100": "{:.0f}",
            "ai_readiness_0_100": "{:.0f}",
            "trained_labor_pct": "{:.2f}%",
            "rd_intensity_pct": "{:.3f}%",
            "internet_penetration_pct": "{:.1f}%",
            "gini_coef": "{:.3f}"
        }), use_container_width=True)

    # ══ Kết quả mở rộng: so sánh trọng số ═════════════════════════════════
    with tabs[2]:
        section("Câu 6.4.2 — Trọng số Entropy và so sánh khách quan", "⚖️")
        
        # Run three models
        c_exp, _, _ = run_topsis(X, run_ahp(0), is_benefit) # Expert approximation
        c_ent, _, _ = run_topsis(X, run_entropy(X), is_benefit)
        c_ahp1, _, _ = run_topsis(X, run_ahp(-4), is_benefit) # AI high
        c_ahp2, _, _ = run_topsis(X, run_ahp(4), is_benefit) # Labor high
        
        df_multi = pd.DataFrame({
            "Vùng": regions,
            "Chuyên gia": c_exp,
            "Entropy tự động": c_ent,
            "AHP (AI cực ưu tiên)": c_ahp1,
            "AHP (LĐ qua đào tạo ưu tiên)": c_ahp2
        })
        
        # Rank conversion
        df_multi_rank = df_multi.copy()
        for col in ["Chuyên gia", "Entropy tự động", "AHP (AI cực ưu tiên)", "AHP (LĐ qua đào tạo ưu tiên)"]:
            df_multi_rank[col] = df_multi[col].rank(ascending=False, method="min").astype(int)
            
        st.dataframe(df_multi_rank, use_container_width=True)

        # Heatmap of ranks
        fig_heat = px.imshow(df_multi_rank.set_index("Vùng").T, text_auto=True, color_continuous_scale="Viridis_r")
        fig_heat.update_layout(**PLOTLY_LAYOUT, title="Heatmap thứ hạng 6 vùng theo các phương pháp trọng số")
        st.plotly_chart(fig_heat, use_container_width=True)

        section("Câu 6.4.3 — Phân tích độ nhạy theo trọng số AI Readiness", "📡")
        ai_focus = df_multi_rank[["Vùng", "AHP (AI cực ưu tiên)", "AHP (LĐ qua đào tạo ưu tiên)"]].copy()
        st.dataframe(ai_focus, use_container_width=True)
        section("Câu 6.4.4 — Mở rộng AHP đơn giản", "🧭")
        st.markdown("""
        AHP được dùng như một bộ trọng số chủ quan để kiểm tra mức nhạy của TOPSIS khi nhà hoạch định tăng ưu tiên cho AI hoặc nhân lực.
        Bảng heatmap ở trên cho thấy thứ hạng vùng có thể thay đổi khi ưu tiên chính sách thay đổi.
        """)

    # ══ Tab 4: Phân tích chính sách ═══════════════════════════════════════
    with tabs[3]:
        top1_r = df_res.iloc[0]["region_name_vi"]
        top2_r = df_res.iloc[1]["region_name_vi"]
        lowest_r = df_res.iloc[-1]["region_name_vi"]
        section("1. Vùng nào nên được ưu tiên đầu tư AI trước?", "📌")
        st.markdown(f"""
        Theo TOPSIS hiện tại, vùng dẫn đầu là **{top1_r}**, tiếp theo là **{top2_r}**.
        Hai vùng này gần phương án lý tưởng hơn nhờ nền tảng GRDP, FDI, digital index, AI readiness hoặc lao động qua đào tạo tốt hơn.
        Đây là cơ sở để đặt trung tâm AI, sandbox dữ liệu hoặc cụm thử nghiệm công nghệ cao.
        """)
        section("2. Có nên dùng Entropy thay cho trọng số chuyên gia?", "⚖️")
        st.markdown("""
        Entropy giúp giảm thiên lệch chủ quan vì tiêu chí nào phân hóa mạnh giữa các vùng sẽ có trọng số lớn hơn.
        Tuy nhiên, trọng số chuyên gia vẫn cần thiết để phản ánh mục tiêu chính trị như bao trùm vùng, an sinh xã hội và cân bằng phát triển.
        Do đó, kết quả tốt nhất nên đọc bằng nhiều bộ trọng số, thay vì chỉ chọn một bảng xếp hạng duy nhất.
        """)
        section("3. Vùng thấp nhất nên nhận chính sách gì?", "🧩")
        st.markdown(f"""
        Vùng có điểm thấp nhất là **{lowest_r}**, không nên bị loại khỏi chính sách AI.
        Thay vì đầu tư trung tâm AI quy mô lớn ngay, vùng này cần gói nền tảng: internet, dữ liệu hành chính, đào tạo kỹ năng số và hỗ trợ SME.
        Cách tiếp cận này giảm rủi ro lãng phí khi năng lực hấp thụ công nghệ chưa đủ.
        """)

    # ══ Tab 5: AI Agent ═══════════════════════════════════════════════════
    with tabs[4]:
        st.caption("Bấm nút AI Agent bên dưới để hiển thị phân tích chính sách mô phỏng.")
        top1_r = df_res.iloc[0]["region_name_vi"]
        top2_r = df_res.iloc[1]["region_name_vi"]
        lowest_r = df_res.iloc[-1]["region_name_vi"]
        
        ai_agent(f"""
        <b>1. Phân tích kết quả xếp hạng TOPSIS:</b><br>
        • Vùng đứng **đầu tiên** được ưu tiên đầu tư AI là <b>{top1_r}</b>, theo sau bởi <b>{top2_r}</b>.<br>
        • Vùng có mức độ ưu tiên **thấp nhất** là <b>{lowest_r}</b>.<br><br>
        
        <b>2. Sự khác biệt giữa các phương pháp trọng số:</b><br>
        • <b>Phương pháp Entropy</b>: Tự động phát hiện các tiêu chí có độ phân hóa cao (như giá trị FDI và đầu tư R&D) để nâng trọng số, 
        tránh sự thiên vị chủ quan của chuyên gia.<br>
        • <b>Phương pháp AHP</b>: Cho thấy sự nhạy cảm chính sách cao. Khi bạn tăng ưu tiên cho <i>Lao động qua đào tạo</i>, 
        các vùng có tỷ lệ đào tạo tốt (như Đồng bằng sông Hồng) sẽ bứt phá mạnh mẽ.<br><br>
        
        <b>3. Gợi ý hành động chính sách:</b><br>
        • <b>Đông Nam Bộ & ĐB Sông Hồng</b>: Nên là nơi đặt các Trung tâm AI Quốc gia, Sandbox dữ liệu mở vì có điểm sẵn sàng số và năng lực R&D vượt trội.<br>
        • <b>Tây Nguyên & Trung du miền núi phía Bắc</b>: Cần tập trung đầu tư cơ bản về phổ cập internet và nâng cấp chỉ số Digital Index trước khi tiến hành giải ngân đầu tư trực tiếp vào AI để tránh lãng phí.
        """)
