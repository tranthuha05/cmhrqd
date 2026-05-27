"""
Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và Số hóa
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

COLORS = ["#00D4FF","#6366F1","#10B981","#F59E0B","#EC4899","#8B5CF6"]


# ─────────────────────────────────────────────────────────────────────────────
def show():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
      <div style="display:flex;align-items:center;gap:1rem;">
        <span style="font-size:2rem;">📈</span>
        <div>
          <div style="color:#00D4FF;font-size:.75rem;font-weight:600;letter-spacing:1px;">BÀI 1</div>
          <div style="color:#E2E8F0;font-size:1.5rem;font-weight:800;">Hàm sản xuất Cobb–Douglas mở rộng với AI và Số hóa</div>
          <div style="margin-top:.3rem;">
            <span class="badge b-blue">numpy/pandas</span>
            <span class="badge b-purple">Growth Accounting</span>
            <span class="badge b-green">Dự báo 2030</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📖 Bối cảnh & Mô hình","📊 Dữ liệu & Tham số","📈 Kết quả","🔍 Phân tích chính sách","🤖 AI Agent"])

    # ══ Tab 1: Bối cảnh & Mô hình ══════════════════════════════════════════
    with tabs[0]:
        section("Bối cảnh bài toán", "🌏")
        st.markdown("""
        Trong giai đoạn **2020–2025**, kinh tế Việt Nam có sự chuyển dịch đáng chú ý theo hướng tăng trưởng
        dựa nhiều hơn vào khoa học công nghệ, chuyển đổi số và đổi mới sáng tạo.

        GDP Việt Nam tăng từ **8.044,4 nghìn tỷ VND** năm 2020 lên **12.847,6 nghìn tỷ VND** năm 2025.
        Đồng thời, tỷ trọng kinh tế số trong GDP tăng từ khoảng **12,0%** lên **19,5%**.

        Bài toán đặt ra là: nếu mô hình hóa nền kinh tế Việt Nam bằng hàm sản xuất Cobb–Douglas mở rộng,
        có bổ sung các yếu tố số hóa D, năng lực AI và vốn nhân lực số H, thì mô hình có mô phỏng
        sát GDP thực tế không? Và **yếu tố nào đóng góp lớn nhất** vào tăng trưởng GDP?
        """)

        section("Mô hình toán học", "📐")
        math_box("""
Y<sub>t</sub> = A<sub>t</sub> × K<sub>t</sub><sup>α</sup> × L<sub>t</sub><sup>β</sup> × D<sub>t</sub><sup>γ</sup> × AI<sub>t</sub><sup>δ</sup> × H<sub>t</sub><sup>θ</sup>
&nbsp;
Trong đó:
  Y<sub>t</sub>  : GDP, đơn vị nghìn tỷ VND        A<sub>t</sub>  : TFP (năng suất nhân tố tổng hợp)
  K<sub>t</sub>  : Vốn vật chất                     L<sub>t</sub>  : Lao động (triệu người)
  D<sub>t</sub>  : Mức độ số hóa (% kinh tế số/GDP) AI<sub>t</sub> : Năng lực AI (nghìn DN số)
  H<sub>t</sub>  : Vốn nhân lực số (% LĐ qua đào tạo)
&nbsp;
Phân rã tăng trưởng (Growth Accounting):
ΔlnY = ΔlnA + αΔlnK + βΔlnL + γΔlnD + δΔlnAI + θΔlnH
&nbsp;
Ràng buộc CRS: α + β + γ + δ + θ = 1
        """)

        math_box("""
Ước lượng TFP ngược:
  A<sub>t</sub> = Y<sub>t</sub> / (K<sub>t</sub><sup>α</sup> × L<sub>t</sub><sup>β</sup> × D<sub>t</sub><sup>γ</sup> × AI<sub>t</sub><sup>δ</sup> × H<sub>t</sub><sup>θ</sup>)
&nbsp;
Đánh giá dự báo — MAPE:
  MAPE = (1/n) × Σ |Y<sub>t</sub> - Ŷ<sub>t</sub>| / Y<sub>t</sub> × 100%
        """)

    # ══ Tab 2: Dữ liệu & Tham số ═══════════════════════════════════════════
    with tabs[1]:
        section("Điều chỉnh tham số hệ số sản xuất", "🎛️")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Bộ hệ số Cobb-Douglas** *(CRS: tổng = 1, θ được tính tự động)*")
            alpha = st.slider("α — Vốn vật chất K", 0.10, 0.60, 0.33, 0.01, key="b1_alpha")
            beta  = st.slider("β — Lao động L",     0.10, 0.60, 0.42, 0.01, key="b1_beta")
            gamma = st.slider("γ — Số hóa D",        0.01, 0.25, 0.10, 0.01, key="b1_gamma")
            delta = st.slider("δ — Năng lực AI",     0.01, 0.20, 0.08, 0.01, key="b1_delta")
            theta = round(1 - alpha - beta - gamma - delta, 4)

        with col2:
            if theta < 0:
                st.error(f"⚠️ θ = **{theta:.4f}** < 0 — Tổng hệ số vượt 1, vui lòng giảm các tham số!")
                theta = 0.01
            else:
                st.success(f"**θ (Nhân lực số H) = {theta:.4f}** *(tự động từ CRS)*")
                st.markdown(f"""
                | Tham số | Giá trị | Ý nghĩa |
                |---------|---------|---------|
                | α | **{alpha:.2f}** | Hệ số co dãn của vốn K |
                | β | **{beta:.2f}** | Hệ số co dãn của lao động L |
                | γ | **{gamma:.2f}** | Hệ số co dãn của số hóa D |
                | δ | **{delta:.2f}** | Hệ số co dãn của AI |
                | θ | **{theta:.4f}** | Hệ số co dãn của nhân lực số H |
                | **Tổng** | **{alpha+beta+gamma+delta+theta:.4f}** | = 1 (CRS) |
                """)

        section("Dữ liệu giai đoạn 2020–2025", "📋")
        year = np.array([2020,2021,2022,2023,2024,2025])
        Y    = np.array([8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6])
        K    = np.array([16500, 17800, 19600, 21300, 23500, 25900], dtype=float)
        L    = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
        D    = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])
        AI   = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
        H    = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])

        input_df = pd.DataFrame({
            "Năm": year, "Y_thực_tế": Y, "K": K, "L": L, "D": D, "AI": AI, "H": H
        })
        st.dataframe(input_df.set_index("Năm").style.format("{:,.2f}"), use_container_width=True)

    # ══ Tab 3: Kết quả ══════════════════════════════════════════════════════
    with tabs[2]:
        alpha = st.session_state.get("b1_alpha", 0.33)
        beta  = st.session_state.get("b1_beta",  0.42)
        gamma = st.session_state.get("b1_gamma", 0.10)
        delta = st.session_state.get("b1_delta", 0.08)
        theta = max(0.001, round(1 - alpha - beta - gamma - delta, 4))

        year = np.array([2020,2021,2022,2023,2024,2025])
        Y  = np.array([8044.4,8487.5,9513.3,10221.8,11511.9,12847.6])
        K  = np.array([16500,17800,19600,21300,23500,25900],dtype=float)
        L  = np.array([53.6,50.5,51.7,52.4,52.9,53.4])
        D  = np.array([12.0,12.7,14.3,16.5,18.3,19.5])
        AI = np.array([55.6,60.2,65.4,67.0,73.8,80.1])
        H  = np.array([24.1,26.1,26.2,27.0,28.4,29.2])

        # Tính TFP
        A      = Y / ((K**alpha)*(L**beta)*(D**gamma)*(AI**delta)*(H**theta))
        A_mean = A.mean()
        Y_hat  = A_mean * ((K**alpha)*(L**beta)*(D**gamma)*(AI**delta)*(H**theta))
        mape   = np.mean(np.abs((Y - Y_hat)/Y)) * 100

        # Growth decomposition
        dlnY  = np.diff(np.log(Y))
        dlnA  = np.diff(np.log(A))
        dlnK  = np.diff(np.log(K))
        dlnL  = np.diff(np.log(L))
        dlnD  = np.diff(np.log(D))
        dlnAI = np.diff(np.log(AI))
        dlnH  = np.diff(np.log(H))

        contrib = {
            "TFP": dlnA,
            "K":   alpha*dlnK,
            "L":   beta*dlnL,
            "D":   gamma*dlnD,
            "AI":  delta*dlnAI,
            "H":   theta*dlnH,
        }

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("TFP₂₀₂₅", f"{A[-1]:.4f}", f"+{A[-1]-A[0]:.4f} vs 2020")
        c2.metric("MAPE mô hình", f"{mape:.2f}%", "Độ chính xác dự báo")
        c3.metric("TFP trung bình", f"{A_mean:.4f}", "dùng cho dự báo")
        c4.metric("GDP 2025 thực tế", f"{Y[-1]:,.1f}", "nghìn tỷ VND")

        # Dự báo 2030
        section("Câu 1.4.1 — TFP A_t theo năm", "📉")
        fig_tfp = go.Figure()
        fig_tfp.add_trace(go.Scatter(x=year, y=A, mode="lines+markers",
            line=dict(color=COLORS[0], width=3), marker=dict(size=8),
            name="TFP A_t", fill="tozeroy",
            fillcolor="rgba(0,212,255,0.06)"))
        fig_tfp.update_layout(**PLOTLY_LAYOUT, title="Xu hướng TFP A_t, 2020–2025",
            yaxis_title="TFP A_t", xaxis_title="Năm")
        st.plotly_chart(fig_tfp, use_container_width=True)

        tfp_df = pd.DataFrame({"Năm": year, "Y_thực_tế": Y, "TFP_A_t": A, "Y_dự_báo": Y_hat,
                                "Sai_số_%": np.abs((Y-Y_hat)/Y)*100})
        st.dataframe(tfp_df.set_index("Năm").style.format("{:.4f}"), use_container_width=True)

        section("Câu 1.4.2 — GDP Thực tế vs Dự báo (MAPE)", "📊")
        col1, col2 = st.columns(2)
        with col1:
            fig_gdp = go.Figure()
            fig_gdp.add_trace(go.Scatter(x=year, y=Y, name="GDP Thực tế",
                mode="lines+markers", line=dict(color=COLORS[0], width=3), marker=dict(size=8)))
            fig_gdp.add_trace(go.Scatter(x=year, y=Y_hat, name="GDP Dự báo",
                mode="lines+markers", line=dict(color=COLORS[1], width=3, dash="dash"), marker=dict(size=8)))
            fig_gdp.update_layout(**PLOTLY_LAYOUT, title=f"GDP Thực tế vs Dự báo (MAPE={mape:.2f}%)",
                yaxis_title="GDP (nghìn tỷ VND)", legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_gdp, use_container_width=True)
        with col2:
            ape = np.abs((Y - Y_hat)/Y)*100
            fig_ape = go.Figure(go.Bar(x=year, y=ape, marker_color=COLORS[2]))
            fig_ape.update_layout(**PLOTLY_LAYOUT, title="Sai số tuyệt đối phần trăm (APE%)",
                yaxis_title="APE (%)")
            st.plotly_chart(fig_ape, use_container_width=True)

        section("Câu 1.4.3 — Phân rã tăng trưởng GDP", "🔬")
        periods = [f"{year[i]}-{year[i+1]}" for i in range(len(year)-1)]
        growth_df = pd.DataFrame({"Giai_đoạn": periods, **{k: v*100 for k,v in contrib.items()},
                                   "Tổng": sum(v for v in contrib.values())*100})
        st.dataframe(growth_df.set_index("Giai_đoạn").style.format("{:.4f}%"), use_container_width=True)

        avg = {k: v.mean() for k, v in contrib.items()}
        avg_growth = dlnY.mean()
        share = {k: v/avg_growth*100 for k,v in avg.items()}

        fig_share = go.Figure(go.Bar(
            x=list(share.keys()), y=list(share.values()),
            marker=dict(color=COLORS[:6]), text=[f"{v:.1f}%" for v in share.values()],
            textposition="outside"))
        fig_share.update_layout(**PLOTLY_LAYOUT,
            title="Tỷ trọng đóng góp bình quân 2020–2025 (%)",
            yaxis_title="Tỷ trọng (%)")
        st.plotly_chart(fig_share, use_container_width=True)

        section("Câu 1.4.4 — Dự báo GDP đến 2030", "🔮")
        col_fc1, col_fc2 = st.columns(2)
        with col_fc1:
            k_grow = st.slider("Tăng trưởng K (%/năm)", 4.0, 10.0, 6.0, 0.5, key="b1_kg") / 100
            l_grow = st.slider("Tăng trưởng L (%/năm)", 0.5,  5.0, 6.0, 0.5, key="b1_lg") / 100
        with col_fc2:
            tfp_grow = st.slider("Tăng trưởng TFP (%/năm)", 0.5, 3.0, 1.2, 0.1, key="b1_tg") / 100
            d2030    = st.slider("D đạt năm 2030 (%)", 22.0, 40.0, 30.0, 0.5, key="b1_d30")

        years_fc = np.arange(2026, 2031)
        fc_rows = []
        for t, yr in enumerate(years_fc, 1):
            Kt   = K[-1] * ((1+k_grow)**t)
            Lt   = L[-1] * ((1+l_grow)**t)
            Dt   = D[-1] + (d2030 - D[-1]) * t/5
            AIrt = AI[-1] + (100 - AI[-1]) * t/5
            Ht   = H[-1] + (35 - H[-1]) * t/5
            At   = A[-1] * ((1+tfp_grow)**t)
            Yt   = At * (Kt**alpha)*(Lt**beta)*(Dt**gamma)*(AIrt**delta)*(Ht**theta)
            fc_rows.append({"Năm": yr, "GDP_dự_báo": Yt, "TFP": At, "K": Kt})
        fc_df = pd.DataFrame(fc_rows)
        Y2030 = fc_df.iloc[-1]["GDP_dự_báo"]

        st.metric("GDP dự báo năm 2030", f"{Y2030:,.2f} nghìn tỷ VND",
                  f"+{(Y2030/Y[-1]-1)*100:.1f}% so với 2025")

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=year, y=Y, name="GDP Thực tế 2020-2025",
            mode="lines+markers", line=dict(color=COLORS[0], width=3), marker=dict(size=8)))
        fig_fc.add_trace(go.Scatter(x=years_fc, y=fc_df["GDP_dự_báo"], name="GDP Dự báo 2026-2030",
            mode="lines+markers", line=dict(color=COLORS[1], width=3, dash="dash"), marker=dict(size=8)))
        fig_fc.update_layout(**PLOTLY_LAYOUT,
            title=f"Dự báo GDP Việt Nam đến 2030",
            yaxis_title="GDP (nghìn tỷ VND)", legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_fc, use_container_width=True)

    # ══ Tab 4: Phân tích chính sách ══════════════════════════════════════════
    with tabs[3]:
        alpha = st.session_state.get("b1_alpha", 0.33)
        beta  = st.session_state.get("b1_beta",  0.42)
        gamma = st.session_state.get("b1_gamma", 0.10)
        delta = st.session_state.get("b1_delta", 0.08)
        theta = max(0.001, 1 - alpha - beta - gamma - delta)

        year = np.array([2020,2021,2022,2023,2024,2025])
        Y  = np.array([8044.4,8487.5,9513.3,10221.8,11511.9,12847.6])
        K  = np.array([16500,17800,19600,21300,23500,25900],dtype=float)
        L  = np.array([53.6,50.5,51.7,52.4,52.9,53.4])
        D  = np.array([12.0,12.7,14.3,16.5,18.3,19.5])
        AI = np.array([55.6,60.2,65.4,67.0,73.8,80.1])
        H  = np.array([24.1,26.1,26.2,27.0,28.4,29.2])
        A  = Y / ((K**alpha)*(L**beta)*(D**gamma)*(AI**delta)*(H**theta))

        contrib_vals = {
            "TFP": np.diff(np.log(A)).mean(),
            "K":   alpha*np.diff(np.log(K)).mean(),
            "L":   beta*np.diff(np.log(L)).mean(),
            "D":   gamma*np.diff(np.log(D)).mean(),
            "AI":  delta*np.diff(np.log(AI)).mean(),
            "H":   theta*np.diff(np.log(H)).mean(),
        }
        avg_g = np.diff(np.log(Y)).mean()
        share = {k: v/avg_g*100 for k, v in contrib_vals.items()}
        tfp_trend = "tăng" if A[-1] > A[0] else "giảm"
        top_new   = max(["D","AI","H"], key=lambda x: share[x])

        section("1. Xu hướng TFP", "📊")
        st.markdown(f"""
TFP của Việt Nam có xu hướng **{tfp_trend}** trong giai đoạn 2020–2025 (từ {A[0]:.4f} → {A[-1]:.4f}).
Điều này cho thấy chất lượng tăng trưởng được cải thiện — tăng trưởng không chỉ đến từ
mở rộng vốn đầu tư và lao động, mà còn từ hiệu quả sử dụng nguồn lực, công nghệ và đổi mới sáng tạo.
        """)

        section("2. Yếu tố mới đóng góp lớn nhất", "🔍")
        st.markdown(f"""
Trong nhóm yếu tố mới (D, AI, H), yếu tố đóng góp lớn nhất là **{top_new}** ({share[top_new]:.2f}%).

- **D (Số hóa)**: {share['D']:.2f}% — Kinh tế số và chuyển đổi số tạo tác động rõ nhất
- **AI (Năng lực AI)**: {share['AI']:.2f}% — Phản ánh vai trò công nghệ AI trong năng suất
- **H (Nhân lực số)**: {share['H']:.2f}% — Chất lượng lao động và đào tạo số
        """)

        section("3. Khả năng đạt mục tiêu 30% kinh tế số/GDP vào năm 2030", "🎯")
        st.markdown("""
Mục tiêu 30% kinh tế số/GDP vào 2030 **có thể khả thi**, nhưng đòi hỏi các điều kiện đi kèm:

| Điều kiện | Mô tả |
|-----------|-------|
| 🏗️ Hạ tầng số | Đầu tư mạnh vào băng thông rộng, điện toán đám mây, IoT |
| 🤖 Năng lực AI | Phát triển trung tâm dữ liệu, AI quốc gia |
| 👨‍💻 Nhân lực số | Đào tạo 50.000 kỹ sư AI/bán dẫn đến 2030 |
| 🏢 Doanh nghiệp | Hỗ trợ SME chuyển đổi số, tăng khả năng hấp thụ công nghệ |
| 🔒 An ninh | Bảo đảm an ninh dữ liệu, cybersecurity quốc gia |
        """)

        section("4. Hàm ý chính sách", "📋")
        st.info("""
Kết quả mô hình nhấn mạnh Việt Nam cần **chuyển từ tăng trưởng dựa vào vốn và lao động**
sang tăng trưởng dựa trên **năng suất, công nghệ và đổi mới sáng tạo**.

Đầu tư vào hạ tầng số, AI và nhân lực số cần được xem là nhóm **chính sách trung tâm**
trong chiến lược phát triển kinh tế giai đoạn tới, không phải là chi phụ trợ.
        """)

        section("5. Kết luận", "✅")
        st.success(f"""
Bài 1 cho thấy mô hình Cobb–Douglas mở rộng là công cụ hữu ích để phân tích tăng trưởng
kinh tế Việt Nam trong bối cảnh chuyển đổi số và AI.

Kết quả thực nghiệm:
- **TFP {tfp_trend}** trong 2020–2025 → chất lượng tăng trưởng cải thiện
- **Yếu tố nổi bật nhất trong nhóm mới: {top_new}** → số hóa tạo tác động lớn nhất
- **TFP và K** vẫn là hai động lực chính của tăng trưởng
- **Dự báo 2030**: GDP có thể tăng mạnh nếu mục tiêu 30% kinh tế số được thực hiện
        """)

    # ══ Tab 5: AI Agent ══════════════════════════════════════════════════════
    with tabs[4]:
        alpha = st.session_state.get("b1_alpha", 0.33)
        beta  = st.session_state.get("b1_beta",  0.42)
        gamma = st.session_state.get("b1_gamma", 0.10)
        delta = st.session_state.get("b1_delta", 0.08)
        theta = max(0.001, 1 - alpha - beta - gamma - delta)

        year = np.array([2020,2021,2022,2023,2024,2025])
        Y  = np.array([8044.4,8487.5,9513.3,10221.8,11511.9,12847.6])
        K  = np.array([16500,17800,19600,21300,23500,25900],dtype=float)
        L  = np.array([53.6,50.5,51.7,52.4,52.9,53.4])
        D  = np.array([12.0,12.7,14.3,16.5,18.3,19.5])
        AI = np.array([55.6,60.2,65.4,67.0,73.8,80.1])
        H  = np.array([24.1,26.1,26.2,27.0,28.4,29.2])
        A  = Y / ((K**alpha)*(L**beta)*(D**gamma)*(AI**delta)*(H**theta))
        A_mean = A.mean()
        Y_hat  = A_mean * ((K**alpha)*(L**beta)*(D**gamma)*(AI**delta)*(H**theta))
        mape   = np.mean(np.abs((Y - Y_hat)/Y)) * 100

        contrib_vals = {
            "TFP": np.diff(np.log(A)).mean(),
            "K":   alpha*np.diff(np.log(K)).mean(),
            "L":   beta*np.diff(np.log(L)).mean(),
            "D":   gamma*np.diff(np.log(D)).mean(),
            "AI":  delta*np.diff(np.log(AI)).mean(),
            "H":   theta*np.diff(np.log(H)).mean(),
        }
        avg_g = np.diff(np.log(Y)).mean()
        share = {k: v/avg_g*100 for k, v in contrib_vals.items()}
        top_f = max(share, key=share.get)
        top_new = max(["D","AI","H"], key=lambda x: share[x])
        mape_eval = "rất tốt (< 3%)" if mape < 3 else "chấp nhận được" if mape < 8 else "cần cải thiện"
        st.caption("Bấm nút AI Agent bên dưới để hiển thị phân tích chính sách mô phỏng.")
        ai_agent(f"""
<b>1. Đánh giá chất lượng mô hình:</b><br>
MAPE = <b>{mape:.2f}%</b> → Mức độ chính xác dự báo: <b>{mape_eval}</b>.<br>
Với bộ tham số hiện tại (α={alpha}, β={beta}, γ={gamma}, δ={delta}, θ={theta:.4f}),
mô hình mô phỏng xu hướng GDP thực tế Việt Nam với sai số tương đối ở mức
{'có thể chấp nhận trong phân tích kinh tế vĩ mô giản lược' if mape < 10 else 'khá cao — nên xem xét điều chỉnh tham số'}.<br><br>

<b>2. Nhận định về đóng góp tăng trưởng:</b><br>
Yếu tố đóng góp <b>lớn nhất</b> vào tăng trưởng GDP là <b>{top_f}</b> ({share[top_f]:.2f}%).<br>
Trong nhóm yếu tố số hóa/công nghệ, <b>{top_new}</b> nổi bật nhất với {share[top_new]:.2f}% tổng tăng trưởng.<br><br>

<b>3. Khuyến nghị điều chỉnh chính sách:</b><br>
{'✅ Tăng γ (trọng số số hóa D) nếu ưu tiên kinh tế số' if gamma < 0.12 else '⚠️ γ đã cao — cần đi kèm đầu tư hạ tầng số thực chất'}<br>
{'✅ Tăng δ (trọng số AI) phù hợp với chiến lược AI quốc gia' if delta < 0.10 else '⚠️ δ cao — cần nhân lực AI đủ năng lực hấp thụ'}<br>
{'⚠️ θ thấp — cần đầu tư mạnh hơn vào nhân lực số!' if theta < 0.06 else '✅ θ hợp lý — duy trì chương trình đào tạo 50.000 kỹ sư AI'}<br><br>

<b>4. Dự báo chiến lược đến 2030:</b><br>
Để đạt mục tiêu kinh tế số 30% GDP, Việt Nam cần:<br>
• Duy trì tăng trưởng TFP ≥ 1,5%/năm<br>
• Tăng AI capacity từ 80,1 lên ~100+ nghìn DN số<br>
• Nâng tỷ lệ lao động qua đào tạo lên 35%+<br>
• Đảm bảo tăng trưởng K ổn định ≥ 6%/năm
        """)

        # Radar chart comparison
        section("So sánh đóng góp theo bộ tham số hiện tại", "📡")
        categories = list(share.keys())
        vals       = [max(0, share[k]) for k in categories]
        categories_c = categories + [categories[0]]
        vals_c       = vals + [vals[0]]

        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=vals_c, theta=categories_c, fill="toself",
            name="Đóng góp %", line=dict(color=COLORS[0], width=2),
            fillcolor="rgba(0,212,255,0.12)"))
        fig_r.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            polar=dict(bgcolor="rgba(13,17,32,0.6)",
                       radialaxis=dict(gridcolor="rgba(255,255,255,.1)", color="#64748B"),
                       angularaxis=dict(gridcolor="rgba(255,255,255,.1)", color="#94A3B8")),
            font=dict(color="#94A3B8", family="Inter"),
            title=dict(text="Tỷ trọng đóng góp tăng trưởng (%)", font=dict(color="#00D4FF")),
            showlegend=False, margin=dict(t=60,b=30,l=30,r=30), height=380
        )
        st.plotly_chart(fig_r, use_container_width=True)
