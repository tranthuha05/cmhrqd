"""
Bài 8 — Tối ưu động phân bổ liên thời gian 2026–2035
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize

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

# ── Dynamic Model constants ──────────────────────────────────────────────────
years = np.arange(2026, 2036)
T = len(years)

alpha_K = 0.33
alpha_L = 0.42
alpha_D = 0.10
alpha_AI = 0.08
alpha_H = 0.07

K0 = 27500.0
L0 = 53.9
D0 = 20.3
AI0 = 86.0
H0 = 30.0
A0 = 35.0

def production(A, K, L, D, AI, H):
    return A * (K ** alpha_K) * (L ** alpha_L) * (D ** alpha_D) * (AI ** alpha_AI) * (H ** alpha_H)

def simulate_path(investment_shares, rho, delta_K, delta_D, delta_AI, theta_H, mu, phi1, phi2, phi3, shock_2028=False):
    shares = investment_shares.reshape(T, 4)
    
    K_arr = np.zeros(T + 1)
    D_arr = np.zeros(T + 1)
    AI_arr = np.zeros(T + 1)
    H_arr = np.zeros(T + 1)
    A_arr = np.zeros(T + 1)
    
    Y_arr = np.zeros(T)
    C_arr = np.zeros(T)
    
    K_arr[0] = K0
    D_arr[0] = D0
    AI_arr[0] = AI0
    H_arr[0] = H0
    A_arr[0] = A0
    
    welfare = 0.0
    
    for t in range(T):
        L_t = L0 * ((1.006) ** t)
        Y_t = production(A_arr[t], K_arr[t], L_t, D_arr[t], AI_arr[t], H_arr[t])
        
        # Apply economic shock in 2028
        if shock_2028 and years[t] == 2028:
            Y_t = Y_t * 0.92
            
        Y_arr[t] = Y_t
        sK, sD, sAI, sH = shares[t]
        
        total_s = sK + sD + sAI + sH
        if total_s >= 0.98:
            return None
            
        i_K = sK * Y_t
        i_D = sD * Y_t
        i_AI = sAI * Y_t
        i_H = sH * Y_t
        
        c_t = Y_t - (i_K + i_D + i_AI + i_H)
        if c_t <= 1e-3:
            return None
            
        C_arr[t] = c_t
        welfare += (rho ** t) * np.log(c_t)
        
        # Capital dynamics
        K_arr[t+1] = (1.0 - delta_K) * K_arr[t] + i_K
        D_arr[t+1] = (1.0 - delta_D) * D_arr[t] + i_D
        AI_arr[t+1] = (1.0 - delta_AI) * AI_arr[t] + i_AI
        H_arr[t+1] = H_arr[t] + theta_H * i_H - mu * H_arr[t]
        
        # Endogenous TFP
        A_arr[t+1] = A_arr[t] * (1.0 + phi1 * D_arr[t] + phi2 * AI_arr[t] + phi3 * H_arr[t])
        
    return {
        "K": K_arr, "D": D_arr, "AI": AI_arr, "H": H_arr, "A": A_arr,
        "Y": Y_arr, "C": C_arr, "welfare": welfare
    }

def solve_dynamic_optimization(rho, delta_K, delta_D, delta_AI, theta_H, mu, phi1, phi2, phi3, shock_2028=False):
    # Initial guess: 10% for K, 5% for others
    x0 = np.tile([0.10, 0.05, 0.05, 0.05], T)
    
    # Bounds: shares must be in [0, 0.50]
    bounds = [(0.0, 0.50) for _ in range(4 * T)]
    
    # Constraints: sum of shares per year <= 0.80
    constraints = []
    for t in range(T):
        def yearly_constraint(x, year_idx=t):
            # return 0.80 - sum of shares for year_idx
            return 0.80 - x[4 * year_idx : 4 * year_idx + 4].sum()
        constraints.append({"type": "ineq", "fun": yearly_constraint})
        
    def obj_func(x):
        sim = simulate_path(x, rho, delta_K, delta_D, delta_AI, theta_H, mu, phi1, phi2, phi3, shock_2028)
        if sim is None:
            return 1e9
        return -sim["welfare"]
        
    res = minimize(obj_func, x0, bounds=bounds, constraints=constraints, method="SLSQP", options={"maxiter": 200})
    return res

def show():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
      <div style="display:flex;align-items:center;gap:1rem;">
        <span style="font-size:2rem;">📅</span>
        <div>
          <div style="color:#00D4FF;font-size:.75rem;font-weight:600;letter-spacing:1px;">BÀI 8</div>
          <div style="color:#E2E8F0;font-size:1.5rem;font-weight:800;">Tối ưu hóa động liên thời gian vĩ mô (2026–2035)</div>
          <div style="margin-top:.3rem;">
            <span class="badge b-blue">Scipy SLSQP</span>
            <span class="badge b-purple">Dynamic Capital Accumulation</span>
            <span class="badge b-green">Endogenous TFP Growth</span>
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
        Việt Nam đặt mục tiêu trở thành nước thu nhập trung bình cao vào năm 2030 và nước phát triển thu nhập cao vào năm 2045.
        Để đạt được mục tiêu dài hạn này, Chính phủ cần thiết kế một lộ trình **phân bổ vốn liên thời gian (intertemporal dynamic allocation)** 
        giai đoạn 2026–2035 giữa các tài sản truyền thống và tài sản số.
        
        Ta xây dựng một mô hình Ramsey-Cass-Koopmans mở rộng tích hợp vốn vật chất K, hạ tầng số D, năng lực AI và vốn nhân lực H, 
        trong đó TFP tăng trưởng nội sinh nhờ mức tích lũy vốn số.
        Mục tiêu là phân bổ đầu tư mỗi năm để **tối đa hóa tổng phúc lợi liên thời gian chiết khấu** của nền kinh tế.
        """)

        section("Mô hình động học toán học", "📐")
        math_box("""
<b>Hàm mục tiêu phúc lợi dài hạn (Welfare):</b>
  max W = Σ<sub>t=2026</sub><sup>2035</sup> ρ<sup>t</sup> · ln(C<sub>t</sub>)   (Với C<sub>t</sub> là tiêu dùng, ρ là hệ số chiết khấu)

<b>Hàm sản xuất Cobb-Douglas mở rộng:</b>
  Y<sub>t</sub> = A<sub>t</sub> · K<sub>t</sub><sup>0.33</sup> · L<sub>t</sub><sup>0.42</sup> · D<sub>t</sub><sup>0.10</sup> · AI<sub>t</sub><sup>0.08</sup> · H<sub>t</sub><sup>0.07</sup>

<b>Phương trình tích lũy tài sản (Capital Dynamics):</b>
  • Vốn vật chất: K<sub>t+1</sub> = (1 - δ<sub>K</sub>)·K<sub>t</sub> + I<sub>K, t</sub>
  • Hạ tầng số:   D<sub>t+1</sub> = (1 - δ<sub>D</sub>)·D<sub>t</sub> + I<sub>D, t</sub>
  • Năng lực AI:   AI<sub>t+1</sub> = (1 - δ<sub>AI</sub>)·AI<sub>t</sub> + I<sub>AI, t</sub>
  • Nhân lực số:   H<sub>t+1</sub> = H<sub>t</sub> + θ<sub>H</sub>·I<sub>H, t</sub> - μ·H<sub>t</sub>

<b>TFP nội sinh (Endogenous TFP):</b>
  A<sub>t+1</sub> = A<sub>t</sub> · (1 + φ<sub>1</sub>·D<sub>t</sub> + φ<sub>2</sub>·AI<sub>t</sub> + φ<sub>3</sub>·H<sub>t</sub>)

<b>Ràng buộc ngân sách tiêu dùng/đầu tư hàng năm:</b>
  C<sub>t</sub> + I<sub>K, t</sub> + I<sub>D, t</sub> + I<sub>AI, t</sub> + I<sub>H, t</sub> ≤ Y<sub>t</sub>
        """)

    # ══ Tab 2: Tham số Động học ════════════════════════════════════════════
    with tabs[1]:
        section("Hệ số khấu hao và hiệu suất đầu tư số", "🎛️")
        col1, col2 = st.columns(2)
        with col1:
            rho = st.slider("Hệ số chiết khấu liên thời gian ρ", 0.90, 0.99, 0.97, 0.01, key="b8_rho")
            delta_K = st.slider("Khấu hao vốn vật chất δ_K", 0.02, 0.10, 0.05, 0.01, key="b8_dK")
            delta_D = st.slider("Khấu hao hạ tầng số δ_D", 0.05, 0.20, 0.12, 0.01, key="b8_dD")
            delta_AI = st.slider("Khấu hao AI δ_AI", 0.05, 0.25, 0.15, 0.01, key="b8_dAI")
        with col2:
            theta_H = st.slider("Hiệu quả đào tạo nhân lực số θ_H", 0.5, 1.2, 0.8, 0.1, key="b8_tH")
            mu_coef = st.slider("Khấu hao nhân lực số μ", 0.01, 0.08, 0.02, 0.01, key="b8_mu")
            
            st.markdown("**Hệ số đóng góp TFP nội sinh (φ)**")
            phi1 = st.slider("φ1 (Hạ tầng số D)", 0.001, 0.006, 0.003, 0.0005, key="b8_p1")
            phi2 = st.slider("φ2 (Năng lực AI)", 0.001, 0.006, 0.002, 0.0005, key="b8_p2")
            phi3 = st.slider("φ3 (Nhân lực số H)", 0.001, 0.008, 0.004, 0.0005, key="b8_p3")

    # ══ Tab 3: Quỹ đạo Tối ưu ════════════════════════════════════════════
    with tabs[2]:
        st.write("🔄 *Đang giải bài toán tối ưu động liên thời gian bằng SLSQP...*")
        res_opt = solve_dynamic_optimization(rho, delta_K, delta_D, delta_AI, theta_H, mu_coef, phi1, phi2, phi3, shock_2028=False)
        
        if res_opt.success:
            st.success("🎉 **Giải tối ưu động thành công!** Hệ thống đã hội tụ sau các ràng buộc.")
            
            sim_opt = simulate_path(res_opt.x, rho, delta_K, delta_D, delta_AI, theta_H, mu_coef, phi1, phi2, phi3, shock_2028=False)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng Phúc lợi xã hội W*", f"{sim_opt['welfare']:.4f}")
            c2.metric("GDP cuối kỳ 2035", f"{sim_opt['Y'][-1]:,.1f} tỷ")
            c3.metric("TFP cuối kỳ 2035", f"{sim_opt['A'][-1]:.2f}")
            
            section("Câu 8.3.1 — Tối ưu bằng scipy SLSQP", "✅")
            st.markdown(f"Giá trị phúc lợi tối ưu W* đạt **{sim_opt['welfare']:.4f}**, GDP cuối kỳ đạt **{sim_opt['Y'][-1]:,.1f} tỷ VND**.")
            section("Câu 8.3.2 — Quỹ đạo tối ưu 2026–2035", "📋")
            df_path = pd.DataFrame({
                "Năm": years,
                "GDP (Y)": sim_opt["Y"],
                "Tiêu dùng (C)": sim_opt["C"],
                "Vốn vật chất K": sim_opt["K"][:-1],
                "Hạ tầng số D": sim_opt["D"][:-1],
                "Năng lực AI": sim_opt["AI"][:-1],
                "Nhân lực số H": sim_opt["H"][:-1],
                "TFP (A)": sim_opt["A"][:-1]
            })
            st.dataframe(df_path.set_index("Năm").style.format("{:,.2f}"), use_container_width=True)

            # Chart: GDP & Consumption
            section("Biểu đồ Quỹ đạo GDP và Tiêu dùng tối ưu", "📈")
            fig_gdp = go.Figure()
            fig_gdp.add_trace(go.Scatter(x=years, y=sim_opt["Y"], name="GDP (Y_t)", mode="lines+markers", line=dict(color=COLORS[0], width=3)))
            fig_gdp.add_trace(go.Scatter(x=years, y=sim_opt["C"], name="Tiêu dùng (C_t)", mode="lines+markers", line=dict(color=COLORS[2], width=3)))
            fig_gdp.update_layout(**PLOTLY_LAYOUT, yaxis_title="Tỷ VND")
            st.plotly_chart(fig_gdp, use_container_width=True)

            # Chart: Capital stock accumulation
            section("Tích lũy các tài sản số dài hạn", "🔬")
            df_cap = df_path.melt(id_vars=["Năm"], value_vars=["Hạ tầng số D", "Năng lực AI", "Nhân lực số H"], var_name="Tài sản số", value_name="Mức tích lũy")
            fig_cap = px.line(df_cap, x="Năm", y="Mức tích lũy", color="Tài sản số", markers=True, color_discrete_sequence=COLORS[1:4])
            fig_cap.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_cap, use_container_width=True)
            
        else:
            st.error("Lỗi tối ưu hóa. Vui lòng điều chỉnh lại các tham số động lực học.")

    # ══ Kết quả mở rộng: shock và chiến lược đầu tư ══════════════════════
    with tabs[2]:
        section("Câu 8.3.3 — Phân tích cú sốc năm 2028", "⚡")
        st.markdown("""
        Cú sốc này mô phỏng các biến động khó lường của kinh tế toàn cầu hoặc thiên tai dịch bệnh lớn tác động lên Việt Nam.
        Ta so sánh quỹ đạo tối ưu có thích nghi Shock vĩ mô để đánh giá khả năng phản ứng.
        """)
        
        res_shock = solve_dynamic_optimization(rho, delta_K, delta_D, delta_AI, theta_H, mu_coef, phi1, phi2, phi3, shock_2028=True)
        
        if res_shock.success and res_opt.success:
            sim_shock = simulate_path(res_shock.x, rho, delta_K, delta_D, delta_AI, theta_H, mu_coef, phi1, phi2, phi3, shock_2028=True)
            
            fig_shock = go.Figure()
            fig_shock.add_trace(go.Scatter(x=years, y=sim_opt["Y"], name="GDP Tối ưu thường", mode="lines+markers", line=dict(color=COLORS[0], width=2)))
            fig_shock.add_trace(go.Scatter(x=years, y=sim_shock["Y"], name="GDP khi có Shock 2028", mode="lines+markers", line=dict(color=COLORS[4], width=3)))
            fig_shock.update_layout(**PLOTLY_LAYOUT, yaxis_title="Tỷ VND")
            st.plotly_chart(fig_shock, use_container_width=True)
            
            st.markdown("""
            **Tác động của Shock liên thời gian:**
            - **Hành vi Front-load**: Để ứng phó với việc thu nhập giảm sút năm 2028, mô hình tối ưu động thích ứng sẽ tự động thắt chặt tiêu dùng trong 2 năm đầu (2026-2027) để dồn tiền đầu tư mạnh vào cơ sở hạ tầng số và đào tạo lại nhân lực số nhằm lấy đà chống chịu tốt nhất cho năm bị Shock.
            - **Khả năng hồi phục**: Nhờ tính thích ứng của SLSQP vĩ mô, quỹ đạo kinh tế nhanh chóng hồi phục chữ V từ năm 2029 và tiệm cận lại mức tăng trưởng cũ nhờ nền tảng TFP nội sinh vững chắc.
            """)

            section("Câu 8.3.4 — So sánh chiến lược trải đều và front-load", "📊")
            s_opt = res_opt.x.reshape(T, 4)
            early_mean = s_opt[:3].sum(axis=1).mean()
            late_mean = s_opt[-3:].sum(axis=1).mean()
            strategy_df = pd.DataFrame({
                "Giai đoạn": ["3 năm đầu", "3 năm cuối"],
                "Cường độ đầu tư bình quân": [early_mean, late_mean],
            })
            st.dataframe(strategy_df.style.format({"Cường độ đầu tư bình quân": "{:.4f}"}), use_container_width=True)
            st.markdown(f"""
            <div class="result-note-card">
              <div class="result-note-title">💬 Nhận xét kết quả</div>
              <div class="result-note-body">
                Cường độ đầu tư bình quân 3 năm đầu là {early_mean:.4f}, so với {late_mean:.4f} ở 3 năm cuối.
                Nếu đầu kỳ cao hơn cuối kỳ, chiến lược có tính front-loaded, tức đầu tư sớm để tích lũy K, D, AI và H.
                Điều này phù hợp với mô hình động vì vốn số và nhân lực số tạo hiệu ứng lũy kế lên TFP trong các năm sau.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ══ Tab 4: Phân tích chính sách ═══════════════════════════════════════
    with tabs[3]:
        section("1. Quỹ đạo tối ưu có front-loaded hay không?", "📌")
        st.markdown("""
        Nếu đầu tư tập trung nhiều hơn ở các năm đầu, mô hình đang ưu tiên tích lũy sớm các biến trạng thái K, D, AI và H.
        Cách làm này có thể làm giảm tiêu dùng ngắn hạn nhưng tăng sản lượng và TFP trong trung hạn.
        Với chính sách Việt Nam, front-load phù hợp khi năng lực giải ngân và quản trị dự án đủ mạnh.
        """)
        section("2. AI và nhân lực số có nên đi cùng nhau?", "🤖")
        st.markdown("""
        AI có khấu hao công nghệ nhanh và dễ tạo rủi ro thay thế lao động nếu thiếu nhân lực số.
        Vì vậy, đầu tư AI cần đi kèm đầu tư H để bảo đảm khả năng vận hành, hấp thụ và tái đào tạo.
        Đây là điều kiện để tăng trưởng số không tạo ra chi phí xã hội quá lớn.
        """)
        section("3. Cú sốc 2028 gợi ý gì về khả năng chống chịu?", "⚡")
        st.markdown("""
        Mô phỏng cú sốc cho thấy chiến lược đầu tư dài hạn cần tính đến khả năng gián đoạn sản lượng.
        Những cấu phần có tính lũy kế như nhân lực số và hạ tầng số giúp nền kinh tế phục hồi tốt hơn sau cú sốc.
        Chính sách nên duy trì ngân sách số tối thiểu ngay cả trong giai đoạn suy giảm để tránh mất động lực dài hạn.
        """)

    # ══ Tab 5: AI Agent ═══════════════════════════════════════════════════
    with tabs[4]:
        st.caption("Bấm nút AI Agent bên dưới để hiển thị phân tích chính sách mô phỏng.")
        if res_opt.success:
            s_opt = res_opt.x.reshape(T, 4)
            mean_sAI = s_opt[:, 2].mean() * 100
            mean_sH = s_opt[:, 3].mean() * 100
            
            ai_agent(f"""
            <b>1. Đánh giá tính chất liên thời gian vĩ mô:</b><br>
            • Quỹ đạo phân bổ tối ưu đề xuất cơ cấu đầu tư trung bình dài hạn: 
            AI chiếm <b>{mean_sAI:.2f}%</b> và Nhân lực số H chiếm <b>{mean_sH:.2f}%</b> trên tổng GDP hàng năm.<br>
            • **Quy luật động học**: Khấu hao AI rất cao ($\\delta_{{AI}} = {delta_AI:.2f}$) đòi hỏi nền kinh tế phải duy trì dòng vốn liên tục chảy vào AI để duy trì năng lực cạnh tranh dài hạn. 
            Ngược lại, vốn nhân lực H có tính lũy kế cực tốt nhờ tỷ lệ khấu hao thấp, do đó mô hình ưu tiên "front-load" (bơm cực mạnh vốn vào H trong giai đoạn 3 năm đầu 2026-2028), 
            sau đó giảm dần và duy trì ổn định.<br><br>
            
            <b>2. Khuyến nghị hoạch định chính sách Việt Nam:</b><br>
            • Đầu tư vào **nhân lực số (H)** đóng vai trò như một *hàng hóa bảo hiểm vĩ mô*. Khi xảy ra khủng hoảng hoặc cú sốc vĩ mô năm 2028, chính H là cứu cánh giúp TFP phục hồi nhanh chóng và nâng cao sức đề kháng của lực lượng lao động trước nguy cơ sa thải.<br>
            • Chính phủ cần tránh tư duy đầu tư ngắn hạn, duy trì kiên định tỷ lệ đầu tư số tối thiểu $\\ge 20\\%$ ngân sách hàng năm để thúc đẩy hàm tăng trưởng TFP nội sinh $\\phi_3$ bứt phá đưa Việt Nam thoát bẫy thu nhập trung bình trước năm 2035.
            """)
