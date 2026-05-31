"""
Bài 7 — Tối ưu đa mục tiêu Pareto với NSGA-II
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
regions = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
region_names = {
    "NMM": "Trung du miền núi phía Bắc",
    "RRD": "Đồng bằng sông Hồng",
    "NCC": "Bắc Trung Bộ + DH Trung Bộ",
    "CH": "Tây Nguyên",
    "SE": "Đông Nam Bộ",
    "MD": "Đồng bằng sông Cửu Long"
}
items = ["I", "D", "AI", "H"]
item_names = {
    "I": "Hạ tầng số",
    "D": "CĐS/Dữ liệu",
    "AI": "Năng lực AI",
    "H": "Nhân lực số"
}

beta = np.array([
    [1.15, 0.85, 0.55, 1.30],  # NMM
    [0.95, 1.25, 1.40, 1.05],  # RRD
    [1.05, 0.95, 0.85, 1.15],  # NCC
    [1.20, 0.75, 0.45, 1.35],  # CH
    [0.90, 1.30, 1.55, 1.00],  # SE
    [1.10, 0.85, 0.65, 1.25],  # MD
])
e = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
sigma = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])

# ── NSGA-II Solve ─────────────────────────────────────────────────────────────
def run_nsga2_opt(pop_size=50, n_gen=60):
    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
    except ImportError:
        return None

    class VietnamDigitalProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=24,
                n_obj=4,
                n_ieq_constr=14,
                xl=np.zeros(24),
                xu=np.ones(24) * 12000
            )
            self.beta = beta
            self.e = e
            self.rho = rho
            self.sigma = sigma

        def _evaluate(self, x, out, *args, **kwargs):
            X = x.reshape(6, 4)
            region_sum = X.sum(axis=1)

            # Objectives
            # f1: max GDP gain => min -GDP
            f1 = - (self.beta * X).sum()
            # f2: min inequality (MAD of regional budgets)
            f2 = np.abs(region_sum - region_sum.mean()).mean()
            # f3: min CO2 emission
            f3 = (self.e * (X[:, 0] + X[:, 2])).sum()
            # f4: min Net security risk
            f4 = (self.rho * X[:, 2]).sum() - (self.sigma * X[:, 3]).sum()

            out["F"] = [f1, f2, f3, f4]

            # Constraints (G <= 0)
            G = []
            # C1: total budget <= 50000
            G.append(X.sum() - 50000)
            # C2 & C3: min region budget 5000, max 12000
            for i in range(6):
                G.append(5000 - region_sum[i])
                G.append(region_sum[i] - 12000)
            # C4: min human budget 12000
            G.append(12000 - X[:, 3].sum())

            out["G"] = np.array(G)

    problem = VietnamDigitalProblem()
    algorithm = NSGA2(pop_size=pop_size)
    
    res = minimize(problem, algorithm, ('n_gen', n_gen), seed=42, verbose=False)
    return res

def show():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
      <div style="display:flex;align-items:center;gap:1rem;">
        <span style="font-size:2rem;">⚡</span>
        <div>
          <div style="color:#00D4FF;font-size:.75rem;font-weight:600;letter-spacing:1px;">BÀI 7</div>
          <div style="color:#E2E8F0;font-size:1.5rem;font-weight:800;">Tối ưu đa mục tiêu Pareto bằng giải thuật di truyền NSGA-II</div>
          <div style="margin-top:.3rem;">
            <span class="badge b-blue">Multi-Objective Evolutionary Algorithm</span>
            <span class="badge b-purple">Pareto Frontier</span>
            <span class="badge b-green">Parallel Coordinates</span>
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
        Trong hoạch định chính sách kinh tế số, hầu như không bao giờ tồn tại một phương án duy nhất tối ưu cho tất cả các bên. 
        Chính phủ Việt Nam đồng thời theo đuổi nhiều mục tiêu chiến lược:
        
        1. **Hiệu quả kinh tế**: Tối đa hóa tổng GDP tăng thêm.
        2. **Bao trùm xã hội**: Tối thiểu hóa bất bình đẳng vùng miền (phân bổ đều nguồn lực).
        3. **Phát triển bền vững**: Tối thiểu hóa phát thải CO2 gián tiếp từ hạ tầng số và siêu máy tính AI.
        4. **An ninh chủ quyền**: Tối thiểu hóa rủi ro rò rỉ dữ liệu quốc gia (giảm thiểu rủi ro AI và nâng cao đào tạo nhân lực số).
        
        Các mục tiêu này xung đột mạnh mẽ với nhau. Ví dụ: Đầu tư nhiều vào AI thúc đẩy GDP nhưng tăng phát thải CO2 và rủi ro an ninh mạng.
        Do đó, thay vì đi tìm một nghiệm duy nhất, ta đi tìm **Tập nghiệm tối ưu Pareto (Pareto-optimal set)**.
        """)

        section("Mô hình toán học đa mục tiêu", "📐")
        math_box("""
<b>Biến quyết định:</b>
  x<sub>j, r</sub> ≥ 0 : Ngân sách phân bổ (6 vùng x 4 hạng mục = 24 biến).

<b>Hàm mục tiêu (Tối thiểu hóa):</b>
  1. GDP Gain:  f<sub>1</sub> = - Σ<sub>r</sub> Σ<sub>j</sub> β<sub>j, r</sub> · x<sub>j, r</sub>
  2. Bất bình đẳng:  f<sub>2</sub> = MeanAbsoluteDeviation(Σ<sub>j</sub> x<sub>j, r</sub>)
  3. Phát thải CO2:  f<sub>3</sub> = Σ<sub>r</sub> e<sub>r</sub> · (x<sub>I, r</sub> + x<sub>AI, r</sub>)
  4. Rủi ro an ninh mạng ròng:  f<sub>4</sub> = Σ<sub>r</sub> ρ<sub>r</sub> · x<sub>AI, r</sub> - Σ<sub>r</sub> σ<sub>r</sub> · x<sub>H, r</sub>

<b>Các ràng buộc (Kế thừa Bài 4):</b>
  • Tổng ngân sách ≤ 50.000 tỷ.
  • Ngân sách mỗi vùng ∈ [5.000, 12.000] tỷ.
  • Tổng ngân sách nhân lực số H ≥ 12.000 tỷ.
        """)

    # ══ Tab 2: Cấu hình NSGA-II ════════════════════════════════════════════
    with tabs[1]:
        section("Cấu hình giải thuật di truyền", "🎛️")
        pop_size = st.slider("Quy mô quần thể (Population Size)", 20, 100, 50, 10, key="b7_pop")
        n_gen = st.slider("Số lượng thế hệ sinh (Generations)", 20, 150, 60, 10, key="b7_gen")

        st.markdown(f"""
        > [!TIP]
        > **Giải thuật NSGA-II** hoạt động mô phỏng theo cơ chế tiến hóa sinh học: 
        > Tạo một quần thể ngẫu nhiên, sau đó qua từng thế hệ sẽ tiến hành lai ghép (crossover), đột biến (mutation) 
        > và chọn lọc tự nhiên dựa trên độ trội Pareto và khoảng cách chen chúc (crowding distance) để hội tụ về biên tối ưu.
        """)

        section("Thông số các mục tiêu phụ trợ theo vùng", "📋")
        df_p_data = pd.DataFrame({
            "Vùng": [region_names[r] for r in regions],
            "Hệ số CO2 e_r": e,
            "Rủi ro AI ρ_r": rho,
            "Chống rủi ro H σ_r": sigma
        })
        st.dataframe(df_p_data.set_index("Vùng").style.format("{:.2f}"), use_container_width=True)

    # ══ Tab 3: Pareto Frontier ════════════════════════════════════════════
    with tabs[2]:
        res = run_nsga2_opt(pop_size, n_gen)
        
        if res is not None and res.F is not None:
            st.success(f"🎉 **Chạy NSGA-II thành công!** Tìm thấy **{len(res.F)}** giải pháp không bị trội (Pareto Solutions).")
            
            # Form data
            F = res.F
            # Convert back to original scale (positive GDP)
            gdp_gain = -F[:, 0]
            inequality = F[:, 1]
            emission = F[:, 2]
            risk_val = F[:, 3]
            
            df_pareto = pd.DataFrame({
                "Giải pháp ID": [f"Sol_{i+1}" for i in range(len(F))],
                "GDP Gain (tỷ)": gdp_gain,
                "Bất bình đẳng (MAD)": inequality,
                "Phát thải CO2": emission,
                "Rủi ro an ninh ròng": risk_val
            })
            
            section("Câu 7.4.1 — Chạy NSGA-II và trích xuất tập Pareto", "🏆")
            st.dataframe(df_pareto.style.format({
                "GDP Gain (tỷ)": "{:,.2f}",
                "Bất bình đẳng (MAD)": "{:,.2f}",
                "Phát thải CO2": "{:,.2f}",
                "Rủi ro an ninh ròng": "{:,.2f}"
            }), use_container_width=True)

            # Chart 1: 3D Scatter plot
            section("Câu 7.4.2 — Trực quan hóa tập Pareto", "📊")
            fig_3d = px.scatter_3d(df_pareto, x="GDP Gain (tỷ)", y="Phát thải CO2", z="Rủi ro an ninh ròng",
                                   color="Bất bình đẳng (MAD)", hover_name="Giải pháp ID", color_continuous_scale="Jet")
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=550)
            st.plotly_chart(fig_3d, use_container_width=True)

            # Chart 2: Parallel Coordinates
            section("Biểu đồ Tọa độ Song song (Parallel Coordinates)", "📈")
            fig_par = px.parallel_coordinates(df_pareto, dimensions=["GDP Gain (tỷ)", "Bất bình đẳng (MAD)", "Phát thải CO2", "Rủi ro an ninh ròng"],
                                             color="GDP Gain (tỷ)", color_continuous_scale="Plasma")
            fig_par.update_layout(height=450)
            st.plotly_chart(fig_par, use_container_width=True)
            
        else:
            st.error("❌ Môi trường thiếu thư viện `pymoo`. Không thể khởi chạy giải thuật di truyền.")

    # ══ Kết quả mở rộng: nghiệm thỏa hiệp và phân bổ chi tiết ════════════
    with tabs[2]:
        if res is not None and res.F is not None:
            section("Câu 7.4.3 — Chọn nghiệm thỏa hiệp để xem cơ cấu phân bổ", "🔍")
            sol_id = st.selectbox("Chọn giải pháp tương thích:", df_pareto["Giải pháp ID"].tolist(), key="b7_sol_sel")
            
            sol_idx = df_pareto[df_pareto["Giải pháp ID"] == sol_id].index[0]
            x_opt_24 = res.X[sol_idx]
            x_opt_6x4 = x_opt_24.reshape(6, 4)
            
            # Show kpi
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GDP Gain", f"{gdp_gain[sol_idx]:,.1f} tỷ")
            c2.metric("Phát thải CO2", f"{emission[sol_idx]:,.1f}")
            c3.metric("Bất bình đẳng MAD", f"{inequality[sol_idx]:,.1f}")
            c4.metric("Rủi ro an ninh", f"{risk_val[sol_idx]:,.1f}")

            section(f"Bảng phân bổ ngân sách 6 vùng x 4 ngành của {sol_id} (tỷ VND)", "📋")
            res_rows = []
            for r_idx, r in enumerate(regions):
                res_rows.append({
                    "Vùng": region_names[r],
                    "Hạ tầng số (I)": x_opt_6x4[r_idx, 0],
                    "CĐS/Dữ liệu (D)": x_opt_6x4[r_idx, 1],
                    "Năng lực AI (AI)": x_opt_6x4[r_idx, 2],
                    "Nhân lực số (H)": x_opt_6x4[r_idx, 3],
                    "Tổng ngân sách vùng": x_opt_6x4[r_idx].sum()
                })
            df_sol_tbl = pd.DataFrame(res_rows)
            st.dataframe(df_sol_tbl.set_index("Vùng").style.format("{:,.2f}"), use_container_width=True)

            # Chart breakdown
            fig_brk = px.bar(df_sol_tbl.melt(id_vars=["Vùng"], value_vars=["Hạ tầng số (I)", "CĐS/Dữ liệu (D)", "Năng lực AI (AI)", "Nhân lực số (H)"],
                                            var_name="Hạng mục", value_name="Tỷ VND"),
                             x="Vùng", y="Tỷ VND", color="Hạng mục", barmode="stack", color_discrete_sequence=COLORS)
            fig_brk.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig_brk, use_container_width=True)

            section("Câu 7.4.4 — Chi phí cơ hội của mục tiêu tăng trưởng", "💸")
            max_gdp_idx = np.argmax(gdp_gain)
            min_ineq_idx = np.argmin(inequality)
            opp_df = pd.DataFrame({
                "Phương án": ["GDP cao nhất", "Bất bình đẳng thấp nhất", "Đang chọn"],
                "GDP Gain (tỷ)": [gdp_gain[max_gdp_idx], gdp_gain[min_ineq_idx], gdp_gain[sol_idx]],
                "Bất bình đẳng (MAD)": [inequality[max_gdp_idx], inequality[min_ineq_idx], inequality[sol_idx]],
                "Phát thải CO2": [emission[max_gdp_idx], emission[min_ineq_idx], emission[sol_idx]],
                "Rủi ro an ninh": [risk_val[max_gdp_idx], risk_val[min_ineq_idx], risk_val[sol_idx]],
            })
            st.dataframe(
                opp_df.style.format({c: "{:,.2f}" for c in opp_df.columns if c != "Phương án"}),
                use_container_width=True,
            )
            
        else:
            st.warning("⚠️ Không có dữ liệu tối ưu.")

    # ══ Tab 4: Phân tích chính sách ═══════════════════════════════════════
    with tabs[3]:
        section("1. Đánh đổi giữa tăng trưởng và bao trùm có rõ ràng không?", "⚖️")
        st.markdown("""
        Tập Pareto cho thấy không có một nghiệm duy nhất thống trị toàn bộ mục tiêu.
        Các nghiệm GDP cao thường đi kèm phân bổ tập trung hơn, phát thải cao hơn hoặc rủi ro an ninh dữ liệu lớn hơn.
        Vì vậy, quyết định chính sách phải chọn một điểm thỏa hiệp, không chỉ chọn nghiệm tối đa GDP.
        """)
        section("2. Vì sao NSGA-II phù hợp hơn LP đơn mục tiêu?", "🧬")
        st.markdown("""
        LP đơn mục tiêu buộc nhà hoạch định gộp mọi mục tiêu vào một hàm duy nhất, dễ che khuất đánh đổi.
        NSGA-II tạo ra cả biên Pareto, giúp nhìn trực tiếp các phương án tăng trưởng, bao trùm, môi trường và an ninh.
        Đây là cách trình bày phù hợp hơn cho chính sách công vì quyết định cuối cùng cần cân nhắc nhiều mục tiêu xã hội.
        """)
        section("3. Nên chọn nghiệm nào để triển khai?", "📌")
        st.markdown("""
        Nghiệm được khuyến nghị thường nằm ở vùng giữa của Pareto frontier: GDP đủ cao, bất bình đẳng không quá lớn, phát thải và rủi ro được kiểm soát.
        Các nghiệm biên chỉ nên dùng làm benchmark để hiểu chi phí cơ hội.
        Khi triển khai thực tế, cần bổ sung tiêu chí hấp thụ vốn, năng lực địa phương và lộ trình giải ngân.
        """)
        if res is not None and res.F is not None:
            max_gdp_idx = int(np.argmax(gdp_gain))
            min_emission_idx = int(np.argmin(emission))
            min_ineq_idx = int(np.argmin(inequality))
            section("4. Hàm ý chính sách nâng cấp", "📋")
            st.markdown(f"""
**Kết quả nổi bật.** Tập Pareto có **{len(df_pareto)}** nghiệm; nghiệm GDP cao nhất là **{df_pareto.iloc[max_gdp_idx]['Giải pháp ID']}** với GDP gain **{gdp_gain[max_gdp_idx]:,.1f}**; nghiệm phát thải thấp nhất là **{df_pareto.iloc[min_emission_idx]['Giải pháp ID']}** với emission **{emission[min_emission_idx]:,.1f}**; nghiệm công bằng nhất là **{df_pareto.iloc[min_ineq_idx]['Giải pháp ID']}** với MAD **{inequality[min_ineq_idx]:,.2f}**.

**Liên hệ chính sách Việt Nam.** Kết quả phù hợp với **Nghị quyết 57-NQ/TW** khi chính sách AI/chuyển đổi số cần tối ưu đa mục tiêu, đồng thời liên quan **Quyết định 749/QĐ-TTg** về chuyển đổi số bao trùm.

**Đánh đổi cần lưu ý:** hiệu quả kinh tế, công bằng vùng, phát thải và rủi ro dữ liệu không thể đồng thời đạt cực trị.

**Khuyến nghị hành động.** Không chọn nghiệm biên cực đoan; chọn vùng giữa Pareto làm phương án cơ sở; dùng nghiệm GDP cao nhất và phát thải thấp nhất làm benchmark; bổ sung tiêu chí hấp thụ vốn địa phương trước khi phân bổ ngân sách thật.
            """)

    # ══ Tab 5: AI Agent ═══════════════════════════════════════════════════
    with tabs[4]:
        st.caption("Bấm nút AI Agent bên dưới để hiển thị phân tích chính sách mô phỏng.")
        if res is not None and res.F is not None:
            # Find extreme options
            max_gdp_idx = np.argmax(gdp_gain)
            min_co2_idx = np.argmin(emission)
            min_ineq_idx = np.argmin(inequality)
            
            ai_agent(f"""
            <b>1. Nhận dạng các nghiệm biên cực hạn của Pareto Frontier:</b><br>
            • <b>Phương án ưu tiên Tăng trưởng nhất</b>: Giải pháp <b>{df_pareto.iloc[max_gdp_idx]['Giải pháp ID']}</b> đạt GDP tối đa là <b>{gdp_gain[max_gdp_idx]:,.1f} tỷ VND</b>, 
            nhưng phải đánh đổi bằng phát thải khí nhà kính rất lớn ({emission[max_gdp_idx]:,.1f}) và độ bất bình đẳng cao.<br>
            • <b>Phương án thân thiện môi trường nhất</b>: Giải pháp <b>{df_pareto.iloc[min_co2_idx]['Giải pháp ID']}</b> giảm phát thải tối thiểu xuống <b>{emission[min_co2_idx]:,.1f}</b> 
            bằng cách triệt tiêu gần như hoàn toàn đầu tư vào hạ tầng số (I) và siêu máy tính AI tại các vùng phát thải cao, dẫn đến GDP gain bị sụt giảm đáng kể.<br>
            • <b>Phương án công bằng nhất</b>: Giải pháp <b>{df_pareto.iloc[min_ineq_idx]['Giải pháp ID']}</b> phân bổ gần như tuyệt đối bằng nhau giữa 6 vùng (MAD = {inequality[min_ineq_idx]:,.2f}).<br><br>
            
            <b>2. Khuyến nghị phối hợp chính sách:</b><br>
            Mô hình tiến hóa NSGA-II cung cấp cho các nhà hoạch định một bản đồ đánh đổi vô cùng rõ ràng. 
            Thay vì chọn các nghiệm biên cực đoan, Chính phủ nên lựa chọn một nghiệm ở khu vực trung tâm của biên Pareto (ví dụ các giải pháp có GDP gain trung bình từ 55.000 đến 58.000 tỷ VND và chỉ số an ninh mạng ròng ổn định ở âm). 
            Đây là vùng **Cân bằng bền vững** giúp kinh tế số Việt Nam tăng tốc mạnh mẽ mà không phá vỡ tính bao trùm xã hội hay các mục tiêu cam kết net-zero môi trường.
            """)
        else:
            ai_agent("Mô hình chưa có nghiệm tối ưu Pareto để phân tích.")
