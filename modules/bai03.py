"""
Bài 3 — Chỉ số ưu tiên chuyển đổi số ngành (TOPSIS đơn giản)
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
df_raw = pd.DataFrame({
    "sector_name_vi": [
        "Nông-Lâm-Thủy sản",
        "CN chế biến chế tạo",
        "Xây dựng",
        "Khai khoáng",
        "Bán buôn-bán lẻ",
        "Tài chính-Ngân hàng",
        "Logistics-Vận tải",
        "CNTT-Truyền thông",
        "Giáo dục-Đào tạo",
        "Y tế"
    ],
    "growth_rate_2024_pct": [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
    "productivity_million_VND_per_worker": [103.4, 241.2, 168.8, 1290.5, 145.3, 1072.4, 321.4, 713.8, 205.7, 437.1],
    "spillover_coef_0_1": [0.35, 0.78, 0.42, 0.30, 0.55, 0.85, 0.72, 0.92, 0.65, 0.60],
    "export_billion_USD": [40.5, 290.9, 2.5, 8.2, 5.5, 1.2, 3.1, 178.0, 0.0, 0.0],
    "labor_million": [13.20, 11.50, 4.80, 0.30, 7.80, 0.55, 1.95, 0.62, 2.15, 0.75],
    "ai_readiness_0_100": [15, 55, 20, 30, 48, 72, 42, 88, 38, 45],
    "automation_risk_pct": [18, 42, 25, 55, 38, 52, 35, 28, 22, 18]
})

def norm_good(x):
    if x.max() == x.min():
        return x * 0
    return (x - x.min()) / (x.max() - x.min())

def show():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
      <div style="display:flex;align-items:center;gap:1rem;">
        <span style="font-size:2rem;">🏭</span>
        <div>
          <div style="color:#00D4FF;font-size:.75rem;font-weight:600;letter-spacing:1px;">BÀI 3</div>
          <div style="color:#E2E8F0;font-size:1.5rem;font-weight:800;">Chỉ số ưu tiên 10 ngành phát triển kinh tế số (TOPSIS)</div>
          <div style="margin-top:.3rem;">
            <span class="badge b-blue">Multi-Criteria Decision Making (MCDM)</span>
            <span class="badge b-purple">Min-Max Normalization</span>
            <span class="badge b-green">Sensitivity Analysis</span>
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
        Việt Nam đang đẩy mạnh chuyển đổi số và ứng dụng AI trong nền kinh tế. Tuy nhiên, nguồn lực chính sách và ngân sách
        là có hạn, vì vậy cần xác định **ngành nào nên được ưu tiên đầu tư trước** để tạo hiệu ứng lan tỏa lớn nhất.

        Nền kinh tế Việt Nam gồm các khu vực lớn như: nông-lâm-thủy sản, công nghiệp - xây dựng và dịch vụ. Mỗi ngành có đặc điểm
        khác nhau về tăng trưởng, năng suất, xuất khẩu, việc làm, mức độ sẵn sàng AI và rủi ro tự động hóa.
        
        Bài toán đặt ra là xây dựng một chỉ số định lượng **Priority_i** xếp hạng 10 ngành kinh tế Việt Nam.
        """)

        section("Mô hình toán học", "📐")
        math_box("""
Priority<sub>i</sub> = a<sub>1</sub>·Growth<sub>i</sub> + a<sub>2</sub>·Productivity<sub>i</sub> + a<sub>3</sub>·Spillover<sub>i</sub> + a<sub>4</sub>·Export<sub>i</sub> + a<sub>5</sub>·Employment<sub>i</sub> + a<sub>6</sub>·AIReadiness<sub>i</sub> - a<sub>7</sub>·Risk<sub>i</sub>
&nbsp;
Chuẩn hóa Min-Max về đoạn [0, 1]:
  • Tiêu chí tốt (lợi ích):  x_norm = (x - min(x)) / (max(x) - min(x))
  • Tiêu chí xấu (chi phí - Risk): risk_norm = (x - min(x)) / (max(x) - min(x))
&nbsp;
Risk được trừ đi trong công thức tính Priority để thể hiện tác hại bất lợi từ rủi ro tự động hóa việc làm.
        """)

    # ══ Tab 2: Dữ liệu & Trọng số ══════════════════════════════════════════
    with tabs[1]:
        section("Điều chỉnh trọng số tiêu chí chính sách", "🎛️")
        col1, col2 = st.columns(2)
        with col1:
            w_growth = st.slider("a1 — Tăng trưởng", 0.0, 0.4, 0.15, 0.01, key="b3_w1")
            w_prod = st.slider("a2 — Năng suất lao động", 0.0, 0.4, 0.15, 0.01, key="b3_w2")
            w_spill = st.slider("a3 — Hệ số lan tỏa", 0.0, 0.4, 0.20, 0.01, key="b3_w3")
            w_exp = st.slider("a4 — Giá trị xuất khẩu", 0.0, 0.4, 0.15, 0.01, key="b3_w4")
        with col2:
            w_emp = st.slider("a5 — Quy mô việc làm", 0.0, 0.4, 0.10, 0.01, key="b3_w5")
            w_ai = st.slider("a6 — Sẵn sàng AI", 0.0, 0.4, 0.20, 0.01, key="b3_w6")
            w_risk = st.slider("a7 — Rủi ro tự động hóa", 0.0, 0.4, 0.15, 0.01, key="b3_w7")
            
            w_sum = w_growth + w_prod + w_spill + w_exp + w_emp + w_ai + w_risk
            st.markdown(f"**Tổng trọng số hiện tại:** ` {w_sum:.2f} ` (tự động quy đổi tỷ lệ để tổng tuyệt đối bằng 1)")

        # Normalization of weights
        sum_w = max(0.0001, w_sum)
        w = {
            "Growth": w_growth / sum_w,
            "Productivity": w_prod / sum_w,
            "Spillover": w_spill / sum_w,
            "Export": w_exp / sum_w,
            "Employment": w_emp / sum_w,
            "AIReadiness": w_ai / sum_w,
            "Risk": w_risk / sum_w
        }

        section("Bảng dữ liệu 10 ngành kinh tế gốc (2024)", "📋")
        st.dataframe(df_raw.style.format({
            "growth_rate_2024_pct": "{:.2f}%",
            "productivity_million_VND_per_worker": "{:,.1f}",
            "spillover_coef_0_1": "{:.2f}",
            "export_billion_USD": "{:,.1f}",
            "labor_million": "{:.2f}",
            "ai_readiness_0_100": "{:.0f}",
            "automation_risk_pct": "{:.0f}%"
        }), use_container_width=True)

    # ══ Tab 3: Kết quả xếp hạng ════════════════════════════════════════════
    with tabs[2]:
        # Compute normalized data
        cols_good = [
            "growth_rate_2024_pct", "productivity_million_VND_per_worker",
            "spillover_coef_0_1", "export_billion_USD",
            "labor_million", "ai_readiness_0_100"
        ]
        col_bad = "automation_risk_pct"
        
        Xg = df_raw[cols_good].apply(norm_good)
        Xrisk = norm_good(df_raw[col_bad])
        
        df_norm = pd.DataFrame({
            "Ngành": df_raw["sector_name_vi"],
            "Growth": Xg["growth_rate_2024_pct"],
            "Productivity": Xg["productivity_million_VND_per_worker"],
            "Spillover": Xg["spillover_coef_0_1"],
            "Export": Xg["export_billion_USD"],
            "Employment": Xg["labor_million"],
            "AIReadiness": Xg["ai_readiness_0_100"],
            "Risk_penalty": Xrisk
        })

        # Calculate Priority
        priority_score = (
            w["Growth"] * df_norm["Growth"]
            + w["Productivity"] * df_norm["Productivity"]
            + w["Spillover"] * df_norm["Spillover"]
            + w["Export"] * df_norm["Export"]
            + w["Employment"] * df_norm["Employment"]
            + w["AIReadiness"] * df_norm["AIReadiness"]
            - w["Risk"] * df_norm["Risk_penalty"]
        )

        df_res = df_raw.copy()
        df_res["Priority"] = priority_score
        df_res = df_res.sort_values("Priority", ascending=False).reset_index(drop=True)
        df_res["Xếp hạng"] = df_res.index + 1

        c1, c2 = st.columns(2)
        with c1:
            section("Câu 3.4.1 — Chuẩn hóa Min-Max các chỉ tiêu", "📐")
            st.dataframe(
                df_norm.style.format({c: "{:.4f}" for c in df_norm.columns if c != "Ngành"}),
                use_container_width=True,
            )
            section("Câu 3.4.2 — Tính Priority_i với trọng số hiện tại", "🏆")
            st.dataframe(df_res[["Xếp hạng", "sector_name_vi", "Priority"]].style.format({"Priority": "{:.4f}"}), use_container_width=True)
        with c2:
            section("Biểu đồ xếp hạng ưu tiên", "📊")
            fig_bar = go.Figure(go.Bar(
                x=df_res["Priority"][::-1],
                y=df_res["sector_name_vi"][::-1],
                orientation='h',
                marker_color=COLORS[0],
                text=df_res["Priority"][::-1].round(3).astype(str),
                textposition='outside'
            ))
            fig_bar.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        section("Biểu đồ mở rộng — Phân rã cấu phần điểm Priority", "🧬")
        df_comp = pd.DataFrame({
            "Ngành": df_raw["sector_name_vi"],
            "Tăng trưởng": w["Growth"] * df_norm["Growth"],
            "Năng suất": w["Productivity"] * df_norm["Productivity"],
            "Lan tỏa": w["Spillover"] * df_norm["Spillover"],
            "Xuất khẩu": w["Export"] * df_norm["Export"],
            "Việc làm": w["Employment"] * df_norm["Employment"],
            "Sẵn sàng AI": w["AIReadiness"] * df_norm["AIReadiness"],
            "Rủi ro tự động hóa (-v)": -w["Risk"] * df_norm["Risk_penalty"]
        }).set_index("Ngành").loc[df_res["sector_name_vi"]]

        fig_stack = go.Figure()
        for c_idx, col in enumerate(df_comp.columns):
            fig_stack.add_trace(go.Bar(
                x=df_comp.index,
                y=df_comp[col],
                name=col,
                marker_color=COLORS[c_idx % len(COLORS)]
            ))
        fig_stack.update_layout(**PLOTLY_LAYOUT, barmode='relative', title="Đóng góp cấu phần vào điểm ưu tiên", height=450)
        st.plotly_chart(fig_stack, use_container_width=True)

    # ══ Kết quả mở rộng: độ nhạy và so sánh chính sách ════════════════════
    with tabs[2]:
        section("Câu 3.4.3 — Phân tích độ nhạy theo trọng số AI Readiness", "📡")
        st.markdown("Chúng ta chạy phân tích khi thay đổi trọng số AI Readiness từ **0.05 đến 0.40** để kiểm tra tính ổn định của quyết định chính sách.")
        
        ai_weights = np.arange(0.05, 0.41, 0.05)
        sens_rows = []
        base_arr = np.array([w_growth, w_prod, w_spill, w_exp, w_emp, w_risk])
        base_arr_sum = base_arr.sum() if base_arr.sum() > 0 else 1.0

        for ai_w in ai_weights:
            scale_factor = (1.0 - ai_w) / base_arr_sum
            w_temp = base_arr * scale_factor
            
            p_score = (
                w_temp[0] * df_norm["Growth"]
                + w_temp[1] * df_norm["Productivity"]
                + w_temp[2] * df_norm["Spillover"]
                + w_temp[3] * df_norm["Export"]
                + w_temp[4] * df_norm["Employment"]
                + ai_w * df_norm["AIReadiness"]
                - w_temp[5] * df_norm["Risk_penalty"]
            )
            
            for idx, r_row in df_raw.iterrows():
                sens_rows.append({
                    "Ngành": r_row["sector_name_vi"],
                    "Trọng số AI": round(ai_w, 2),
                    "Điểm Priority": p_score[idx]
                })

        df_sens = pd.DataFrame(sens_rows)
        fig_sens = px.line(df_sens, x="Trọng số AI", y="Điểm Priority", color="Ngành", markers=True, color_discrete_sequence=COLORS)
        fig_sens.update_layout(**PLOTLY_LAYOUT, title="Sự thay đổi điểm Priority khi điều chỉnh trọng số AI Readiness")
        st.plotly_chart(fig_sens, use_container_width=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            section("Câu 3.4.4 — So sánh hai bộ trọng số chính sách", "⚖️")
            st.markdown("""
            - **Định hướng tăng trưởng**: Tập trung năng suất, xuất khẩu, AI readiness và tăng trưởng kinh tế.
            - **Định hướng bao trùm**: Tập trung việc làm rộng rãi, tính lan tỏa liên ngành và hạn chế rủi ro thất nghiệp.
            """)
            
            w_gr = {"Growth": 0.25, "Productivity": 0.20, "Spillover": 0.15, "Export": 0.20, "Employment": 0.05, "AIReadiness": 0.10, "Risk": 0.05}
            w_in = {"Growth": 0.10, "Productivity": 0.10, "Spillover": 0.25, "Export": 0.05, "Employment": 0.25, "AIReadiness": 0.10, "Risk": 0.15}
            
            p_gr = w_gr["Growth"]*df_norm["Growth"] + w_gr["Productivity"]*df_norm["Productivity"] + w_gr["Spillover"]*df_norm["Spillover"] + w_gr["Export"]*df_norm["Export"] + w_gr["Employment"]*df_norm["Employment"] + w_gr["AIReadiness"]*df_norm["AIReadiness"] - w_gr["Risk"]*df_norm["Risk_penalty"]
            p_in = w_in["Growth"]*df_norm["Growth"] + w_in["Productivity"]*df_norm["Productivity"] + w_in["Spillover"]*df_norm["Spillover"] + w_in["Export"]*df_norm["Export"] + w_in["Employment"]*df_norm["Employment"] + w_in["AIReadiness"]*df_norm["AIReadiness"] - w_in["Risk"]*df_norm["Risk_penalty"]
            
            df_compare = pd.DataFrame({
                "Ngành": df_raw["sector_name_vi"],
                "Priority Tăng trưởng": p_gr,
                "Priority Bao trùm": p_in
            }).sort_values("Priority Tăng trưởng", ascending=False)
            
            st.dataframe(df_compare.style.format({
                "Priority Tăng trưởng": "{:.4f}",
                "Priority Bao trùm": "{:.4f}"
            }), use_container_width=True)

        with col_p2:
            section("Biểu đồ mở rộng — Bản đồ đánh đổi AI Readiness vs. Rủi ro tự động hóa", "🗺️")
            fig_scatter = px.scatter(df_raw, x="ai_readiness_0_100", y="automation_risk_pct", size="labor_million",
                                     hover_name="sector_name_vi", text="sector_name_vi", color_discrete_sequence=[COLORS[0]])
            fig_scatter.update_traces(textposition='top center')
            fig_scatter.update_layout(**PLOTLY_LAYOUT, title="Phân nhóm các ngành theo AI Readiness & Risk")
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ══ Tab 4: Phân tích chính sách ═══════════════════════════════════════
    with tabs[3]:
        top1 = df_res.iloc[0]["sector_name_vi"]
        top3 = ", ".join(df_res.head(3)["sector_name_vi"].tolist())
        low_risk_high_ai = df_raw.sort_values(["ai_readiness_0_100", "automation_risk_pct"], ascending=[False, True]).iloc[0]["sector_name_vi"]
        section("1. Vì sao không chỉ chọn ngành có tăng trưởng cao nhất?", "📌")
        st.markdown(f"""
        Ngành đứng đầu theo Priority hiện tại là **{top1}**, không nhất thiết là ngành có một chỉ tiêu đơn lẻ cao nhất.
        Mô hình tổng hợp đồng thời tăng trưởng, năng suất, lan tỏa, xuất khẩu, việc làm, AI readiness và rủi ro tự động hóa.
        Vì vậy, chính sách nên ưu tiên ngành có **điểm cân bằng đa tiêu chí**, thay vì chỉ nhìn tốc độ tăng trưởng hoặc năng suất.
        """)
        section("2. Chính sách tăng trưởng và chính sách bao trùm khác nhau thế nào?", "⚖️")
        st.markdown(f"""
        Nhóm ưu tiên theo bộ trọng số hiện tại gồm **{top3}**.
        Khi tăng trọng số việc làm và giảm rủi ro, thứ hạng có thể dịch chuyển sang các ngành sử dụng nhiều lao động hoặc có tác động xã hội rộng.
        Điều này cho thấy Chính phủ cần công bố rõ mục tiêu: tăng trưởng nhanh, lan tỏa công nghệ, hay bao trùm lao động.
        """)
        section("3. AI Readiness cao có luôn nên đầu tư mạnh không?", "🤖")
        st.markdown(f"""
        Ngành có AI readiness nổi bật là **{low_risk_high_ai}**, nhưng chính sách vẫn phải kiểm tra automation risk.
        Nếu AI readiness cao đi cùng rủi ro tự động hóa lớn, đầu tư AI nên kèm ngân sách đào tạo lại và bảo vệ việc làm.
        Vì vậy, Priority Index nên được dùng như công cụ sàng lọc ban đầu, sau đó kết hợp đánh giá lao động và an sinh xã hội.
        """)
        top_row = df_res.iloc[0]
        low_row = df_res.iloc[-1]
        risk_row = df_raw.sort_values("automation_risk_pct", ascending=False).iloc[0]
        section("4. Hàm ý chính sách nâng cấp", "📋")
        st.markdown(f"""
**Kết quả nổi bật.** Ngành đứng đầu là **{top_row['sector_name_vi']}** với Priority = **{top_row['Priority']:.4f}**; nhóm Top 3 gồm **{top3}**; ngành cuối bảng là **{low_row['sector_name_vi']}**; ngành rủi ro tự động hóa cao nhất là **{risk_row['sector_name_vi']}** với **{risk_row['automation_risk_pct']:.1f}%**.

**Liên hệ chính sách Việt Nam.** Kết quả phù hợp với **Nghị quyết 57-NQ/TW** khi ưu tiên ngành dựa trên đổi mới sáng tạo, AI readiness và lan tỏa; đồng thời hỗ trợ **Quyết định 411/QĐ-TTg** trong lựa chọn ngành kinh tế số trọng điểm.

**Đánh đổi cần lưu ý:** AI và đào tạo lại lao động; ngành có AI readiness cao cần ngân sách kỹ năng số nếu automation risk lớn.

**Khuyến nghị hành động.** Ưu tiên thử nghiệm AI trong Top 3 ngành; tăng đào tạo lại cho ngành có automation risk cao nhất; theo dõi Priority khi thay đổi trọng số việc làm/rủi ro; không dùng tăng trưởng đơn lẻ để quyết định phân bổ ngân sách.
        """)

    # ══ Tab 5: AI Agent ═══════════════════════════════════════════════════
    with tabs[4]:
        st.caption("Bấm nút AI Agent bên dưới để hiển thị phân tích chính sách mô phỏng.")
        top1 = df_res.iloc[0]["sector_name_vi"]
        top2 = df_res.iloc[1]["sector_name_vi"]
        top3 = df_res.iloc[2]["sector_name_vi"]
        lowest = df_res.iloc[-1]["sector_name_vi"]
        
        ai_agent(f"""
        <b>1. Nhận định về kết quả xếp hạng hiện tại:</b><br>
        Với bộ trọng số bạn vừa cấu hình, Top 3 ngành được ưu tiên chuyển đổi số là 
        <b>1. {top1}</b>, <b>2. {top2}</b> và <b>3. {top3}</b>. Ngành có mức ưu tiên thấp nhất là <b>{lowest}</b>.<br><br>
        
        <b>2. Giải mã hiện tượng Khai khoáng:</b><br>
        Mặc dù ngành <i>Khai khoáng</i> có năng suất lao động vượt trội ({df_raw.iloc[3]['productivity_million_VND_per_worker']:.1f} triệu VND/LĐ), 
        ngành này bị tụt hạng sâu trong danh mục ưu tiên. Nguyên nhân cốt lõi là hệ số lan tỏa thấp, quy mô việc làm nhỏ, tăng trưởng âm, 
        và rủi ro tự động hóa đặc biệt cao. Điều này chứng minh rằng một ngành có năng suất cao đơn lẻ sẽ không phải là trung tâm chính sách 
        nếu thiếu khả năng lan tỏa và tạo giá trị gia tăng liên ngành rộng lớn.<br><br>
        
        <b>3. Đề xuất chính sách:</b><br>
        • <b>Nhóm Tiên phong (CNTT, Tài chính, Chế biến chế tạo)</b>: Tập trung phát triển các mô hình AI chuyên dụng quy mô lớn, tạo đòn bẩy dẫn dắt toàn bộ chuỗi giá trị.<br>
        • <b>Nhóm Xã hội (Nông nghiệp, Bán buôn bán lẻ)</b>: Do có lượng lao động khổng lồ, chuyển đổi số tại đây cần hướng đến sự bao trùm xã hội, nâng cao kỹ năng số cơ bản để phòng ngừa rủi ro tự động hóa thay thế việc làm đột ngột.
        """)
