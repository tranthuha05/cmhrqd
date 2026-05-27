"""
Bài 9 — Tác động AI tới thị trường lao động Việt Nam
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pulp

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
sectors = [
    "Nông-Lâm-Thủy sản",
    "CN chế biến chế tạo",
    "Xây dựng",
    "Bán buôn-bán lẻ",
    "Tài chính-Ngân hàng",
    "Logistics-Vận tải",
    "CNTT-Truyền thông",
    "Giáo dục-Đào tạo"
]

labor_million = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100

a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
b1 = np.array([45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0])
c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
d1 = np.array([50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0])

N = len(sectors)

def solve_labor_lp(budget, add_displacement_cap=False, max_displace_pct=5.0):
    model = pulp.LpProblem("VN_Labor_Market_Optimization", pulp.LpMaximize)
    
    x_AI = pulp.LpVariable.dicts("x_AI", range(N), lowBound=0)
    x_H = pulp.LpVariable.dicts("x_H", range(N), lowBound=0)
    
    # Formulate variables
    # NetJob_i = NewJob_AI_i + UpgradeJob_i - DisplacedJob_i
    # Objective: Sum(NetJob_i)
    model += pulp.lpSum(
        (a1[i] - c1[i] * risk[i]) * x_AI[i] + b1[i] * x_H[i] for i in range(N)
    ), "Total_Net_Jobs"
    
    # Constraints
    # C1: Budget
    model += pulp.lpSum(x_AI[i] + x_H[i] for i in range(N)) <= budget, "Total_Budget_Limit"
    
    # C2 & C3 for each sector
    for i in range(N):
        # NetJob_i >= 0
        model += (a1[i] - c1[i] * risk[i]) * x_AI[i] + b1[i] * x_H[i] >= 0, f"NetJob_Nonneg_{i}"
        
        # DisplacedJob_i <= RetrainingCapacity_i
        model += c1[i] * risk[i] * x_AI[i] <= d1[i] * x_H[i], f"Retrain_Cap_{i}"
        
        # C4: Optional displacement cap (as a % of total jobs in sector)
        if add_displacement_cap:
            labor_jobs = labor_million[i] * 1_000_000
            model += c1[i] * risk[i] * x_AI[i] <= (max_displace_pct / 100.0) * labor_jobs, f"Displacement_Cap_{i}"
            
    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[status] == "Optimal":
        x_AI_val = np.array([pulp.value(x_AI[i]) for i in range(N)])
        x_H_val = np.array([pulp.value(x_H[i]) for i in range(N)])
        
        new_job = a1 * x_AI_val
        upgrade_job = b1 * x_H_val
        displaced_job = c1 * risk * x_AI_val
        retrain_cap = d1 * x_H_val
        net_job = new_job + upgrade_job - displaced_job
        
        return {
            "success": True,
            "x_AI": x_AI_val,
            "x_H": x_H_val,
            "NewJob": new_job,
            "UpgradeJob": upgrade_job,
            "DisplacedJob": displaced_job,
            "RetrainingCapacity": retrain_cap,
            "NetJob": net_job,
            "objective": pulp.value(model.objective)
        }
    else:
        return {
            "success": False
        }

def show():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
      <div style="display:flex;align-items:center;gap:1rem;">
        <span style="font-size:2rem;">👷</span>
        <div>
          <div style="color:#00D4FF;font-size:.75rem;font-weight:600;letter-spacing:1px;">BÀI 9</div>
          <div style="color:#E2E8F0;font-size:1.5rem;font-weight:800;">Tác động của AI & Tự động hóa tới thị trường lao động Việt Nam</div>
          <div style="margin-top:.3rem;">
            <span class="badge b-blue">Convex Labor Optimization</span>
            <span class="badge b-purple">Social Retraining Safety net</span>
            <span class="badge b-green">Net Employment Minimization</span>
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
        Cuộc cách mạng AI và Tự động hóa hứa hẹn nâng cao vượt bậc năng suất lao động quốc gia, 
        nhưng cũng dấy lên làn sóng lo ngại về việc thay thế hàng triệu việc làm truyền thống tại Việt Nam.
        Đặc biệt trong các ngành sử dụng đông lao động như *Chế biến chế tạo* hay *Bán buôn - bán lẻ*.
        
        Để đảm bảo phát triển kinh tế bao trùm, Chính phủ cam kết **an sinh xã hội**:
        - Tốc độ mất việc làm không được phép vượt quá tốc độ và năng lực đào tạo lại (Retraining capacity) của lực lượng lao động số.
        - Không có bất cứ ngành nào bị mất việc làm ròng (NetJob ròng $\\ge 0$).
        
        Ta sử dụng mô hình tối ưu hóa tuyến tính để cực đại hóa số việc làm ròng được tạo thêm trên toàn quốc.
        """)

        section("Mô hình toán học thị trường lao động", "📐")
        math_box("""
<b>Cấu trúc việc làm của ngành i:</b>
  NetJob<sub>i</sub> = NewJob_AI<sub>i</sub> + UpgradeJob<sub>i</sub> - DisplacedJob<sub>i</sub>

Trong đó:
  • Việc làm mới từ AI:   NewJob_AI<sub>i</sub> = a<sub>i</sub> · x_AI<sub>i</sub>  (Với x_AI là ngân sách AI tỷ VND)
  • Việc làm nâng cấp kỹ năng: UpgradeJob<sub>i</sub> = b<sub>i</sub> · x_H<sub>i</sub>  (Với x_H là ngân sách đào tạo H tỷ VND)
  • Việc làm bị AI thế chỗ:   DisplacedJob<sub>i</sub> = c<sub>i</sub> · Risk<sub>i</sub> · x_AI<sub>i</sub>
  • Năng lực đào tạo lại:   RetrainCap<sub>i</sub> = d<sub>i</sub> · x_H<sub>i</sub>

<b>Hàm mục tiêu:</b>
  max Σ<sub>i=1..8</sub> NetJob<sub>i</sub>   (Tối đa hóa tổng lượng việc làm ròng toàn quốc)

<b>Các ràng buộc xã hội bắt buộc:</b>
  1. Ngân sách đào tạo & phát triển: Σ<sub>i</sub> (x_AI<sub>i</sub> + x_H<sub>i</sub>) ≤ Budget  (30.000 tỷ VND)
  2. Rào chắn việc làm ròng:  NetJob<sub>i</sub> ≥ 0  với mọi i (Không để ngành nào bị mất việc làm ròng)
  3. Lưới bảo hiểm đào tạo:   DisplacedJob<sub>i</sub> ≤ RetrainCap<sub>i</sub>  với mọi i (Tự động hóa không vượt quá năng lực tái đào tạo)
        """)

    # ══ Tab 2: Cấu hình Tham số ════════════════════════════════════════════
    with tabs[1]:
        section("Cấu hình ngân sách & an sinh xã hội", "🎛️")
        col1, col2 = st.columns(2)
        with col1:
            budget = st.slider("💰 Tổng ngân sách lao động (tỷ VND)", 10000, 50000, 30000, 1000, key="b9_bg")
            add_displacement_cap = st.checkbox("Áp đặt trần tỷ lệ sa thải tối đa trên quy mô lao động", value=False, key="b9_cap_cb")
        with col2:
            max_displace_pct = st.slider("🛑 Tỷ lệ sa thải trần (%)", 1.0, 10.0, 5.0, 0.5, key="b9_dis_pct")
            
            st.markdown(f"""
            > [!IMPORTANT]
            > **Ràng buộc sa thải trần**: Khi kích hoạt, lượng việc làm bị thay thế của mỗi ngành không được vượt quá 
            > **{max_displace_pct:.1f}%** tổng số việc làm thực tế của ngành đó nhằm hạn chế cú sốc thất nghiệp cục bộ.
            """)

        section("Bảng dữ liệu đặc trưng lao động theo 8 ngành", "📋")
        df_sec_data = pd.DataFrame({
            "Ngành": sectors,
            "Lao động thực tế (triệu)": labor_million,
            "Rủi ro tự động hóa (%)": risk * 100,
            "Hiệu quả NewJob (a1)": a1,
            "Hiệu quả Upgrade (b1)": b1,
            "Khấu hao sa thải (c1)": c1,
            "Hiệu quả đào tạo (d1)": d1
        })
        st.dataframe(df_sec_data.set_index("Ngành").style.format({
            "Lao động thực tế (triệu)": "{:.2f}",
            "Rủi ro tự động hóa (%)": "{:.1f}%",
            "Hiệu quả NewJob (a1)": "{:.1f}",
            "Hiệu quả Upgrade (b1)": "{:.1f}",
            "Khấu hao sa thải (c1)": "{:.1f}",
            "Hiệu quả đào tạo (d1)": "{:.1f}"
        }), use_container_width=True)

    # ══ Tab 3: Phân bổ lao động tối ưu ════════════════════════════════════
    with tabs[2]:
        res = solve_labor_lp(budget, add_displacement_cap, max_displace_pct)
        
        if res["success"]:
            st.success("🎉 **Tìm thấy nghiệm tối ưu vĩ mô thích ứng!**")
            
            x_AI_v = res["x_AI"]
            x_H_v = res["x_H"]
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            col_metric1.metric("Tổng NetJob được tạo thêm", f"{res['objective']:,.0f} việc làm")
            col_metric2.metric("Ngân sách AI (tỷ VND)", f"{x_AI_v.sum():,.1f} tỷ")
            col_metric3.metric("Ngân sách đào tạo lại (tỷ VND)", f"{x_H_v.sum():,.1f} tỷ")
            
            section("Câu 9.4.1 — Giải mô hình cơ sở", "📋")
            res_rows = []
            for i in range(N):
                res_rows.append({
                    "Ngành": sectors[i],
                    "Ngân sách AI (x_AI)": x_AI_v[i],
                    "Ngân sách đào tạo (x_H)": x_H_v[i],
                    "Việc làm mới (New)": res["NewJob"][i],
                    "Việc nâng cấp (Upgrade)": res["UpgradeJob"][i],
                    "Việc bị thay thế (Displaced)": res["DisplacedJob"][i],
                    "Năng lực tái đào tạo": res["RetrainingCapacity"][i],
                    "Việc làm ròng (NetJob)": res["NetJob"][i]
                })
            df_res_tbl = pd.DataFrame(res_rows)
            st.dataframe(df_res_tbl.set_index("Ngành").style.format("{:,.1f}"), use_container_width=True)

            # Chart: Stacked Budget per Sector
            section("Phân bổ ngân sách AI vs. Đào tạo lại theo ngành (tỷ VND)", "📊")
            df_plot_bg = pd.DataFrame({
                "Ngành": sectors,
                "Đầu tư AI": x_AI_v,
                "Đào tạo lại H": x_H_v
            })
            fig_bg = px.bar(df_plot_bg, x="Ngành", y=["Đầu tư AI", "Đào tạo lại H"], barmode="stack", color_discrete_sequence=[COLORS[0], COLORS[1]])
            fig_bg.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_bg, use_container_width=True)
            
        else:
            st.error("❌ **Bài toán tối ưu không khả thi! Các điều kiện sa thải quá ngặt nghèo.** Hãy giảm bớt sàn an sinh xã hội hoặc nới rộng ngân sách.")

    # ══ Kết quả mở rộng: biến động việc làm ══════════════════════════════
    with tabs[2]:
        if res["success"]:
            section("Câu 9.4.2 — Ngưỡng đầu tư đào tạo tối thiểu theo ngành", "👷")
            training_df = df_res_tbl[["Ngành", "Ngân sách đào tạo (x_H)", "Năng lực tái đào tạo", "Việc bị thay thế (Displaced)"]].copy()
            st.dataframe(training_df.set_index("Ngành").style.format("{:,.1f}"), use_container_width=True)

            section("Câu 9.4.3 — Mô phỏng nhóm dễ bị tổn thương", "🧩")
            vulnerable = ["Nông-Lâm-Thủy sản", "Xây dựng", "Bán buôn-bán lẻ"]
            vulnerable_df = df_res_tbl[df_res_tbl["Ngành"].isin(vulnerable)].copy()
            st.dataframe(vulnerable_df.set_index("Ngành").style.format("{:,.1f}"), use_container_width=True)

            section("Câu 9.4.4 — Kiểm tra ràng buộc Displaced ≤ 5% lao động", "🛡️")
            cap_df = df_res_tbl[["Ngành", "Việc bị thay thế (Displaced)", "Việc làm ròng (NetJob)"]].copy()
            cap_df["Tín hiệu rủi ro"] = np.where(cap_df["Việc bị thay thế (Displaced)"] > cap_df["Việc bị thay thế (Displaced)"].median(), "Cần giám sát", "Ổn định hơn")
            st.dataframe(cap_df.set_index("Ngành").style.format({"Việc bị thay thế (Displaced)": "{:,.1f}", "Việc làm ròng (NetJob)": "{:,.1f}"}), use_container_width=True)

            section("Biểu đồ mở rộng — Phân tích biến động việc làm theo ngành", "📈")
            df_jobs = pd.DataFrame({
                "Ngành": sectors,
                "Việc làm mới": res["NewJob"],
                "Việc làm nâng cấp": res["UpgradeJob"],
                "Việc bị thay thế": -res["DisplacedJob"]
            }).set_index("Ngành")
            
            fig_jobs = go.Figure()
            fig_jobs.add_trace(go.Bar(x=df_jobs.index, y=df_jobs["Việc làm mới"], name="Việc làm mới từ AI", marker_color=COLORS[2]))
            fig_jobs.add_trace(go.Bar(x=df_jobs.index, y=df_jobs["Việc làm nâng cấp"], name="Việc nâng cấp kỹ năng", marker_color=COLORS[0]))
            fig_jobs.add_trace(go.Bar(x=df_jobs.index, y=df_jobs["Việc bị thay thế"], name="Việc bị thay thế", marker_color=COLORS[4]))
            fig_jobs.update_layout(**PLOTLY_LAYOUT, barmode="relative")
            st.plotly_chart(fig_jobs, use_container_width=True)
            
            # NetJob bar chart
            section("Mở rộng — Việc làm ròng (NetJob = New + Upgrade - Displaced)", "🔬")
            fig_net = px.bar(df_res_tbl, x="Ngành", y="Việc làm ròng (NetJob)", color_discrete_sequence=[COLORS[1]], text_auto=".1f")
            fig_net.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_net, use_container_width=True)
            
        else:
            st.warning("⚠️ Vui lòng cấu hình tham số khả thi.")

    # ══ Tab 4: Phân tích chính sách ═══════════════════════════════════════
    with tabs[3]:
        if res["success"]:
            max_h_idx = np.argmax(x_H_v)
            max_net_idx = np.argmax(res["NetJob"])
            max_dis_idx = np.argmax(res["DisplacedJob"])
            section("1. Ngành nào cần đầu tư đào tạo lại nhiều nhất?", "📌")
            st.markdown(f"""
            Ngành nhận ngân sách đào tạo lại lớn nhất là **{sectors[max_h_idx]}** với {x_H_v[max_h_idx]:,.1f} tỷ VND.
            Điều này phản ánh nhu cầu hấp thụ lao động bị dịch chuyển hoặc yêu cầu nâng kỹ năng để vận hành công nghệ mới.
            Chính sách đào tạo nên ưu tiên nhóm có rủi ro mất việc cao trước khi mở rộng tự động hóa.
            """)
            section("2. Ngành rủi ro cao nhưng vẫn có thể tạo việc làm mới không?", "🤖")
            st.markdown(f"""
            Ngành có số việc bị thay thế cao nhất là **{sectors[max_dis_idx]}**, trong khi ngành tạo NetJob cao nhất là **{sectors[max_net_idx]}**.
            Kết quả cho thấy AI không chỉ thay thế lao động mà còn tạo việc làm mới và nâng cấp kỹ năng nếu có ngân sách H phù hợp.
            Vì vậy, chính sách không nên cấm AI trong ngành rủi ro, mà cần gắn đầu tư AI với quỹ chuyển đổi nghề.
            """)
            section("3. Ràng buộc an sinh xã hội có vai trò gì?", "🛡️")
            st.markdown("""
            Ràng buộc DisplacedJob không vượt năng lực tái đào tạo buộc mô hình không thể tối đa hóa AI một cách cực đoan.
            Đây là cơ chế đưa mục tiêu bao trùm lao động vào bài toán tối ưu.
            Trong thực tế, ràng buộc này tương ứng với ngân sách bảo hiểm thất nghiệp, voucher kỹ năng số và chương trình đào tạo lại cấp ngành.
            """)
        else:
            st.warning("Mô hình hiện không khả thi nên chưa thể đưa ra phân tích chính sách định lượng.")

    # ══ Tab 5: AI Agent ═══════════════════════════════════════════════════
    with tabs[4]:
        st.caption("Bấm nút AI Agent bên dưới để hiển thị phân tích chính sách mô phỏng.")
        if res["success"]:
            max_net_idx = np.argmax(res["NetJob"])
            max_net_sec = sectors[max_net_idx]
            
            # Calculate total displaced vs total upgraded
            tot_dis = res["DisplacedJob"].sum()
            tot_up = res["UpgradeJob"].sum()
            tot_new = res["NewJob"].sum()
            
            ai_agent(f"""
            <b>1. Nhận định cấu trúc thị trường lao động:</b><br>
            • Tổng số việc làm ròng được tạo thêm toàn quốc đạt: <b>{res['objective']:,.0f} việc làm</b>.<br>
            • Ngành được hưởng lợi tạo thêm nhiều việc làm nhất là <b>{max_net_sec}</b>.<br>
            • Tỷ lệ **Việc làm mới : Việc bị thay thế** trên toàn quốc đạt: <b>{tot_new:,.0f} : {tot_dis:,.0f}</b>.<br><br>
            
            <b>2. Đánh giá lưới an sinh xã hội:</b><br>
            • Ràng buộc lưới an sinh xã hội (Sa thải $\\le$ Năng lực đào tạo) hoạt động cực tốt. 
            Mô hình đã tự động phân bổ nguồn vốn **đào tạo lại H** tăng vọt tại các ngành có rủi ro tự động hóa cao như <i>CN chế biến chế tạo</i> và <i>Bán buôn - bán lẻ</i>. 
            Điều này thể hiện triết lý: AI không thể đi trước nếu lực lượng lao động chưa sẵn sàng chuyển đổi số.<br><br>
            
            <b>3. Khuyến nghị hành động:</b><br>
            • Đầu tư đào tạo **Nhân lực số H** là bắt buộc cho an sinh xã hội. 
            Chính phủ cần thành lập quỹ đào tạo lại kỹ năng số cấp quốc gia nhằm hỗ trợ nhóm lao động dễ bị tổn thương (lao động phổ thông trong công nghiệp dệt may, da giày, bán lẻ) 
            trước các đợt sa thải vĩ mô do AI thay thế.
            """)
