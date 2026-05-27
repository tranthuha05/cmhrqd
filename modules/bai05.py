"""
Bài 5 — Quy hoạch nguyên hỗn hợp MIP lựa chọn dự án chuyển đổi số
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
projects = list(range(1, 16))
project_name = {
    1: "Trung tâm dữ liệu quốc gia Hòa Lạc",
    2: "Trung tâm dữ liệu quốc gia phía Nam",
    3: "Hệ thống 5G phủ sóng toàn quốc",
    4: "Hệ thống định danh điện tử VNeID 2.0",
    5: "Cổng dịch vụ công quốc gia v3",
    6: "Y tế số quốc gia",
    7: "Giáo dục số K-12 toàn quốc",
    8: "Trung tâm AI quốc gia + supercomputing",
    9: "Sandbox tài chính số",
    10: "Logistics thông minh + cảng biển số",
    11: "Nông nghiệp số ĐBSCL",
    12: "Đào tạo 50.000 kỹ sư AI/bán dẫn",
    13: "Khu CN bán dẫn Bắc Ninh - Bắc Giang",
    14: "An ninh mạng quốc gia SOC",
    15: "Open Data + dữ liệu mở quốc gia"
}
field = {
    1: "Hạ tầng", 2: "Hạ tầng", 3: "Hạ tầng",
    4: "Chính phủ số", 5: "Chính phủ số",
    6: "Y tế số", 7: "Giáo dục", 8: "AI",
    9: "Tài chính số", 10: "Logistics", 11: "Nông nghiệp",
    12: "Nhân lực", 13: "Bán dẫn", 14: "An ninh", 15: "Dữ liệu"
}
C = {
    1:12000, 2:11500, 3:18000, 4:4500, 5:3200,
    6:5800, 7:6500, 8:15000, 9:2500, 10:7200,
    11:4800, 12:8500, 13:20000, 14:3800, 15:1500
}
C1 = {
    1:8500, 2:7500, 3:12000, 4:3500, 5:2500,
    6:4000, 7:4500, 8:9000, 9:1800, 10:5000,
    11:3500, 12:5500, 13:13000, 14:2800, 15:1200
}
B = {
    1:21500, 2:20800, 3:32500, 4:9200, 5:6800,
    6:11400, 7:12200, 8:28500, 9:5800, 10:13800,
    11:8500, 12:16200, 13:35000, 14:7500, 15:3800
}

def solve_mip(total_budget, early_budget, allow_both_dc, force_soc, min_proj, max_proj):
    model = pulp.LpProblem("VN_Project_Selection", pulp.LpMaximize)
    y = pulp.LpVariable.dicts("y", projects, cat="Binary")
    
    # Objective
    model += pulp.lpSum(B[i] * y[i] for i in projects), "Total_NPV_Benefit"
    
    # Constraints
    # C1: Ngân sách tổng
    model += pulp.lpSum(C[i] * y[i] for i in projects) <= total_budget, "Total_Budget_Limit"
    
    # C2: Ngân sách năm 1-2
    model += pulp.lpSum(C1[i] * y[i] for i in projects) <= early_budget, "Early_Budget_Limit"
    
    # C3: Loại trừ hoặc ràng buộc trung tâm dữ liệu
    if not allow_both_dc:
        model += y[1] + y[2] <= 1, "DC_Exclusion"
        
    # C4 & C5: Tiên quyết (AI và bán dẫn cần đào tạo kỹ sư)
    model += y[8] <= y[12], "AI_Prerequisite"
    model += y[13] <= y[12], "Semiconductor_Prerequisite"
    
    # C6: Cân đối lĩnh vực
    model += y[4] + y[5] >= 1, "Gov_Balance"
    if force_soc:
        model += y[14] == 1, "Force_SOC_Security"
        
    # C7: Số lượng dự án chọn
    model += pulp.lpSum(y[i] for i in projects) >= min_proj, "Min_Projects"
    model += pulp.lpSum(y[i] for i in projects) <= max_proj, "Max_Projects"
    
    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[status] == "Optimal":
        y_val = {i: int(pulp.value(y[i])) for i in projects}
        total_npv = pulp.value(model.objective)
        return {
            "success": True,
            "y": y_val,
            "total_npv": total_npv
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
        <span style="font-size:2rem;">🎯</span>
        <div>
          <div style="color:#00D4FF;font-size:.75rem;font-weight:600;letter-spacing:1px;">BÀI 5</div>
          <div style="color:#E2E8F0;font-size:1.5rem;font-weight:800;">Quy hoạch nguyên hỗn hợp (MIP) — Lựa chọn dự án tối ưu</div>
          <div style="margin-top:.3rem;">
            <span class="badge b-blue">Mixed-Integer Programming (MIP)</span>
            <span class="badge b-purple">Capital Rationing</span>
            <span class="badge b-green">Prerequisite Constraints</span>
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
        Chương trình chuyển đổi số quốc gia giai đoạn 2026–2030 đề xuất **15 dự án chiến lược** cạnh tranh để nhận vốn. 
        Tổng ngân sách của chương trình có hạn (**80.000 tỷ VND**) và ngân sách giải ngân sớm cho 2 năm đầu tiên bị giới hạn ở 
        **40.000 tỷ VND**. 
        
        Mỗi dự án có chi phí, lợi ích NPV dự kiến, thời gian giải ngân, và mối quan hệ ràng buộc logic.
        Mục tiêu là tìm danh mục tối ưu các dự án được chọn sao cho **tổng lợi ích NPV đạt tối đa** mà không vi phạm 
        bất cứ giới hạn nguồn lực hay tính đồng bộ chính trị nào.
        """)

        section("Mô hình toán học", "📐")
        math_box("""
<b>Biến quyết định:</b>
  y<sub>i</sub> ∈ {0, 1} : y<sub>i</sub> = 1 nếu chọn dự án i; y<sub>i</sub> = 0 nếu không chọn.

<b>Hàm mục tiêu:</b>
  max Z = Σ<sub>i=1..15</sub> B<sub>i</sub> · y<sub>i</sub>   (Tối đa tổng lợi ích NPV)

<b>Các ràng buộc cốt lõi:</b>
  1. Ngân sách tổng 5 năm:  Σ<sub>i</sub> C<sub>i</sub> · y<sub>i</sub> ≤ B_total  (80.000 tỷ)
  2. Ngân sách năm 1–2:  Σ<sub>i</sub> C1<sub>i</sub> · y<sub>i</sub> ≤ B_early  (40.000 tỷ)
  3. Loại trừ: y<sub>1</sub> + y<sub>2</sub> ≤ 1  (Chỉ chọn nhiều nhất 1 trong 2 Trung tâm dữ liệu Hòa Lạc hoặc phía Nam)
  4. Tiên quyết: y<sub>8</sub> ≤ y<sub>12</sub>  (Chỉ chọn trung tâm AI khi đã chọn dự án đào tạo 50.000 kỹ sư)
  5. Tiên quyết: y<sub>13</sub> ≤ y<sub>12</sub> (Chỉ chọn khu bán dẫn khi đã chọn dự án đào tạo kỹ sư)
  6. Cân đối chính phủ số: y<sub>4</sub> + y<sub>5</sub> ≥ 1  (Bắt buộc chọn ít nhất định danh VNeID 2.0 hoặc Cổng dịch vụ công v3)
  7. Bắt buộc an ninh mạng: y<sub>14</sub> = 1  (Bắt buộc chọn dự án SOC)
  8. Số lượng dự án:  Min_Proj ≤ Σ<sub>i</sub> y<sub>i</sub> ≤ Max_Proj
        """)

    # ══ Tab 2: Ràng buộc & Tham số ════════════════════════════════════════
    with tabs[1]:
        section("Điều chỉnh các ràng buộc ngân sách & chính sách", "🎛️")
        col1, col2 = st.columns(2)
        with col1:
            total_budget = st.slider("💰 Tổng ngân sách 5 năm (tỷ VND)", 60000, 120000, 80000, 2000, key="b5_tb")
            early_budget = st.slider("⏳ Ngân sách giới hạn năm 1-2 (tỷ VND)", 30000, 60000, 40000, 1000, key="b5_eb")
            allow_both_dc = st.checkbox("Cho phép xây dựng cả 2 Trung tâm dữ liệu quốc gia", value=False, key="b5_both_dc")
        with col2:
            force_soc = st.checkbox("Bắt buộc đầu tư dự án An ninh mạng quốc gia SOC (P14)", value=True, key="b5_force_soc")
            min_proj = st.number_input("📉 Số dự án tối thiểu được chọn", min_value=5, max_value=10, value=7, key="b5_min_p")
            max_proj = st.number_input("📈 Số dự án tối đa được chọn", min_value=10, max_value=15, value=11, key="b5_max_p")

        section("Bảng thông số 15 dự án đầu tư", "📋")
        df_p_data = pd.DataFrame({
            "Mã": [f"P{i}" for i in projects],
            "Tên dự án": [project_name[i] for i in projects],
            "Lĩnh vực": [field[i] for i in projects],
            "Chi phí tổng C (tỷ)": [C[i] for i in projects],
            "NPV Lợi ích B (tỷ)": [B[i] for i in projects],
            "Chi phí Năm 1-2": [C1[i] for i in projects],
            "Hiệu suất (NPV/C)": [B[i]/C[i] for i in projects]
        })
        st.dataframe(df_p_data.set_index("Mã").style.format({
            "Chi phí tổng C (tỷ)": "{:,.0f}",
            "NPV Lợi ích B (tỷ)": "{:,.0f}",
            "Chi phí Năm 1-2": "{:,.0f}",
            "Hiệu suất (NPV/C)": "{:.2f}"
        }), use_container_width=True)

    # ══ Tab 3: Kết quả Lựa chọn ════════════════════════════════════════════
    with tabs[2]:
        res = solve_mip(total_budget, early_budget, allow_both_dc, force_soc, min_proj, max_proj)
        
        if res["success"]:
            st.success("🎉 **Tìm thấy danh mục đầu tư tối ưu nhất!**")
            
            y_opt = res["y"]
            total_npv = res["total_npv"]
            
            # Filter selected
            df_selected = df_p_data.copy()
            df_selected["Được chọn"] = [y_opt[i] for i in projects]
            df_sel_only = df_selected[df_selected["Được chọn"] == 1].reset_index(drop=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tổng NPV Benefit", f"{total_npv:,.0f} tỷ")
            c2.metric("Số dự án được chọn", f"{len(df_sel_only)}")
            c3.metric("Tổng chi phí tổng", f"{df_sel_only['Chi phí tổng C (tỷ)'].sum():,.0f} tỷ")
            c4.metric("Chi phí năm 1-2", f"{df_sel_only['Chi phí Năm 1-2'].sum():,.0f} tỷ")

            section("Câu 5.4.1 — Giải mô hình cơ sở bằng PuLP CBC", "🏆")
            st.dataframe(df_sel_only[["Mã", "Tên dự án", "Lĩnh vực", "Chi phí tổng C (tỷ)", "NPV Lợi ích B (tỷ)", "Hiệu suất (NPV/C)"]].style.format({
                "Chi phí tổng C (tỷ)": "{:,.0f}",
                "NPV Lợi ích B (tỷ)": "{:,.0f}",
                "Hiệu suất (NPV/C)": "{:.2f}"
            }), use_container_width=True)

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                section("Cơ cấu chi phí theo Lĩnh vực (Tỷ VND)", "📊")
                df_field = df_sel_only.groupby("Lĩnh vực")["Chi phí tổng C (tỷ)"].sum().reset_index()
                fig_pie = px.pie(df_field, names="Lĩnh vực", values="Chi phí tổng C (tỷ)", hole=0.4, color_discrete_sequence=COLORS)
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"), margin=dict(t=30, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_g2:
                section("So sánh Lợi ích vs. Chi phí các dự án được duyệt", "📈")
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(x=df_sel_only["Mã"], y=df_sel_only["Chi phí tổng C (tỷ)"], name="Chi phí", marker_color=COLORS[1]))
                fig_comp.add_trace(go.Bar(x=df_sel_only["Mã"], y=df_sel_only["NPV Lợi ích B (tỷ)"], name="NPV Lợi ích", marker_color=COLORS[0]))
                fig_comp.update_layout(**PLOTLY_LAYOUT, barmode="group")
                st.plotly_chart(fig_comp, use_container_width=True)

            section("Câu 5.4.2 — Phân tích trường hợp nới ngân sách lên 100.000", "💰")
            res_100 = solve_mip(100000, early_budget, allow_both_dc, force_soc, min_proj, max_proj)
            if res_100["success"]:
                y100 = res_100["y"]
                df_100 = df_p_data.copy()
                df_100["Được chọn"] = [y100[i] for i in projects]
                df_100_sel = df_100[df_100["Được chọn"] == 1].reset_index(drop=True)
                compare_budget_df = pd.DataFrame({
                    "Kịch bản": ["Ngân sách hiện tại", "Ngân sách 100.000"],
                    "Tổng NPV": [total_npv, res_100["total_npv"]],
                    "Số dự án": [len(df_sel_only), len(df_100_sel)],
                    "Tổng chi phí": [df_sel_only["Chi phí tổng C (tỷ)"].sum(), df_100_sel["Chi phí tổng C (tỷ)"].sum()],
                })
                st.dataframe(compare_budget_df.style.format({"Tổng NPV": "{:,.0f}", "Tổng chi phí": "{:,.0f}"}), use_container_width=True)
            else:
                st.warning("Mô hình ngân sách 100.000 không có nghiệm tối ưu với cấu hình hiện tại.")

            section("Câu 5.4.3 — Kịch bản bắt buộc/cho phép cả P1 và P2", "🏛️")
            res_both = solve_mip(total_budget, early_budget, True, force_soc, min_proj, max_proj)
            if res_both["success"]:
                y_both = res_both["y"]
                st.dataframe(pd.DataFrame({
                    "Chỉ tiêu": ["P1 được chọn", "P2 được chọn", "Tổng NPV"],
                    "Giá trị": [str(y_both[1]), str(y_both[2]), f"{res_both['total_npv']:,.0f}"],
                }), use_container_width=True)
            else:
                st.warning("Kịch bản cho phép cả P1 và P2 không khả thi với cấu hình hiện tại.")

            section("Câu 5.4.4 — Mở rộng rủi ro dự án", "⚠️")
            risk_prob = {"Hạ tầng": 0.85, "Chính phủ số": 0.75, "AI": 0.65, "Bán dẫn": 0.65}
            risk_df = df_selected.copy()
            risk_df["Xác suất hoàn thành"] = risk_df["Lĩnh vực"].map(risk_prob).fillna(0.80)
            risk_df["NPV kỳ vọng sau rủi ro"] = risk_df["NPV Lợi ích B (tỷ)"] * risk_df["Xác suất hoàn thành"]
            st.dataframe(risk_df[["Mã", "Tên dự án", "Lĩnh vực", "Được chọn", "NPV Lợi ích B (tỷ)", "Xác suất hoàn thành", "NPV kỳ vọng sau rủi ro"]].style.format({
                "NPV Lợi ích B (tỷ)": "{:,.0f}",
                "Xác suất hoàn thành": "{:.0%}",
                "NPV kỳ vọng sau rủi ro": "{:,.0f}",
            }), use_container_width=True)
        else:
            st.error("❌ **Bài toán MIP không khả thi (Infeasible). Các ràng buộc quá chặt chẽ, vui lòng nới rộng ngân sách hoặc giảm số lượng dự án tối thiểu!**")

    # ══ Kết quả mở rộng: đánh giá dự án ═══════════════════════════════════
    with tabs[2]:
        section("Mở rộng — Đánh giá hiệu suất đầu tư (NPV / Chi phí)", "🔬")
        fig_eff = px.bar(df_p_data.sort_values("Hiệu suất (NPV/C)", ascending=False), x="Mã", y="Hiệu suất (NPV/C)",
                         color="Lĩnh vực", hover_name="Tên dự án", color_discrete_sequence=COLORS)
        fig_eff.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_eff, use_container_width=True)
        st.markdown("""
        **Nhận xét quy luật đầu tư:**
        - Các dự án có **NPV/C cao** (như VNeID 2.0 - 2.04, Cổng Dịch vụ công - 2.12, Supercomputing AI - 1.90) sẽ được mô hình MIP ưu tiên duyệt đầu tiên.
        - Dự án **Khu CN bán dẫn Bắc Ninh - Bắc Giang (P13)** mặc dù có NPV rất cao (35.000 tỷ VND) nhưng có chi phí khổng lồ (20.000 tỷ VND, chiếm 25% ngân sách tổng) nên thường chỉ được duyệt khi tổng ngân sách được nới rộng đáng kể.
        - Dự án **Đào tạo 50.000 kỹ sư (P12)** là dự án bản lề. Nhờ ràng buộc tiên quyết logic, nó hoạt động như một "chìa khóa" mở cổng duyệt cho dự án siêu máy tính AI (P8) và bán dẫn (P13).
        """)

    # ══ Tab 4: Phân tích chính sách ═══════════════════════════════════════
    with tabs[3]:
        section("1. Vì sao mô hình có thể bỏ qua dự án tỷ suất cao?", "📌")
        st.markdown("""
        MIP tối đa hóa **tổng NPV** dưới nhiều ràng buộc đồng thời, không tối đa hóa riêng tỷ suất NPV/chi phí.
        Một dự án nhỏ có tỷ suất cao vẫn có thể bị loại nếu không phù hợp ràng buộc ngân sách sớm, số lượng dự án, tiên quyết hoặc cân đối lĩnh vực.
        Vì vậy, kết quả cần được đọc như một danh mục đầu tư tối ưu tổng thể, không phải bảng xếp hạng hiệu suất đơn lẻ.
        """)
        section("2. Có nên bắt buộc dự án an ninh mạng SOC?", "🛡️")
        st.markdown(f"""
        Cấu hình hiện tại {'đang' if force_soc else 'không'} bắt buộc P14.
        Nếu SOC vẫn được chọn dù không bắt buộc, ràng buộc này không tạo chi phí kinh tế đáng kể trong mô hình.
        Nếu SOC chỉ xuất hiện khi bị bắt buộc, đó là đánh đổi giữa hiệu quả NPV và an ninh quốc gia, cần được trình bày minh bạch trong quyết định đầu tư.
        """)
        section("3. Nới ngân sách có tạo thêm giá trị không?", "💰")
        st.markdown("""
        Kịch bản ngân sách 100.000 tỷ cho thấy khi nguồn lực tăng, mô hình có thể mở thêm các dự án chi phí lớn như AI, bán dẫn hoặc hạ tầng lõi.
        Tuy nhiên, nới ngân sách chỉ có ý nghĩa nếu năng lực giải ngân 2 năm đầu và ràng buộc tiên quyết nhân lực không trở thành nút thắt mới.
        Chính sách nên đi kèm kế hoạch triển khai theo giai đoạn để tránh chọn danh mục lớn nhưng không giải ngân được.
        """)

    # ══ Tab 5: AI Agent ═══════════════════════════════════════════════════
    with tabs[4]:
        st.caption("Bấm nút AI Agent bên dưới để hiển thị phân tích chính sách mô phỏng.")
        if res["success"]:
            y_opt = res["y"]
            dc_selected = []
            if y_opt[1] == 1: dc_selected.append("Hòa Lạc")
            if y_opt[2] == 1: dc_selected.append("Phía Nam")
            
            ai_agent(f"""
            <b>1. Đánh giá tính đồng bộ của danh mục tối ưu:</b><br>
            • Ràng buộc loại trừ Trung tâm dữ liệu quốc gia hoạt động hoàn hảo: Chỉ duy nhất <b>Trung tâm dữ liệu {', '.join(dc_selected) if dc_selected else 'không có'}</b> được chọn.<br>
            • Dự án đào tạo <b>50.000 kỹ sư AI/bán dẫn (P12)</b> đã được phê duyệt làm cơ sở tiền đề, từ đó mở khóa thành công cho <b>{ 'dự án siêu máy tính AI (P8)' if y_opt[8]==1 else 'nhưng siêu máy tính AI (P8) vẫn chưa được chọn do giới hạn ngân sách' }</b> và <b>{ 'khu công nghệ bán dẫn (P13)' if y_opt[13]==1 else 'nhưng khu bán dẫn (P13) bị loại bỏ' }</b>.<br><br>
            
            <b>2. Khuyến nghị chính sách:</b><br>
            {'⚠️ Dự án An ninh mạng SOC (P14) là bắt buộc chính trị nên đã được duyệt mặc dù hiệu suất NPV/C chỉ ở mức trung bình (1.97).' if force_soc else '✅ Dự án SOC không bị bắt buộc.'}<br>
            • Tổng NPV đạt **{res['total_npv']:,.0f} tỷ VND** là mức sinh lời tối đa tuyệt đối trên phạm vi toán học. 
            Để tăng hiệu quả triển khai, Chính phủ cần chú ý giải ngân đúng tiến độ của 2 năm đầu tiên ({df_sel_only['Chi phí Năm 1-2'].sum():,.0f} tỷ VND), 
            tránh tình trạng dồn ứ giải ngân vào các năm cuối gây lãng phí nguồn lực đầu tư số.
            """)
        else:
            ai_agent("<b>⚠️ Phân tích lỗi Infeasible:</b><br>Các ràng buộc về chính phủ số, an ninh mạng, hoặc số lượng dự án tối thiểu vượt quá khả năng chi trả của dòng ngân sách 5 năm. Cần nới rộng ngân sách để tiếp tục phân tích.")
