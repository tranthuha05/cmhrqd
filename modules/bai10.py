from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,32,0.6)",
    font=dict(color="#94A3B8", family="Inter"),
    xaxis=dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.1)"),
    margin=dict(l=40, r=20, t=50, b=40),
)
COLORS = ["#00D4FF", "#6366F1", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6"]

J = ["I", "D", "AI", "H"]
J_name = {"I": "Hạ tầng số", "D": "Chuyển đổi số", "AI": "Trí tuệ nhân tạo", "H": "Nhân lực số"}
S = ["s1", "s2", "s3", "s4"]
S_name = {"s1": "Lạc quan", "s2": "Cơ sở", "s3": "Bi quan", "s4": "Khủng hoảng"}
beta = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}
beta_s = {
    ("s1", "I"): 1.25, ("s1", "D"): 1.35, ("s1", "AI"): 1.55, ("s1", "H"): 1.05,
    ("s2", "I"): 1.00, ("s2", "D"): 1.10, ("s2", "AI"): 1.25, ("s2", "H"): 0.95,
    ("s3", "I"): 0.75, ("s3", "D"): 0.85, ("s3", "AI"): 0.90, ("s3", "H"): 1.00,
    ("s4", "I"): 0.40, ("s4", "D"): 0.50, ("s4", "AI"): 0.55, ("s4", "H"): 1.10,
}


def section(title: str, icon: str = "📌") -> None:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,rgba(0,212,255,.07),rgba(99,102,241,.05));
             border:1px solid rgba(0,212,255,.14);border-radius:10px;padding:.7rem 1.2rem;margin:.8rem 0 .5rem;">
          <span style="color:#00D4FF;font-size:1.05rem;font-weight:700;">{icon} {title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def math_box(text: str) -> None:
    st.markdown(f'<div class="math-box">{text}</div>', unsafe_allow_html=True)


def ai_agent(text: str) -> None:
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
def _normalize_prob(prob_dict: dict[str, float]) -> dict[str, float]:
    total = sum(max(v, 0.0) for v in prob_dict.values())
    if total <= 0:
        return {"s1": 0.30, "s2": 0.45, "s3": 0.20, "s4": 0.05}
    return {s: max(prob_dict[s], 0.0) / total for s in S}


def solve_sp_model(prob_dict: dict[str, float], beta_s_input: dict[tuple[str, str], float] | None = None, fixed_x: dict[str, float] | None = None):
    beta_s_input = beta_s if beta_s_input is None else beta_s_input
    n = 20
    c = np.zeros(n)
    for j_idx, j in enumerate(J):
        c[j_idx] = -beta[j]
    for s_idx, s in enumerate(S):
        for j_idx, j in enumerate(J):
            c[4 + 4 * s_idx + j_idx] = -prob_dict[s] * beta_s_input[(s, j)]

    A_ub, b_ub = [], []
    row = np.zeros(n)
    row[:4] = 1.0
    A_ub.append(row)
    b_ub.append(65000.0)

    for s_idx in range(len(S)):
        row = np.zeros(n)
        row[4 + 4 * s_idx : 8 + 4 * s_idx] = 1.0
        A_ub.append(row)
        b_ub.append(15000.0)

    for s_idx in range(len(S)):
        row = np.zeros(n)
        row[3] = -0.5
        row[6 + 4 * s_idx] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    A_eq, b_eq = [], []
    if fixed_x is not None:
        for j_idx, j in enumerate(J):
            row = np.zeros(n)
            row[j_idx] = 1.0
            A_eq.append(row)
            b_eq.append(float(fixed_x[j]))

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=[(0.0, None)] * n,
        method="highs",
    )
    if not res.success:
        return None

    x_sol = {j: res.x[j_idx] for j_idx, j in enumerate(J)}
    y_sol = {}
    for s_idx, s in enumerate(S):
        for j_idx, j in enumerate(J):
            y_sol[(s, j)] = res.x[4 + 4 * s_idx + j_idx]
    return {"x": x_sol, "y": y_sol, "objective": -res.fun, "status": "Optimal"}


def solve_deterministic_for_scenario(scenario: str):
    scenario_prob = {s: 1.0 if s == scenario else 0.0 for s in S}
    return solve_sp_model(scenario_prob)


def solve_robust_regret(det_solutions: dict[str, dict]):
    n = 21
    c = np.zeros(n)
    c[-1] = 1.0
    A_ub, b_ub = [], []

    row = np.zeros(n)
    row[:4] = 1.0
    A_ub.append(row)
    b_ub.append(65000.0)

    for s_idx in range(len(S)):
        row = np.zeros(n)
        row[4 + 4 * s_idx : 8 + 4 * s_idx] = 1.0
        A_ub.append(row)
        b_ub.append(15000.0)

    for s_idx in range(len(S)):
        row = np.zeros(n)
        row[3] = -0.5
        row[6 + 4 * s_idx] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    for s_idx, s in enumerate(S):
        row = np.zeros(n)
        for j_idx, j in enumerate(J):
            row[j_idx] = -beta[j]
            row[4 + 4 * s_idx + j_idx] = -beta_s[(s, j)]
        row[-1] = -1.0
        A_ub.append(row)
        b_ub.append(-det_solutions[s]["objective"])

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0.0, None)] * n, method="highs")
    if not res.success:
        return None
    x_sol = {j: res.x[j_idx] for j_idx, j in enumerate(J)}
    y_sol = {}
    for s_idx, s in enumerate(S):
        for j_idx, j in enumerate(J):
            y_sol[(s, j)] = res.x[4 + 4 * s_idx + j_idx]
    return {"x": x_sol, "y": y_sol, "max_regret": res.x[-1], "status": "Optimal"}


def _result_tables(prob_dict: dict[str, float]):
    sp_solution = solve_sp_model(prob_dict)
    det_solutions = {s: solve_deterministic_for_scenario(s) for s in S}
    beta_ev = {j: sum(prob_dict[s] * beta_s[(s, j)] for s in S) for j in J}
    beta_s_ev = {(s, j): beta_ev[j] for s in S for j in J}
    ev_solution = solve_sp_model(prob_dict, beta_s_ev)
    eev_solution = solve_sp_model(prob_dict, beta_s, fixed_x=ev_solution["x"])
    robust_solution = solve_robust_regret(det_solutions)

    RP = sp_solution["objective"]
    EEV = eev_solution["objective"]
    VSS = RP - EEV
    WS = sum(prob_dict[s] * det_solutions[s]["objective"] for s in S)
    EVPI = WS - RP
    return sp_solution, det_solutions, ev_solution, robust_solution, {"RP": RP, "EEV": EEV, "VSS": VSS, "WS": WS, "EVPI": EVPI}


def show() -> None:
    st.markdown(
        """
        <div class="page-header">
          <div style="display:flex;align-items:center;gap:1rem;">
            <span style="font-size:2rem;">🎲</span>
            <div>
              <div style="color:#00D4FF;font-size:.75rem;font-weight:600;letter-spacing:1px;">BÀI 10</div>
              <div style="color:#E2E8F0;font-size:1.5rem;font-weight:800;">Stochastic Programming — Quy hoạch ngẫu nhiên hai giai đoạn</div>
              <div style="margin-top:.3rem;">
                <span class="badge b-blue">Two-stage SP</span>
                <span class="badge b-purple">VSS & EVPI</span>
                <span class="badge b-green">SciPy HiGHS solver</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["📖 Bối cảnh & Mô hình", "🎛️ Tham số", "📊 Kết quả", "📈 Phân tích chính sách", "🤖 AI Agent"])

    with tabs[0]:
        section("Bối cảnh bài toán", "🌏")
        st.markdown(
            """
            Việt Nam là nền kinh tế có độ mở thương mại cao, phụ thuộc lớn vào cầu xuất khẩu,
            dòng vốn FDI và môi trường kinh tế toàn cầu. Khi hoạch định chính sách đầu tư số
            giai đoạn 2026-2030, Chính phủ phải phân bổ ngân sách ban đầu khi chưa biết chắc
            kịch bản tương lai sẽ xảy ra.
            """
        )
        section("Mô hình toán học", "📐")
        math_box(
            """
            <b>Giai đoạn 1:</b> x<sub>j</sub> ≥ 0, Σ<sub>j</sub>x<sub>j</sub> ≤ 65.000.<br>
            <b>Giai đoạn 2:</b> y<sub>s,j</sub> ≥ 0, Σ<sub>j</sub>y<sub>s,j</sub> ≤ 15.000 với mọi kịch bản s.<br>
            <b>Năng lực AI:</b> y<sub>s,AI</sub> ≤ 0.5*x<sub>H</sub>.<br>
            <b>Mục tiêu:</b> max Σ<sub>j</sub>β<sub>j</sub>x<sub>j</sub> + Σ<sub>s</sub>p<sub>s</sub>Σ<sub>j</sub>β<sub>s,j</sub>y<sub>s,j</sub>.
            """
        )

    with tabs[1]:
        section("Xác suất kịch bản", "🎛️")
        c1, c2 = st.columns(2)
        with c1:
            p1 = st.slider("p1 — Lạc quan (%)", 5, 80, 30, 5, key="b10_p1") / 100
            p2 = st.slider("p2 — Cơ sở (%)", 5, 80, 45, 5, key="b10_p2") / 100
        with c2:
            p3 = st.slider("p3 — Bi quan (%)", 5, 80, 20, 5, key="b10_p3") / 100
            p4 = 1 - p1 - p2 - p3
            if p4 < 0:
                st.warning("Tổng ba xác suất đầu vượt 100%; hệ thống tự chuẩn hóa để mô hình vẫn chạy.")
        prob_dict = _normalize_prob({"s1": p1, "s2": p2, "s3": p3, "s4": max(p4, 0.0)})

        scenario_df = pd.DataFrame({
            "Kịch bản": [S_name[s] for s in S],
            "Tăng trưởng TG (%)": [3.5, 2.8, 1.5, 0.2],
            "FDI 🇻🇳 (tỷ USD/năm)": [32.0, 27.0, 20.0, 12.0],
            "Xuất khẩu 🇻🇳 tăng (%)": [12.0, 8.0, 3.0, -5.0],
            "Xác suất": [prob_dict[s] for s in S],
        })
        st.dataframe(scenario_df.style.format({"Xác suất": "{:.1%}"}), use_container_width=True)

        beta_s_df = pd.DataFrame([[S_name[s]] + [beta_s[(s, j)] for j in J] for s in S], columns=["Kịch bản"] + [J_name[j] for j in J])
        st.dataframe(beta_s_df, use_container_width=True)

        df_beta_plot = beta_s_df.melt(id_vars="Kịch bản", var_name="Hạng mục", value_name="Beta")
        fig_beta = px.line(df_beta_plot, x="Kịch bản", y="Beta", color="Hạng mục", markers=True, color_discrete_sequence=COLORS)
        fig_beta.update_layout(**PLOTLY_LAYOUT, title="Hệ số beta theo kịch bản")
        st.plotly_chart(fig_beta, use_container_width=True)

    sp_solution, det_solutions, ev_solution, robust_solution, metrics = _result_tables(prob_dict)

    with tabs[2]:
        section("Câu 10.5.1 — Giải mô hình Stochastic Programming", "📊")
        sp_x_df = pd.DataFrame({"Hạng mục": [J_name[j] for j in J], "x_j first-stage": [sp_solution["x"][j] for j in J], "Beta cơ bản": [beta[j] for j in J]})
        st.dataframe(sp_x_df.style.format({"x_j first-stage": "{:,.4f}", "Beta cơ bản": "{:.2f}"}), use_container_width=True)
        st.metric("Giá trị mục tiêu SP", f"{sp_solution['objective']:,.4f}")
        fig_x = px.bar(sp_x_df, x="Hạng mục", y="x_j first-stage", color="Hạng mục", color_discrete_sequence=COLORS)
        fig_x.update_layout(**PLOTLY_LAYOUT, title="Quyết định first-stage tối ưu")
        st.plotly_chart(fig_x, use_container_width=True)

        sp_y_df = pd.DataFrame([{"Kịch bản": S_name[s], "Hạng mục": J_name[j], "y_sj second-stage": sp_solution["y"][(s, j)]} for s in S for j in J])
        st.dataframe(sp_y_df.pivot(index="Kịch bản", columns="Hạng mục", values="y_sj second-stage").style.format("{:,.4f}"), use_container_width=True)
        fig_y = px.bar(sp_y_df, x="Kịch bản", y="y_sj second-stage", color="Hạng mục", barmode="stack", color_discrete_sequence=COLORS)
        fig_y.update_layout(**PLOTLY_LAYOUT, title="Quyết định second-stage theo kịch bản")
        st.plotly_chart(fig_y, use_container_width=True)

        det_df = pd.DataFrame(
            [
                {"Kịch bản": S_name[s], "Status": det_solutions[s]["status"], "Objective": det_solutions[s]["objective"], **{f"x_{j}": det_solutions[s]["x"][j] for j in J}}
                for s in S
            ]
        )
        section("Câu 10.5.2 — Giải bài toán xác định theo từng kịch bản", "🧭")
        st.dataframe(det_df.style.format({c: "{:,.4f}" for c in det_df.columns if c not in ["Kịch bản", "Status"]}), use_container_width=True)

    with tabs[2]:
        section("Câu 10.5.3 — Tính EEV, VSS và EVPI", "📈")
        metrics_df = pd.DataFrame({"Chỉ tiêu": ["RP/SP value", "EEV value", "VSS = RP - EEV", "WS value", "EVPI = WS - RP"], "Giá trị": [metrics["RP"], metrics["EEV"], metrics["VSS"], metrics["WS"], metrics["EVPI"]]})
        st.dataframe(metrics_df.style.format({"Giá trị": "{:,.4f}"}), use_container_width=True)
        fig_m = px.bar(metrics_df, x="Chỉ tiêu", y="Giá trị", color="Chỉ tiêu", color_discrete_sequence=COLORS)
        fig_m.update_layout(**PLOTLY_LAYOUT, title="So sánh RP, EEV, VSS, WS và EVPI")
        st.plotly_chart(fig_m, use_container_width=True)

        section("Câu 10.5.4 — Robust Optimization: Minimax Regret", "🛡️")
        robust_x_df = pd.DataFrame({
            "Hạng mục": [J_name[j] for j in J],
            "x_SP": [sp_solution["x"][j] for j in J],
            "x_EV": [ev_solution["x"][j] for j in J],
            "x_RobustRegret": [robust_solution["x"][j] for j in J],
        })
        st.dataframe(robust_x_df.style.format({c: "{:,.4f}" for c in robust_x_df.columns if c != "Hạng mục"}), use_container_width=True)
        fig_r = px.bar(robust_x_df, x="Hạng mục", y=["x_SP", "x_EV", "x_RobustRegret"], barmode="group", color_discrete_sequence=COLORS)
        fig_r.update_layout(**PLOTLY_LAYOUT, title="So sánh first-stage: SP, EV và robust regret")
        st.plotly_chart(fig_r, use_container_width=True)

        sp_H, ev_H = sp_solution["x"]["H"], ev_solution["x"]["H"]
        direction = "nhiều hơn" if sp_H > ev_H else "ít hơn hoặc tương đương"
        st.markdown(f"""
        <div class="result-note-card">
          <div class="result-note-title">💬 Nhận xét kết quả</div>
          <div class="result-note-body">
            SP đầu tư vào nhân lực số H {direction} EV, với SP={sp_H:,.4f} và EV={ev_H:,.4f}.
            VSS={metrics['VSS']:,.4f} đo lợi ích của việc xét bất định, còn EVPI={metrics['EVPI']:,.4f} đo giá trị tối đa của thông tin hoàn hảo.
            Nếu VSS dương, mô hình stochastic đem lại giá trị chính sách so với tối ưu theo kịch bản trung bình.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with tabs[3]:
        sp_H, ev_H = sp_solution["x"]["H"], ev_solution["x"]["H"]
        direction = "nhiều hơn" if sp_H > ev_H else "ít hơn hoặc tương đương"
        section("1. So với lời giải xác định EV, SP đầu tư H nhiều hơn hay ít hơn?", "📌")
        st.markdown(f"""
        Lời giải SP đầu tư vào H **{direction}** lời giải EV: SP = **{sp_H:,.4f}**, EV = **{ev_H:,.4f}**.
        Nếu SP phân bổ nhiều hơn cho H, nhân lực số đang đóng vai trò hàng hóa bảo hiểm trước kịch bản xấu.
        Đây là lý do mô hình ngẫu nhiên phù hợp với bối cảnh kinh tế mở và nhiều bất định.
        """)
        section("2. VSS và EVPI nói gì về giá trị thông tin?", "🔎")
        st.markdown(f"""
        **VSS = {metrics['VSS']:,.4f}** cho biết giá trị tăng thêm khi giải stochastic thay vì dùng nghiệm kỳ vọng.
        **EVPI = {metrics['EVPI']:,.4f}** là mức tối đa có thể trả cho hệ thống dự báo hoàn hảo.
        Nếu EVPI lớn, đầu tư vào dữ liệu thời gian thực, dự báo vĩ mô và cảnh báo sớm có cơ sở kinh tế rõ ràng.
        """)
        section("3. Robust regret phù hợp khi nào?", "🛡️")
        st.markdown("""
        Robust regret phù hợp khi nhà hoạch định sợ sai lầm lớn trong kịch bản xấu nhất hơn là tối đa hóa kỳ vọng.
        Cách tiếp cận này thường thận trọng hơn SP, nhưng có thể hy sinh một phần hiệu quả trung bình.
        Chính sách Việt Nam nên dùng SP cho kế hoạch cơ sở và robust regret như kiểm tra sức chịu đựng trước cú sốc.
        """)
        section("4. Hàm ý chính sách nâng cấp", "📋")
        st.markdown(f"""
**Kết quả nổi bật.** Giá trị RP đạt **{metrics['RP']:,.4f}**; VSS = **{metrics['VSS']:,.4f}**; EVPI = **{metrics['EVPI']:,.4f}**; phân bổ H trong SP là **{sp_H:,.4f}** so với EV **{ev_H:,.4f}**.

**Liên hệ chính sách Việt Nam.** Kết quả phù hợp với **Quyết định 749/QĐ-TTg** và **Quyết định 411/QĐ-TTg**: quyết định đầu tư số cần tính bất định vĩ mô, dữ liệu thời gian thực và giá trị thông tin.

**Đánh đổi cần lưu ý:** tối ưu kỳ vọng và khả năng chống chịu; nghiệm robust an toàn hơn nhưng có thể hy sinh hiệu quả trung bình.

**Khuyến nghị hành động.** Dùng nghiệm SP làm kế hoạch ngân sách cơ sở; dùng EVPI để xác định mức đầu tư tối đa cho dự báo và dữ liệu sớm; kiểm tra robust regret trước cú sốc xấu; nếu SP tăng H, giữ ngân sách nhân lực số như lớp bảo hiểm chính sách.
        """)

    with tabs[4]:
        sp_H, ev_H = sp_solution["x"]["H"], ev_solution["x"]["H"]
        ai_agent(
            f"""
            <b>1. Đọc kết quả hiện tại:</b><br>
            Giá trị SP đạt <b>{metrics['RP']:,.4f}</b>; VSS = <b>{metrics['VSS']:,.4f}</b>; EVPI = <b>{metrics['EVPI']:,.4f}</b>.<br><br>
            <b>2. Trade-off chính:</b><br>
            SP phân bổ H = <b>{sp_H:,.4f}</b>, EV phân bổ H = <b>{ev_H:,.4f}</b>. Nếu SP cao hơn EV,
            nhân lực số đang đóng vai trò năng lực bảo hiểm trước kịch bản xấu.<br><br>
            <b>3. Khuyến nghị:</b><br>
            Ưu tiên mô hình stochastic khi xác suất kịch bản có độ phân tán lớn; dùng EVPI để quyết định mức đầu tư hợp lý
            vào hệ thống dự báo, dữ liệu sớm và phân tích rủi ro vĩ mô.
            """
        )
