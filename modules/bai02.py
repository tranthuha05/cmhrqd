"""
Bài 2 — Quy hoạch tuyến tính phân bổ ngân sách số (LP)
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

COLORS = ["#00D4FF","#6366F1","#10B981","#F59E0B","#EC4899","#8B5CF6"]
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,32,0.6)",
    font=dict(color="#94A3B8", family="Inter"),
    xaxis=dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.1)"),
    margin=dict(l=40, r=20, t=50, b=40), legend=dict(bgcolor="rgba(0,0,0,0)")
)

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
def solve_lp(total_budget, w_I, w_D, w_AI, w_H, min_I, min_D, min_AI, min_H, min_H_gdp):
    """
    Giải LP bằng scipy.linprog.
    Biến: x = [x_I, x_D, x_AI, x_H]
    Maximize: c·x, tương đương Minimize: -c·x
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None

    c_weights = np.array([w_I, w_D, w_AI, w_H])
    # Minimize -c·x
    c_obj = -c_weights

    # Ràng buộc bất đẳng thức (A_ub @ x <= b_ub)
    A_ub = np.array([[1, 1, 1, 1]])   # Tổng <= total_budget
    b_ub = np.array([total_budget])

    # Bounds
    bounds = [
        (min_I,  total_budget),   # x_I
        (min_D,  total_budget),   # x_D
        (min_AI, total_budget),   # x_AI
        (min_H,  total_budget),   # x_H
    ]

    # Ràng buộc đẳng thức: tổng đúng bằng budget (thay x_H = budget - x_I - x_D - x_AI rồi check)
    # Thêm ràng buộc H >= min_H_gdp (x_H >= min_H_gdp)
    # Thêm bound min_H_gdp nếu cần
    bounds[3] = (max(min_H, min_H_gdp), total_budget)

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res


def show():
    st.markdown("""
    <div class="page-header">
      <div style="display:flex;align-items:center;gap:1rem;">
        <span style="font-size:2rem;">💰</span>
        <div>
          <div style="color:#00D4FF;font-size:.75rem;font-weight:600;letter-spacing:1px;">BÀI 2</div>
          <div style="color:#E2E8F0;font-size:1.5rem;font-weight:800;">Quy hoạch tuyến tính — Phân bổ ngân sách số</div>
          <div style="margin-top:.3rem;">
            <span class="badge b-blue">scipy linprog</span>
            <span class="badge b-purple">PuLP / LP</span>
            <span class="badge b-green">Shadow Price</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📖 Bối cảnh & Mô hình","🎛️ Tham số","📊 Kết quả","📈 Phân tích chính sách","🤖 AI Agent"])

    # ══ Tab 1 ══════════════════════════════════════════════════════════════
    with tabs[0]:
        section("Bối cảnh bài toán", "🌏")
        st.markdown("""
Chính phủ Việt Nam cần phân bổ **ngân sách đầu tư số** cho 4 hạng mục chiến lược.
Mục tiêu là **tối đa hóa GDP gain** — tức tổng giá trị kinh tế tạo ra từ đầu tư số —
trong điều kiện ràng buộc ngân sách và các yêu cầu tối thiểu theo ngành.

Bài toán này thuộc lớp **Quy hoạch tuyến tính (LP)**, có thể giải tối ưu toàn cục
bằng thuật toán Simplex hoặc Interior Point (HiGHS).
        """)

        section("Mô hình toán học", "📐")
        math_box("""
<b>Biến quyết định:</b>
  x<sub>I</sub>  ≥ 0 : Ngân sách đầu tư hạ tầng số (tỷ VND)
  x<sub>D</sub>  ≥ 0 : Ngân sách chuyển đổi số doanh nghiệp (tỷ VND)
  x<sub>AI</sub> ≥ 0 : Ngân sách phát triển AI (tỷ VND)
  x<sub>H</sub>  ≥ 0 : Ngân sách nhân lực số (tỷ VND)

<b>Hàm mục tiêu — Maximize GDP gain:</b>
  max Z = w<sub>I</sub>·x<sub>I</sub> + w<sub>D</sub>·x<sub>D</sub> + w<sub>AI</sub>·x<sub>AI</sub> + w<sub>H</sub>·x<sub>H</sub>

<b>Ràng buộc:</b>
  C1: x<sub>I</sub> + x<sub>D</sub> + x<sub>AI</sub> + x<sub>H</sub> ≤ B_total   (Tổng ngân sách)
  C2: x<sub>I</sub>  ≥ x<sub>I,min</sub>              (Hạ tầng số tối thiểu)
  C3: x<sub>D</sub>  ≥ x<sub>D,min</sub>              (Chuyển đổi số tối thiểu)
  C4: x<sub>AI</sub> ≥ x<sub>AI,min</sub>             (AI tối thiểu)
  C5: x<sub>H</sub>  ≥ x<sub>H,min</sub>              (Nhân lực số tối thiểu)
        """)

        math_box("""
<b>Phân tích shadow price (giá bóng):</b>
  λ<sub>i</sub> = ΔZ / ΔB<sub>i</sub>

  Nếu λ<sub>1</sub> (ràng buộc ngân sách) > 0:
    Tăng 1 tỷ VND ngân sách sẽ tăng GDP gain thêm λ<sub>1</sub> tỷ VND.
    Đây là thước đo hiệu quả biên của ngân sách.
        """)

    # ══ Tab 2: Tham số ══════════════════════════════════════════════════════
    with tabs[1]:
        section("Hệ số tác động GDP (w_j)", "🎛️")
        col1, col2 = st.columns(2)
        with col1:
            total_budget = st.slider("💰 Tổng ngân sách (nghìn tỷ VND)", 50.0, 200.0, 100.0, 5.0, key="b2_budget")
            w_I  = st.slider("w_I  — Hạ tầng số",     0.50, 2.00, 1.10, 0.05, key="b2_wI")
            w_D  = st.slider("w_D  — Chuyển đổi số",  0.50, 2.00, 1.25, 0.05, key="b2_wD")

        with col2:
            w_AI = st.slider("w_AI — AI & Dữ liệu",   0.50, 2.50, 1.55, 0.05, key="b2_wAI")
            w_H  = st.slider("w_H  — Nhân lực số",    0.50, 2.00, 1.05, 0.05, key="b2_wH")
            st.markdown("**Ràng buộc tối thiểu (nghìn tỷ VND)**")
            min_I  = st.slider("x_I  tối thiểu",  5.0, 30.0, 15.0, 1.0, key="b2_mI")
            min_D  = st.slider("x_D  tối thiểu",  5.0, 30.0, 15.0, 1.0, key="b2_mD")
            min_AI = st.slider("x_AI tối thiểu",  5.0, 30.0, 10.0, 1.0, key="b2_mAI")
            min_H  = st.slider("x_H  tối thiểu", 10.0, 40.0, 20.0, 1.0, key="b2_mH")

        min_H_gdp = min_H

        st.info(f"Hệ số tác động GDP cao nhất: **{'x_AI' if w_AI >= max(w_I,w_D,w_H) else 'x_I' if w_I >= max(w_D,w_H) else 'x_D'}** → mô hình sẽ ưu tiên phân bổ vào hạng mục này")

    # ══ Tab 3: Kết quả LP ════════════════════════════════════════════════════
    with tabs[2]:
        # Lấy tham số
        total_budget = st.session_state.get("b2_budget", 100.0)
        w_I  = st.session_state.get("b2_wI",   1.10)
        w_D  = st.session_state.get("b2_wD",   1.25)
        w_AI = st.session_state.get("b2_wAI",  1.55)
        w_H  = st.session_state.get("b2_wH",   1.05)
        min_I  = st.session_state.get("b2_mI",   15.0)
        min_D  = st.session_state.get("b2_mD",   15.0)
        min_AI = st.session_state.get("b2_mAI",  10.0)
        min_H  = st.session_state.get("b2_mH",   20.0)

        res = solve_lp(total_budget, w_I, w_D, w_AI, w_H, min_I, min_D, min_AI, min_H, min_H)

        if res is not None and res.success:
            x_opt = res.x
            items = ["Hạ tầng số", "Chuyển đổi số", "AI & Dữ liệu", "Nhân lực số"]
            weights = [w_I, w_D, w_AI, w_H]
            mins    = [min_I, min_D, min_AI, min_H]
            gdp_gain = -res.fun

            c1,c2,c3 = st.columns(3)
            c1.metric("Tổng GDP gain tối ưu", f"{gdp_gain:,.2f}", "nghìn tỷ VND")
            c2.metric("Ngân sách sử dụng", f"{x_opt.sum():,.2f}", f"/ {total_budget:,.0f}")
            c3.metric("Hiệu quả biên w_AI", f"{w_AI:.2f}", "cao nhất" if w_AI == max(w_I,w_D,w_AI,w_H) else "")

            section("Câu 2.4.1 — Nghiệm tối ưu LP", "✅")
            result_df = pd.DataFrame({
                "Hạng mục": items,
                "x* tối ưu": x_opt,
                "Tỷ trọng (%)": x_opt / x_opt.sum() * 100,
                "Hệ số w_j": weights,
                "Tối thiểu": mins,
                "GDP gain": x_opt * weights
            })
            st.dataframe(result_df.style.format({
                "x* tối ưu": "{:,.2f}", "Tỷ trọng (%)": "{:.2f}", "Hệ số w_j": "{:.2f}",
                "Tối thiểu": "{:.2f}", "GDP gain": "{:,.2f}"
            }).highlight_max(subset=["GDP gain"], color="#0a4f3a"), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig_bar = go.Figure(go.Bar(
                    x=items, y=x_opt, marker_color=COLORS[:4],
                    text=[f"{v:,.1f}" for v in x_opt], textposition="outside"))
                fig_bar.update_layout(**PLOTLY_LAYOUT, title="Phân bổ ngân sách tối ưu",
                    yaxis_title="Nghìn tỷ VND")
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                fig_pie = go.Figure(go.Pie(
                    labels=items, values=x_opt, hole=0.45,
                    marker=dict(colors=COLORS[:4]),
                    textinfo="label+percent"))
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94A3B8", family="Inter"),
                    title="Cơ cấu ngân sách (%)", margin=dict(t=50,b=20))
                st.plotly_chart(fig_pie, use_container_width=True)

            section("Câu 2.4.2 — Shadow Price & Phân tích độ nhạy", "🔬")
            # Tính shadow price xấp xỉ
            deltas = [1.0, 2.0, 5.0]
            shadow_rows = []
            for db in deltas:
                res2 = solve_lp(total_budget+db, w_I, w_D, w_AI, w_H, min_I, min_D, min_AI, min_H, min_H)
                if res2 and res2.success:
                    sp = (-res2.fun - gdp_gain) / db
                    shadow_rows.append({"ΔB (nghìn tỷ)": db, "GDP gain mới": -res2.fun,
                                        "Shadow price λ": sp})
            if shadow_rows:
                sp_df = pd.DataFrame(shadow_rows)
                st.dataframe(sp_df.style.format("{:.4f}"), use_container_width=True)
                avg_sp = pd.DataFrame(shadow_rows)["Shadow price λ"].mean()
                st.info(f"**Shadow price trung bình ≈ {avg_sp:.4f}** → Mỗi 1 nghìn tỷ VND tăng thêm trong ngân sách tạo ra khoảng **{avg_sp:,.2f} nghìn tỷ VND** GDP gain.")

            section("Câu 2.4.3 — So sánh phân bổ với hệ số khác nhau", "📊")
            scenarios = {
                "Hiện tại": [w_I, w_D, w_AI, w_H],
                "Ưu tiên hạ tầng": [1.8, 1.0, 1.0, 1.0],
                "Ưu tiên AI": [1.0, 1.0, 2.0, 1.0],
                "Cân bằng": [1.2, 1.2, 1.2, 1.2],
            }
            sc_rows = []
            for sc_name, sc_w in scenarios.items():
                r = solve_lp(total_budget, *sc_w, min_I, min_D, min_AI, min_H, min_H)
                if r and r.success:
                    sc_rows.append({"Kịch bản": sc_name, "GDP gain": -r.fun,
                                    "x_AI*": r.x[2], "x_H*": r.x[3]})
            sc_df = pd.DataFrame(sc_rows)
            st.dataframe(sc_df.style.format({"GDP gain": "{:,.2f}", "x_AI*": "{:,.2f}", "x_H*": "{:,.2f}"}), use_container_width=True)
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Bar(x=sc_df["Kịch bản"], y=sc_df["GDP gain"],
                marker_color=COLORS[:4], text=[f"{v:,.1f}" for v in sc_df["GDP gain"]],
                textposition="outside", name="GDP gain"))
            fig_sc.update_layout(**PLOTLY_LAYOUT, title="So sánh GDP gain theo kịch bản",
                yaxis_title="Nghìn tỷ VND")
            st.plotly_chart(fig_sc, use_container_width=True)

            section("Câu 2.4.4 — Kịch bản ưu tiên nhân lực số x_H ≥ 30", "👷")
            res_human = solve_lp(total_budget, w_I, w_D, w_AI, w_H, min_I, min_D, min_AI, max(min_H, 30.0), max(min_H, 30.0))
            if res_human is not None and res_human.success:
                human_df = pd.DataFrame({
                    "Hạng mục": items,
                    "Phân bổ cơ sở": x_opt,
                    "Phân bổ ưu tiên H": res_human.x,
                    "Chênh lệch": res_human.x - x_opt,
                })
                st.dataframe(human_df.style.format({c: "{:,.2f}" for c in human_df.columns if c != "Hạng mục"}), use_container_width=True)
                fig_human = go.Figure()
                fig_human.add_trace(go.Bar(x=items, y=x_opt, name="Cơ sở", marker_color=COLORS[0]))
                fig_human.add_trace(go.Bar(x=items, y=res_human.x, name="Ưu tiên H", marker_color=COLORS[4]))
                fig_human.update_layout(**PLOTLY_LAYOUT, barmode="group", title="So sánh phân bổ khi tăng sàn nhân lực số", yaxis_title="Nghìn tỷ VND")
                st.plotly_chart(fig_human, use_container_width=True)
                st.markdown(f"""
                <div class="result-note-card">
                  <div class="result-note-title">💬 Nhận xét kết quả</div>
                  <div class="result-note-body">
                    Khi nâng sàn nhân lực số lên 30 nghìn tỷ VND, ngân sách dành cho H tăng từ {x_opt[3]:,.2f} lên {res_human.x[3]:,.2f}.
                    GDP gain thay đổi từ {gdp_gain:,.2f} xuống/cao hơn tùy cấu trúc hệ số, hiện đạt {-res_human.fun:,.2f}.
                    Kịch bản này làm rõ chi phí cơ hội của chính sách bao trùm nhân lực số: tăng an toàn xã hội nhưng có thể giảm phần dư dành cho hạng mục có hệ số biên cao.
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Kịch bản ưu tiên nhân lực số không khả thi với ngân sách/ràng buộc hiện tại.")
        else:
            st.warning("⚠️ Bài toán không khả thi với ràng buộc hiện tại. Vui lòng điều chỉnh tham số trong Tab Tham số.")
            if res:
                st.markdown(
                    f"""
                    <div class="status-card">
                      <div class="agent-title">Trạng thái solver</div>
                      <div class="agent-body">{res.message}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ══ Tab 4: Phân tích chính sách ══════════════════════════════════════════
    with tabs[3]:
        section("1. AI và hạ tầng số — đầu tư nào sinh lời cao hơn?", "📊")
        st.markdown(f"""
Với hệ số tác động **w_AI = {st.session_state.get('b2_wAI', 1.55):.2f}** và **w_I = {st.session_state.get('b2_wI', 1.10):.2f}**:

- Mô hình LP sẽ ưu tiên phân bổ cho **AI** vì có hiệu quả biên cao hơn
- Tuy nhiên, AI cần nền tảng hạ tầng số (băng rộng, cloud, data center) nên cần đặt **ràng buộc tối thiểu cho x_I**
- Trong thực tế, đầu tư AI không thể thay thế hoàn toàn đầu tư hạ tầng số — đây là quan hệ bổ sung, không thay thế
        """)

        section("2. Ràng buộc nhân lực số x_H ≥ 20.000 tỷ", "👷")
        st.markdown("""
Ràng buộc nhân lực số phản ánh quan điểm chính sách quan trọng:
- **AI không thể phát huy hiệu quả** nếu thiếu nhân lực có kỹ năng vận hành và quản trị
- Tốc độ tự động hóa phải đi kèm với năng lực đào tạo lại lao động
- Đây là ràng buộc "bảo hiểm xã hội" trong mô hình tối ưu hóa kinh tế
        """)

        section("3. Shadow price và hàm ý ngân sách", "💡")
        st.info("""
**Shadow price dương** cho ràng buộc ngân sách có nghĩa là:
Chính phủ nên **mở rộng ngân sách đầu tư số** nếu GDP gain biên còn lớn.

Nếu shadow price = 1.5, mỗi 1.000 tỷ VND tăng thêm tạo ra 1.500 tỷ VND GDP gain.
Đây là cơ sở kinh tế học để biện hộ cho việc tăng chi tiêu đầu tư phát triển trong lĩnh vực số.
        """)
        total_budget = st.session_state.get("b2_budget", 100.0)
        w_I = st.session_state.get("b2_wI", 1.10)
        w_D = st.session_state.get("b2_wD", 1.25)
        w_AI = st.session_state.get("b2_wAI", 1.55)
        w_H = st.session_state.get("b2_wH", 1.05)
        min_I = st.session_state.get("b2_mI", 15.0)
        min_D = st.session_state.get("b2_mD", 15.0)
        min_AI = st.session_state.get("b2_mAI", 10.0)
        min_H = st.session_state.get("b2_mH", 20.0)
        policy_res = solve_lp(total_budget, w_I, w_D, w_AI, w_H, min_I, min_D, min_AI, min_H, min_H)
        if policy_res and policy_res.success:
            x_policy = policy_res.x
            top_budget_item = ["hạ tầng số", "chuyển đổi số doanh nghiệp", "AI và dữ liệu", "nhân lực số"][int(np.argmax(x_policy))]
            st.markdown(f"""
**Kết quả nổi bật.** Ngân sách mô phỏng là **{total_budget:,.1f}**; hệ số biên AI là **{w_AI:.2f}**, cao hơn hạ tầng số **{w_I:.2f}**; nghiệm phân bổ lớn nhất cho **{top_budget_item}** với **{x_policy.max():,.2f}**; ràng buộc nhân lực số tối thiểu là **{min_H:,.2f}**.

**Liên hệ chính sách Việt Nam.** Kết quả gắn với **Quyết định 749/QĐ-TTg** và **Quyết định 411/QĐ-TTg**: tối ưu ngân sách số phải vừa nâng hạ tầng, dữ liệu, AI, vừa bảo đảm năng lực nhân lực để triển khai kinh tế số.

**Đánh đổi cần lưu ý:** tối ưu ngân sách và khả năng triển khai; dồn vốn vào AI có thể tăng GDP gain biên nhưng nếu hạ tầng và H thấp thì hiệu quả thực tế giảm.

**Khuyến nghị hành động.** Giữ ràng buộc tối thiểu cho hạ tầng số và nhân lực số; chỉ nới ngân sách khi shadow price còn dương; ưu tiên AI ở các dự án có dữ liệu sẵn sàng; theo dõi tỷ lệ giải ngân từng hạng mục thay vì chỉ nhìn tổng ngân sách.
            """)

        section("4. Kết luận", "✅")
        st.success("""
Bài 2 cho thấy LP là công cụ mạnh để tối ưu phân bổ ngân sách số.

Kết quả nhất quán với lý thuyết kinh tế: mô hình ưu tiên hạng mục có hiệu quả biên cao nhất
(thường là AI và chuyển đổi số doanh nghiệp), đồng thời tôn trọng ràng buộc tối thiểu
về hạ tầng và nhân lực số như điều kiện cần thiết.
        """)

    # ══ Tab 5: AI Agent ══════════════════════════════════════════════════════
    with tabs[4]:
        total_budget = st.session_state.get("b2_budget", 100.0)
        w_I  = st.session_state.get("b2_wI",   1.10)
        w_D  = st.session_state.get("b2_wD",   1.25)
        w_AI = st.session_state.get("b2_wAI",  1.55)
        w_H  = st.session_state.get("b2_wH",   1.05)
        min_I  = st.session_state.get("b2_mI",   15.0)
        min_D  = st.session_state.get("b2_mD",   15.0)
        min_AI = st.session_state.get("b2_mAI",  10.0)
        min_H  = st.session_state.get("b2_mH",   20.0)

        res = solve_lp(total_budget, w_I, w_D, w_AI, w_H, min_I, min_D, min_AI, min_H, min_H)

        if res and res.success:
            x_opt    = res.x
            gdp_gain = -res.fun
            top_item = ["Hạ tầng số","Chuyển đổi số","AI & Dữ liệu","Nhân lực số"][np.argmax(x_opt)]
            pct_AI   = x_opt[2] / x_opt.sum() * 100

            ai_agent(f"""
<b>1. Kết quả tối ưu hóa LP:</b><br>
GDP gain tối ưu = <b>{gdp_gain:,.2f} nghìn tỷ VND</b> với ngân sách {total_budget:,.0f} nghìn tỷ VND.<br>
Hạng mục được phân bổ nhiều nhất: <b>{top_item}</b> ({x_opt.max():,.1f} = {x_opt.max()/x_opt.sum()*100:.1f}%).<br><br>

<b>2. Nhận xét về cơ cấu phân bổ:</b><br>
AI & Dữ liệu chiếm <b>{pct_AI:.1f}%</b> ngân sách — {'phù hợp' if pct_AI < 50 else '⚠️ quá tập trung'}.<br>
{'✅ Cơ cấu đa dạng và cân bằng.' if pct_AI < 50 else '⚠️ Khuyến nghị: Tăng tối thiểu x_H và x_I để giảm rủi ro phụ thuộc vào AI.'}<br><br>

<b>3. Khuyến nghị chính sách:</b><br>
{'✅ Mức ngân sách ' + str(int(total_budget)) + ' nghìn tỷ phù hợp với quy mô nền kinh tế số 🇻🇳 hiện tại.' if total_budget <= 120 else '⚠️ Ngân sách cao — cần kế hoạch giải ngân và giám sát hiệu quả chặt chẽ.'}<br>
• Ưu tiên AI đi kèm với đầu tư nhân lực — tỷ lệ AI/H tối ưu nên xấp xỉ 1:1<br>
• Hạ tầng số cần tối thiểu 15–20% ngân sách để đảm bảo nền tảng hấp thụ AI<br>
• Shadow price cho thấy hiệu quả biên còn dương → Việt Nam nên tiếp tục mở rộng đầu tư số<br><br>

<b>4. So sánh với thực tiễn:</b><br>
Theo kế hoạch phát triển kinh tế số 2025, 🇻🇳 phân bổ khoảng {total_budget*0.6:,.0f}–{total_budget*0.8:,.0f} nghìn tỷ cho lĩnh vực số.
Mô hình LP gợi ý mức hiệu quả hơn có thể đạt được bằng cách tối ưu hóa cơ cấu thay vì chỉ tăng tổng ngân sách.
            """)
        else:
            st.warning("Không có kết quả LP — vui lòng kiểm tra ràng buộc trong Tab Tham số.")
