"""
Bài 12 — AIDEOM-VN Policy Lab đồ án tích hợp.

Streamlit dashboard phục hồi đầy đủ các mô-đun M1-M6 từ file gốc bài_12.py,
giữ nguyên logic tính toán và chỉ nâng cấp cách trình bày.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


COLORS = ["#F472B6", "#67E8F9", "#A78BFA", "#FDE68A", "#86EFAC", "#FDBA74", "#60A5FA"]
CAPITAL_LABELS = {
    "K": "K - Vốn vật chất",
    "D": "D - Hạ tầng số",
    "AI": "AI - Năng lực AI",
    "H": "H - Nhân lực số",
}


SCENARIOS = {
    "S1": {
        "name": "S1 - Truyền thống",
        "short": "S1 Truyền thống",
        "desc": "Tập trung vốn vật chất, FDI, hạ tầng truyền thống, xuất khẩu.",
        "alloc": np.array([0.70, 0.10, 0.10, 0.10]),
    },
    "S2": {
        "name": "S2 - Số hóa nhanh",
        "short": "S2 Số hóa nhanh",
        "desc": "Tăng đầu tư chính phủ số, doanh nghiệp số, thanh toán số.",
        "alloc": np.array([0.25, 0.45, 0.15, 0.15]),
    },
    "S3": {
        "name": "S3 - AI dẫn dắt",
        "short": "S3 AI dẫn dắt",
        "desc": "Ưu tiên AI, dữ liệu lớn, bán dẫn, trung tâm dữ liệu.",
        "alloc": np.array([0.20, 0.20, 0.45, 0.15]),
    },
    "S4": {
        "name": "S4 - Bao trùm số",
        "short": "S4 Bao trùm số",
        "desc": "Ưu tiên vùng yếu, SME, giáo dục số, nông nghiệp số.",
        "alloc": np.array([0.30, 0.20, 0.10, 0.40]),
    },
    "S5": {
        "name": "S5 - Tối ưu cân bằng",
        "short": "S5 Tối ưu cân bằng",
        "desc": "Kịch bản cân bằng từ logic AIDEOM-VN.",
        "alloc": np.array([0.40, 0.25, 0.15, 0.20]),
    },
}


REGIONS_DF = pd.DataFrame(
    {
        "Vùng": [
            "Trung du miền núi phía Bắc",
            "Đồng bằng sông Hồng",
            "Bắc Trung Bộ + DH Trung Bộ",
            "Tây Nguyên",
            "Đông Nam Bộ",
            "Đồng bằng sông Cửu Long",
        ],
        "Digital_Index": [38, 78, 55, 32, 82, 48],
        "AI_Readiness": [22, 68, 40, 18, 75, 30],
        "Internet": [72, 92, 84, 68, 94, 78],
        "Labor_Trained": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
        "Gini": [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
    }
)


SECTOR_DF = pd.DataFrame(
    {
        "Ngành": [
            "Nông-Lâm-Thủy sản",
            "CN chế biến chế tạo",
            "Xây dựng",
            "Bán buôn-bán lẻ",
            "Tài chính-Ngân hàng",
            "Logistics-Vận tải",
            "CNTT-Truyền thông",
            "Giáo dục-Đào tạo",
            "Y tế",
        ],
        "Growth": [3.27, 9.64, 7.45, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
        "Productivity": [103.4, 241.2, 168.8, 145.3, 1072.4, 321.4, 713.8, 205.7, 437.1],
        "Spillover": [0.35, 0.78, 0.42, 0.55, 0.85, 0.72, 0.92, 0.65, 0.60],
        "Export": [40.5, 290.9, 2.5, 5.5, 1.2, 3.1, 178.0, 0.0, 0.0],
        "Labor": [13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15, 0.75],
        "AI_Readiness": [15, 55, 20, 48, 72, 42, 88, 38, 45],
        "Automation_Risk": [18, 42, 25, 38, 52, 35, 28, 22, 18],
    }
)


def section(title: str, icon: str = "📌") -> None:
    st.markdown(
        f"""
        <div class="section-chip" style="margin-top:1rem;margin-bottom:.7rem;">
          {icon} {title}
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_note(title: str, *sentences: str) -> None:
    body = " ".join(sentence.strip() for sentence in sentences if sentence and sentence.strip())
    st.markdown(
        f"""
        <div class="result-note-card">
          <div class="result-note-title">💬 {title}</div>
          <div class="result-note-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def policy_card(title: str, body: str, icon: str = "🍰") -> None:
    st.markdown(
        f"""
        <div class="ai-agent-card">
          <div class="agent-title">{icon} {title}</div>
          <div class="agent-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def norm_good(x: pd.Series) -> pd.Series:
    spread = x.max() - x.min()
    if spread == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.min()) / spread


def norm_bad(x: pd.Series) -> pd.Series:
    spread = x.max() - x.min()
    if spread == 0:
        return pd.Series(np.ones(len(x)), index=x.index)
    return (x.max() - x) / spread


def simulate_macro(allocation, years=np.arange(2026, 2031)):
    K = 27500.0
    D = 20.3
    AI = 86.0
    H = 30.0
    A = 35.0
    L = 54.0

    alpha_K, alpha_L, alpha_D, alpha_AI, alpha_H = 0.33, 0.42, 0.10, 0.08, 0.07
    rows = []

    for _, year in enumerate(years):
        budget = 1000
        IK, ID, IAI, IH = allocation * budget

        K = 0.95 * K + IK
        D = 0.88 * D + ID / 100
        AI = 0.85 * AI + IAI / 20
        H = 0.98 * H + IH / 200

        A = A * (1 + 0.0008 * D + 0.0005 * AI + 0.0010 * H)
        Y = A * (K**alpha_K) * (L**alpha_L) * (D**alpha_D) * (AI**alpha_AI) * (H**alpha_H)

        rows.append(
            {
                "Year": year,
                "K": K,
                "D": D,
                "AI": AI,
                "H": H,
                "TFP_A": A,
                "GDP_forecast": Y,
            }
        )

    return pd.DataFrame(rows)


def allocation_table() -> pd.DataFrame:
    rows = []
    for code, item in SCENARIOS.items():
        rows.append(
            {
                "Mã": code,
                "Kịch bản": item["name"],
                "Mô tả": item["desc"],
                "K": item["alloc"][0],
                "D": item["alloc"][1],
                "AI": item["alloc"][2],
                "H": item["alloc"][3],
            }
        )
    return pd.DataFrame(rows)


def macro_outputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    rows = []
    for code, item in SCENARIOS.items():
        df_s = simulate_macro(item["alloc"])
        results[code] = df_s
        last = df_s.iloc[-1]
        rows.append(
            {
                "Mã": code,
                "Kịch bản": item["name"],
                "GDP 2030": last["GDP_forecast"],
                "K 2030": last["K"],
                "D 2030": last["D"],
                "AI 2030": last["AI"],
                "H 2030": last["H"],
                "TFP 2030": last["TFP_A"],
            }
        )
    return results, pd.DataFrame(rows)


def run_m2_ranking() -> pd.DataFrame:
    df = REGIONS_DF.copy()
    df["Digital_norm"] = norm_good(df["Digital_Index"])
    df["AI_norm"] = norm_good(df["AI_Readiness"])
    df["Internet_norm"] = norm_good(df["Internet"])
    df["Labor_norm"] = norm_good(df["Labor_Trained"])
    df["Gini_norm"] = norm_bad(df["Gini"])
    df["Regional_Readiness"] = (
        0.30 * df["Digital_norm"]
        + 0.30 * df["AI_norm"]
        + 0.15 * df["Internet_norm"]
        + 0.15 * df["Labor_norm"]
        + 0.10 * df["Gini_norm"]
    )
    out = df.sort_values("Regional_Readiness", ascending=False).reset_index(drop=True)
    out["Rank"] = out.index + 1
    return out


def budget_outputs(total_budget: float = 50000) -> pd.DataFrame:
    rows = []
    for code, item in SCENARIOS.items():
        alloc = item["alloc"]
        rows.append(
            {
                "Mã": code,
                "Kịch bản": item["name"],
                "K": alloc[0] * total_budget,
                "D": alloc[1] * total_budget,
                "AI": alloc[2] * total_budget,
                "H": alloc[3] * total_budget,
                "Tổng": total_budget,
            }
        )
    return pd.DataFrame(rows)


def assign_policy_group(row: pd.Series) -> str:
    if row["AI_Readiness"] >= 70 and row["Spillover"] >= 0.80:
        return "A - Dẫn dắt AI"
    if row["Automation_Risk"] >= 40 and row["Labor"] >= 5:
        return "B - Rủi ro tự động hóa cao"
    if row["Labor"] >= 7:
        return "C - Bao trùm lao động lớn"
    if row["Spillover"] >= 0.70:
        return "D - Lan tỏa chuỗi giá trị"
    return "E - Chuyển đổi có chọn lọc"


def policy_recommendation(row: pd.Series) -> str:
    group = row["Policy_Group"]
    if group == "A - Dẫn dắt AI":
        return "Ưu tiên AI, dữ liệu lớn, R&D, sandbox và tiêu chuẩn an toàn dữ liệu."
    if group == "B - Rủi ro tự động hóa cao":
        return "Gắn AI với đào tạo lại, chuyển đổi nghề, bảo hiểm thất nghiệp chủ động."
    if group == "C - Bao trùm lao động lớn":
        return "Ưu tiên công nghệ hỗ trợ lao động, thương mại điện tử, kỹ năng số cơ bản."
    if group == "D - Lan tỏa chuỗi giá trị":
        return "Đầu tư dữ liệu chuỗi cung ứng, logistics số, nền tảng kết nối doanh nghiệp."
    return "Triển khai số hóa có chọn lọc, tránh đầu tư dàn trải, ưu tiên dự án chi phí thấp."


def sector_outputs() -> pd.DataFrame:
    df = SECTOR_DF.copy()
    df["Growth_norm"] = norm_good(df["Growth"])
    df["Productivity_norm"] = norm_good(df["Productivity"])
    df["Spillover_norm"] = norm_good(df["Spillover"])
    df["Export_norm"] = norm_good(df["Export"])
    df["Labor_norm"] = norm_good(df["Labor"])
    df["AI_norm"] = norm_good(df["AI_Readiness"])
    df["Risk_norm"] = norm_good(df["Automation_Risk"])
    df["Priority_Score"] = (
        0.15 * df["Growth_norm"]
        + 0.15 * df["Productivity_norm"]
        + 0.20 * df["Spillover_norm"]
        + 0.15 * df["Export_norm"]
        + 0.10 * df["Labor_norm"]
        + 0.20 * df["AI_norm"]
        - 0.15 * df["Risk_norm"]
    )
    df["Labor_Vulnerability"] = 0.60 * df["Labor_norm"] + 0.40 * df["Risk_norm"]
    df["Policy_Group"] = df.apply(assign_policy_group, axis=1)
    df["Policy_Recommendation"] = df.apply(policy_recommendation, axis=1)
    out = df.sort_values("Priority_Score", ascending=False).reset_index(drop=True)
    out["Rank"] = out.index + 1
    return out


def simulate_labor(allocation: np.ndarray, sector_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if sector_df is None:
        sector_df = sector_outputs()
    base = SECTOR_DF.copy()
    risk = base["Automation_Risk"].values / 100
    a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5, 20.0])
    b1 = np.array([45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0, 40.0])
    c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5, 15.0])

    labor_budget = 30000
    x_ai_total = allocation[2] * labor_budget
    x_h_total = allocation[3] * labor_budget

    x_ai = x_ai_total * a1 / a1.sum()
    x_h = x_h_total * b1 / b1.sum()
    newjob = a1 * x_ai
    upgrade = b1 * x_h
    displaced = c1 * risk * x_ai
    netjob = newjob + upgrade - displaced

    return pd.DataFrame(
        {
            "Ngành": base["Ngành"],
            "x_AI": x_ai,
            "x_H": x_h,
            "NewJob": newjob,
            "UpgradeJob": upgrade,
            "DisplacedJob": displaced,
            "NetJob": netjob,
        }
    )


def assess_risk(allocation: np.ndarray) -> tuple[float, float, float, float]:
    K, D, AI, H = allocation
    cyber = max(0, 100 * (0.55 * AI + 0.25 * D - 0.30 * H))
    emission = max(0, 100 * (0.50 * K + 0.40 * AI - 0.20 * D))
    dependency = max(0, 100 * (0.45 * AI + 0.30 * D - 0.20 * H))
    total = 0.40 * cyber + 0.35 * emission + 0.25 * dependency
    return cyber, emission, dependency, total


def labor_risk_outputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    detail: dict[str, pd.DataFrame] = {}
    for code, item in SCENARIOS.items():
        labor_s = simulate_labor(item["alloc"])
        detail[code] = labor_s
        cyber, emission, dependency, total_risk = assess_risk(item["alloc"])
        rows.append(
            {
                "Mã": code,
                "Kịch bản": item["name"],
                "Total NewJob": labor_s["NewJob"].sum(),
                "Total UpgradeJob": labor_s["UpgradeJob"].sum(),
                "Total DisplacedJob": labor_s["DisplacedJob"].sum(),
                "Total NetJob": labor_s["NetJob"].sum(),
                "Cyber Risk": cyber,
                "Emission Risk": emission,
                "Dependency Risk": dependency,
                "Total Risk": total_risk,
            }
        )
    return pd.DataFrame(rows), detail


def aideom_score_outputs(macro_summary: pd.DataFrame, labor_risk_df: pd.DataFrame) -> pd.DataFrame:
    summary = macro_summary[["Mã", "Kịch bản", "GDP 2030", "D 2030", "AI 2030", "H 2030"]].merge(
        labor_risk_df[["Mã", "Total NetJob", "Total Risk"]],
        on="Mã",
    )
    if "Kịch bản_x" in summary.columns:
        summary["Kịch bản"] = summary["Kịch bản_x"]
        summary = summary.drop(columns=["Kịch bản_x", "Kịch bản_y"], errors="ignore")
    summary["GDP_norm"] = norm_good(summary["GDP 2030"])
    summary["NetJob_norm"] = norm_good(summary["Total NetJob"])
    summary["Risk_norm"] = norm_bad(summary["Total Risk"])
    summary["AIDEOM_Score"] = 0.45 * summary["GDP_norm"] + 0.30 * summary["NetJob_norm"] + 0.25 * summary["Risk_norm"]
    out = summary.sort_values("AIDEOM_Score", ascending=False).reset_index(drop=True)
    out["Rank"] = out.index + 1
    return out


def warning_rows(summary_rank: pd.DataFrame) -> list[str]:
    warnings = []
    risk_median = summary_rank["Total Risk"].median()
    job_median = summary_rank["Total NetJob"].median()
    gdp_median = summary_rank["GDP 2030"].median()
    for _, row in summary_rank.iterrows():
        if row["Total Risk"] > risk_median:
            warnings.append(f"{row['Kịch bản']}: rủi ro tổng hợp cao, cần tăng H hoặc giảm tốc độ AI/K.")
        if row["Total NetJob"] < job_median:
            warnings.append(f"{row['Kịch bản']}: NetJob thấp, cần bổ sung chính sách đào tạo lại lao động.")
        if row["GDP 2030"] < gdp_median:
            warnings.append(f"{row['Kịch bản']}: GDP 2030 thấp, cần tăng đầu tư năng suất.")
    return warnings


def all_outputs():
    macro_results, macro_summary = macro_outputs()
    regions_rank = run_m2_ranking()
    budget_df = budget_outputs()
    sector_rank = sector_outputs()
    labor_risk_df, labor_detail = labor_risk_outputs()
    score_rank = aideom_score_outputs(macro_summary, labor_risk_df)
    return macro_results, macro_summary, regions_rank, budget_df, sector_rank, labor_risk_df, labor_detail, score_rank


def show() -> None:
    st.session_state["suppress_auto_result_notes"] = True
    macro_results, macro_summary, regions_rank, budget_df, sector_rank, labor_risk_df, labor_detail, score_rank = all_outputs()

    best = score_rank.iloc[0]
    highest_gdp = score_rank.sort_values("GDP 2030", ascending=False).iloc[0]
    highest_jobs = score_rank.sort_values("Total NetJob", ascending=False).iloc[0]
    lowest_risk = score_rank.sort_values("Total Risk", ascending=True).iloc[0]
    top_sector = sector_rank.iloc[0]
    high_risk_sector = sector_rank.sort_values("Automation_Risk", ascending=False).iloc[0]

    st.markdown(
        """
        <div class="page-header">
          <div style="display:flex;align-items:center;gap:1rem;">
            <span style="font-size:2rem;">🇻🇳</span>
            <div>
              <div style="color:#B83280;font-size:.78rem;font-weight:900;letter-spacing:1px;">BÀI 12 — DASHBOARD TỔNG HỢP CUỐI KỲ</div>
              <div style="color:#2B1230;font-size:1.65rem;font-weight:900;">🇻🇳 AIDEOM-VN Policy Lab — Nguyên mẫu đồ án tích hợp</div>
              <div style="margin-top:.35rem;">
                <span class="badge b-blue">Integrated Policy Engine</span>
                <span class="badge b-purple">AIDEOM Score</span>
                <span class="badge b-green">AI Policy Agent</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best Scenario", best["Mã"], f"{best['AIDEOM_Score']:.3f} score")
    k2.metric("Highest GDP", highest_gdp["Mã"], f"{highest_gdp['GDP 2030']:,.0f}")
    k3.metric("Highest NetJob", highest_jobs["Mã"], f"{highest_jobs['Total NetJob']:,.0f}")
    k4.metric("Lowest Risk", lowest_risk["Mã"], f"{lowest_risk['Total Risk']:.1f}")

    tabs = st.tabs(
        [
            "🌸 Tổng quan & S1-S5",
            "📈 M1 Vĩ mô",
            "🌍 M2 Vùng",
            "💰 M3 Ngân sách",
            "🏭 M4 Ngành",
            "⚠️ M5 NetJob & Rủi ro",
            "🏆 M6 AIDEOM Score",
            "🤖 Agent & Kết luận",
        ]
    )

    with tabs[0]:
        section("Bối cảnh AIDEOM-VN", "🌸")
        st.markdown(
            """
            **AIDEOM-VN** là nguyên mẫu hệ thống hỗ trợ ra quyết định chính sách kinh tế số cho Việt Nam.
            Mục tiêu của hệ thống không phải thay thế nhà hoạch định chính sách, mà giúp lượng hóa minh bạch
            các đánh đổi giữa tăng trưởng, việc làm, rủi ro, chuyển đổi số, AI và nhân lực số.

            Hệ thống tích hợp 6 module: **M1** dự báo kinh tế vĩ mô, **M2** đánh giá sẵn sàng số theo vùng,
            **M3** mô phỏng phân bổ ngân sách, **M4** chính sách theo 9 ngành, **M5** lao động và rủi ro,
            **M6** AIDEOM Score và bảng hỗ trợ ra quyết định.
            """
        )

        section("5 kịch bản chính sách S1-S5", "🎛️")
        alloc_df = allocation_table()
        st.dataframe(
            alloc_df.style.format({"K": "{:.0%}", "D": "{:.0%}", "AI": "{:.0%}", "H": "{:.0%}"}),
            use_container_width=True,
        )

        selected_code = st.selectbox(
            "Chọn kịch bản để đọc nhanh cơ cấu phân bổ",
            list(SCENARIOS.keys()),
            format_func=lambda code: SCENARIOS[code]["name"],
            key="b12_selected_scenario_full",
        )
        selected = SCENARIOS[selected_code]
        selected_alloc = pd.DataFrame(
            {"Cấu phần": list(CAPITAL_LABELS.values()), "Tỷ trọng": selected["alloc"] * 100}
        )
        c1, c2 = st.columns([1.2, 1])
        with c1:
            plot_df = alloc_df.melt(
                id_vars=["Mã", "Kịch bản", "Mô tả"],
                value_vars=["K", "D", "AI", "H"],
                var_name="Cấu phần",
                value_name="Tỷ trọng",
            )
            plot_df["Tỷ trọng (%)"] = plot_df["Tỷ trọng"] * 100
            fig_alloc = px.bar(
                plot_df,
                x="Kịch bản",
                y="Tỷ trọng (%)",
                color="Cấu phần",
                title="Cơ cấu phân bổ ngân sách theo 5 kịch bản",
                color_discrete_sequence=COLORS,
            )
            fig_alloc.update_layout(barmode="stack", yaxis_title="Tỷ trọng ngân sách (%)", xaxis_title="")
            st.plotly_chart(fig_alloc, use_container_width=True)
        with c2:
            fig_donut = px.pie(
                selected_alloc,
                names="Cấu phần",
                values="Tỷ trọng",
                hole=0.50,
                title=f"Donut allocation — {selected['short']}",
                color_discrete_sequence=COLORS,
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        result_note(
            "Nhận xét kịch bản hiện tại",
            f"Kịch bản {selected['name']} phân bổ cao nhất cho {selected_alloc.loc[selected_alloc['Tỷ trọng'].idxmax(), 'Cấu phần']} với {selected_alloc['Tỷ trọng'].max():.0f}%.",
            selected["desc"],
            "Đây là điểm xuất phát để đọc các trade-off phía sau: kịch bản nghiêng về AI/K thường kéo GDP hoặc công nghệ lên, còn kịch bản nghiêng về H giúp giảm rủi ro và cải thiện bao trùm lao động.",
        )

    with tabs[1]:
        section("M1 — Dự báo kinh tế vĩ mô 2026-2030", "📈")
        st.dataframe(
            macro_summary.style.format(
                {
                    "GDP 2030": "{:,.2f}",
                    "K 2030": "{:,.2f}",
                    "D 2030": "{:,.2f}",
                    "AI 2030": "{:,.2f}",
                    "H 2030": "{:,.2f}",
                    "TFP 2030": "{:,.4f}",
                }
            ),
            use_container_width=True,
        )
        fig_macro = go.Figure()
        for code, df_s in macro_results.items():
            fig_macro.add_trace(
                go.Scatter(
                    x=df_s["Year"],
                    y=df_s["GDP_forecast"],
                    mode="lines+markers",
                    name=SCENARIOS[code]["name"],
                )
            )
        fig_macro.update_layout(
            title="M1 - Dự báo GDP 2026-2030 theo 5 kịch bản",
            xaxis_title="Năm",
            yaxis_title="GDP dự báo",
            legend_title="Kịch bản",
        )
        st.plotly_chart(fig_macro, use_container_width=True)
        gdp_top = macro_summary.sort_values("GDP 2030", ascending=False).iloc[0]
        gdp_low = macro_summary.sort_values("GDP 2030").iloc[0]
        result_note(
            "Nhận xét M1",
            f"Kịch bản có GDP 2030 cao nhất là {gdp_top['Kịch bản']} với {gdp_top['GDP 2030']:,.2f}.",
            f"Kịch bản thấp nhất là {gdp_low['Kịch bản']} với {gdp_low['GDP 2030']:,.2f}, tạo khoảng cách {gdp_top['GDP 2030'] - gdp_low['GDP 2030']:,.2f}.",
            "Các kịch bản tăng mạnh AI hoặc vốn truyền thống thường cải thiện GDP nhanh hơn, nhưng cần đối chiếu với M5 để xem rủi ro đi kèm.",
        )

    with tabs[2]:
        section("M2 — Đánh giá sẵn sàng số theo vùng", "🌍")
        region_cols = ["Rank", "Vùng", "Digital_Index", "AI_Readiness", "Internet", "Labor_Trained", "Gini", "Regional_Readiness"]
        st.dataframe(
            regions_rank[region_cols].style.format({"Regional_Readiness": "{:.4f}", "Gini": "{:.3f}"}),
            use_container_width=True,
        )
        fig_region = px.bar(
            regions_rank,
            x="Regional_Readiness",
            y="Vùng",
            orientation="h",
            title="M2 - Xếp hạng Regional Readiness theo vùng",
            color="Regional_Readiness",
            color_continuous_scale=["#F9A8D4", "#67E8F9", "#A78BFA"],
        )
        fig_region.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Regional Readiness Score")
        st.plotly_chart(fig_region, use_container_width=True)
        top_region = regions_rank.iloc[0]
        low_region = regions_rank.iloc[-1]
        result_note(
            "Nhận xét M2",
            f"Vùng dẫn đầu là {top_region['Vùng']} với Regional Readiness = {top_region['Regional_Readiness']:.4f}.",
            f"Vùng thấp nhất là {low_region['Vùng']} với {low_region['Regional_Readiness']:.4f}, chủ yếu do Digital Index, AI Readiness hoặc Labor Trained còn yếu.",
            "Hàm ý chính sách là cần phân tầng đầu tư theo năng lực nền: vùng dẫn đầu nên được dùng làm cực tăng trưởng, vùng thấp cần gói bao trùm số và đào tạo nhân lực số.",
        )

    with tabs[3]:
        section("M3 — Phân bổ ngân sách theo kịch bản", "💰")
        st.caption("Tổng ngân sách mô phỏng: 50.000 tỷ.")
        st.dataframe(
            budget_df.style.format({"K": "{:,.0f}", "D": "{:,.0f}", "AI": "{:,.0f}", "H": "{:,.0f}", "Tổng": "{:,.0f}"}),
            use_container_width=True,
        )
        budget_plot = budget_df.melt(
            id_vars=["Mã", "Kịch bản", "Tổng"],
            value_vars=["K", "D", "AI", "H"],
            var_name="Cấu phần",
            value_name="Tỷ VND",
        )
        fig_budget = px.bar(
            budget_plot,
            x="Kịch bản",
            y="Tỷ VND",
            color="Cấu phần",
            title="M3 - Phân bổ ngân sách 50.000 tỷ theo kịch bản",
            color_discrete_sequence=COLORS,
        )
        fig_budget.update_layout(barmode="stack", xaxis_title="", yaxis_title="Tỷ VND")
        st.plotly_chart(fig_budget, use_container_width=True)
        s3_ai = budget_df.loc[budget_df["Mã"] == "S3", "AI"].iloc[0]
        s4_h = budget_df.loc[budget_df["Mã"] == "S4", "H"].iloc[0]
        result_note(
            "Nhận xét M3",
            f"S3 là kịch bản dồn nguồn lực AI mạnh nhất với {s3_ai:,.0f} tỷ cho AI.",
            f"S4 là kịch bản ưu tiên nhân lực số cao nhất với {s4_h:,.0f} tỷ cho H.",
            "Trade-off cốt lõi nằm ở việc chọn tốc độ công nghệ hay mức bao trùm xã hội: tăng AI nhanh có thể kéo năng lực công nghệ, nhưng nếu H thấp thì rủi ro lao động và an ninh dữ liệu sẽ tăng.",
        )

    with tabs[4]:
        section("M4 — 9 ngành chiến lược và chính sách riêng", "🏭")
        sector_cols = [
            "Rank",
            "Ngành",
            "Growth",
            "Productivity",
            "Spillover",
            "Export",
            "Labor",
            "AI_Readiness",
            "Automation_Risk",
            "Priority_Score",
            "Labor_Vulnerability",
            "Policy_Group",
            "Policy_Recommendation",
        ]
        st.dataframe(
            sector_rank[sector_cols].style.format(
                {
                    "Growth": "{:.2f}",
                    "Productivity": "{:,.1f}",
                    "Spillover": "{:.2f}",
                    "Export": "{:,.1f}",
                    "Labor": "{:.2f}",
                    "Priority_Score": "{:.4f}",
                    "Labor_Vulnerability": "{:.4f}",
                }
            ),
            use_container_width=True,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            fig_sector = px.bar(
                sector_rank,
                x="Priority_Score",
                y="Ngành",
                orientation="h",
                title="M4 - Xếp hạng ưu tiên chính sách theo 9 ngành",
                color="Priority_Score",
                color_continuous_scale=["#F9A8D4", "#67E8F9", "#A78BFA"],
            )
            fig_sector.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Priority Score")
            st.plotly_chart(fig_sector, use_container_width=True)
        with col_b:
            fig_scatter = px.scatter(
                sector_rank,
                x="AI_Readiness",
                y="Automation_Risk",
                size="Labor",
                color="Policy_Group",
                hover_name="Ngành",
                title="AI Readiness vs Automation Risk",
                color_discrete_sequence=COLORS,
            )
            fig_scatter.update_layout(xaxis_title="AI Readiness", yaxis_title="Automation Risk (%)")
            st.plotly_chart(fig_scatter, use_container_width=True)
        result_note(
            "Nhận xét M4",
            f"Ngành ưu tiên cao nhất là {top_sector['Ngành']} với Priority Score = {top_sector['Priority_Score']:.4f}.",
            f"Ngành có Automation Risk cao nhất là {high_risk_sector['Ngành']} với {high_risk_sector['Automation_Risk']:.1f}%.",
            "Các ngành có AI Readiness cao nhưng rủi ro tự động hóa lớn cần chính sách kép: đầu tư AI đi kèm đào tạo lại lao động và bảo vệ dữ liệu.",
        )
        section("Policy cards theo nhóm ngành", "🧁")
        for group, group_df in sector_rank.groupby("Policy_Group", sort=True):
            names = ", ".join(group_df["Ngành"].tolist())
            rec = group_df["Policy_Recommendation"].iloc[0]
            policy_card(group, f"<b>Ngành thuộc nhóm:</b> {names}.<br><b>Khuyến nghị:</b> {rec}", "🍬")

    with tabs[5]:
        section("M5 — NetJob và rủi ro theo kịch bản", "⚠️")
        st.dataframe(
            labor_risk_df.style.format(
                {
                    "Total NewJob": "{:,.0f}",
                    "Total UpgradeJob": "{:,.0f}",
                    "Total DisplacedJob": "{:,.0f}",
                    "Total NetJob": "{:,.0f}",
                    "Cyber Risk": "{:.2f}",
                    "Emission Risk": "{:.2f}",
                    "Dependency Risk": "{:.2f}",
                    "Total Risk": "{:.2f}",
                }
            ),
            use_container_width=True,
        )
        col_jobs, col_risk = st.columns(2)
        with col_jobs:
            fig_jobs = px.bar(
                labor_risk_df,
                x="Kịch bản",
                y="Total NetJob",
                title="M5 - Total NetJob theo kịch bản",
                color="Total NetJob",
                color_continuous_scale=["#F9A8D4", "#86EFAC", "#67E8F9"],
            )
            fig_jobs.update_layout(xaxis_title="", yaxis_title="Việc làm ròng")
            st.plotly_chart(fig_jobs, use_container_width=True)
        with col_risk:
            risk_plot = labor_risk_df.melt(
                id_vars=["Mã", "Kịch bản"],
                value_vars=["Cyber Risk", "Emission Risk", "Dependency Risk"],
                var_name="Loại rủi ro",
                value_name="Risk score",
            )
            fig_risk = px.bar(
                risk_plot,
                x="Kịch bản",
                y="Risk score",
                color="Loại rủi ro",
                barmode="group",
                title="M5 - Cyber, Emission, Dependency Risk",
                color_discrete_sequence=COLORS,
            )
            fig_risk.update_layout(xaxis_title="", yaxis_title="Risk score")
            st.plotly_chart(fig_risk, use_container_width=True)
        job_top = labor_risk_df.sort_values("Total NetJob", ascending=False).iloc[0]
        risk_top = labor_risk_df.sort_values("Total Risk", ascending=False).iloc[0]
        result_note(
            "Nhận xét M5",
            f"Kịch bản tạo nhiều việc làm ròng nhất là {job_top['Kịch bản']} với {job_top['Total NetJob']:,.0f} NetJob.",
            f"Kịch bản rủi ro tổng hợp cao nhất là {risk_top['Kịch bản']} với Total Risk = {risk_top['Total Risk']:.2f}.",
            "Kết quả nhấn mạnh rằng chính sách AI không chỉ tối đa hóa việc làm mới, mà phải bù đắp displaced jobs bằng đào tạo lại và kiểm soát cyber/emission/dependency risk.",
        )

        selected_labor = st.selectbox(
            "Xem chi tiết NetJob theo ngành cho một kịch bản",
            list(SCENARIOS.keys()),
            format_func=lambda code: SCENARIOS[code]["name"],
            key="b12_labor_detail",
        )
        detail_df = labor_detail[selected_labor]
        st.dataframe(
            detail_df.style.format(
                {"x_AI": "{:,.1f}", "x_H": "{:,.1f}", "NewJob": "{:,.1f}", "UpgradeJob": "{:,.1f}", "DisplacedJob": "{:,.1f}", "NetJob": "{:,.1f}"}
            ),
            use_container_width=True,
        )

    with tabs[6]:
        section("M6 — AIDEOM Score và bảng ra quyết định", "🏆")
        score_cols = ["Rank", "Mã", "Kịch bản", "GDP 2030", "Total NetJob", "Total Risk", "GDP_norm", "NetJob_norm", "Risk_norm", "AIDEOM_Score"]
        st.dataframe(
            score_rank[score_cols].style.format(
                {
                    "GDP 2030": "{:,.2f}",
                    "Total NetJob": "{:,.0f}",
                    "Total Risk": "{:.2f}",
                    "GDP_norm": "{:.4f}",
                    "NetJob_norm": "{:.4f}",
                    "Risk_norm": "{:.4f}",
                    "AIDEOM_Score": "{:.4f}",
                }
            ),
            use_container_width=True,
        )
        col_score, col_norm = st.columns(2)
        with col_score:
            fig_score = px.bar(
                score_rank,
                x="AIDEOM_Score",
                y="Kịch bản",
                orientation="h",
                title="M6 - Xếp hạng 5 kịch bản theo AIDEOM Score",
                color="AIDEOM_Score",
                color_continuous_scale=["#F9A8D4", "#67E8F9", "#A78BFA"],
            )
            fig_score.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="AIDEOM Score")
            st.plotly_chart(fig_score, use_container_width=True)
        with col_norm:
            norm_plot = score_rank.melt(
                id_vars=["Mã", "Kịch bản"],
                value_vars=["GDP_norm", "NetJob_norm", "Risk_norm", "AIDEOM_Score"],
                var_name="Chỉ số",
                value_name="Điểm chuẩn hóa",
            )
            fig_norm = px.bar(
                norm_plot,
                x="Kịch bản",
                y="Điểm chuẩn hóa",
                color="Chỉ số",
                barmode="group",
                title="So sánh chuẩn hóa GDP - NetJob - Risk - AIDEOM Score",
                color_discrete_sequence=COLORS,
            )
            fig_norm.update_layout(xaxis_title="", yaxis_title="Điểm chuẩn hóa")
            st.plotly_chart(fig_norm, use_container_width=True)
        result_note(
            "Nhận xét M6",
            f"Kịch bản tốt nhất theo AIDEOM Score là {best['Kịch bản']} với điểm {best['AIDEOM_Score']:.4f}.",
            f"GDP cao nhất thuộc về {highest_gdp['Kịch bản']}, NetJob cao nhất thuộc về {highest_jobs['Kịch bản']}, còn rủi ro thấp nhất thuộc về {lowest_risk['Kịch bản']}.",
            "Điều này cho thấy nghiệm tốt nhất không nhất thiết tối đa hóa một mục tiêu đơn lẻ, mà cân bằng giữa tăng trưởng, việc làm và kiểm soát rủi ro.",
        )
        section("Cảnh báo rủi ro tự động", "🚨")
        for warning in warning_rows(score_rank):
            policy_card("Cảnh báo rủi ro tự động", warning, "🚨")

    with tabs[7]:
        section("AI Policy Agent", "🤖")
        st.caption("Bấm nút bên dưới để hiển thị phân tích chính sách mô phỏng dựa trên kết quả hiện tại.")
        state_key = "b12_policy_agent_visible"
        if st.button("🤖 Phân tích bằng AI Agent", key="b12_ai_agent_button", use_container_width=True):
            st.session_state[state_key] = True
        if st.session_state.get(state_key, False):
            with st.spinner("AI Policy Agent đang đọc bảng AIDEOM Score, M4 và M5..."):
                time.sleep(0.35)
            policy_card(
                "AI Policy Agent — Phân tích tự động",
                f"""
                <b>1. Kịch bản tổng hợp tốt nhất:</b> {best['Kịch bản']} với AIDEOM Score = {best['AIDEOM_Score']:.4f}.<br>
                <b>2. Trade-off chính:</b> GDP 2030 cao nhất thuộc về {highest_gdp['Kịch bản']}; NetJob cao nhất thuộc về {highest_jobs['Kịch bản']}; rủi ro thấp nhất thuộc về {lowest_risk['Kịch bản']}.<br>
                <b>3. Ngành ưu tiên:</b> {top_sector['Ngành']} dẫn đầu Priority Score, nên được xem là ngành mũi nhọn để tạo lan tỏa AI và dữ liệu.<br>
                <b>4. Ngành rủi ro:</b> {high_risk_sector['Ngành']} có Automation Risk cao nhất, cần gói đào tạo lại và bảo vệ việc làm trước khi mở rộng tự động hóa.<br>
                <b>5. Khuyến nghị:</b> Việt Nam không nên theo đuổi AI cực đoan nếu chưa đủ nền tảng nhân lực số, an ninh dữ liệu và cơ chế chuyển đổi lao động. Chiến lược phù hợp là kết hợp hạ tầng số, AI có kiểm soát, nhân lực số và cảnh báo rủi ro sớm.
                """,
                "🤖",
            )
        else:
            policy_card(
                "AI Agent sẵn sàng phân tích",
                "Bấm nút <b>Phân tích bằng AI Agent</b> để hiển thị nhận xét dựa trên best scenario, highest GDP, highest NetJob, lowest Risk, top sector và high-risk sector.",
                "🧁",
            )

        section("Câu hỏi thảo luận chính sách tích hợp", "📈")
        policy_card(
            "1. Có nên chọn kịch bản AIDEOM Score cao nhất?",
            f"Kịch bản dẫn đầu là <b>{best['Kịch bản']}</b>, nhưng quyết định cuối cùng cần đọc cùng GDP, NetJob và Risk. Nếu chỉ tối đa hóa GDP, chính sách có thể bỏ qua rủi ro an ninh dữ liệu hoặc dịch chuyển lao động. Vì vậy, AIDEOM Score nên được xem như bảng điều khiển cân bằng, không phải mệnh lệnh cơ học.",
            "📌",
        )
        policy_card(
            "2. Ngành ưu tiên có cần chính sách riêng không?",
            f"Ngành ưu tiên cao nhất là <b>{top_sector['Ngành']}</b>, trong khi ngành rủi ro tự động hóa cao nhất là <b>{high_risk_sector['Ngành']}</b>. Điều này cho thấy chính sách ngành phải phân tầng: ngành mũi nhọn cần R&D, dữ liệu và sandbox; ngành rủi ro cần đào tạo lại, an sinh và bảo vệ việc làm.",
            "🏭",
        )
        policy_card(
            "3. Rủi ro có nên được đặt ngang tăng trưởng?",
            "Nếu Total Risk tăng nhanh, lợi ích GDP có thể không bền vững vì chi phí cyber, emission và dependency sẽ xuất hiện trễ. Do đó, dashboard khuyến nghị tích hợp cảnh báo rủi ro vào quá trình phân bổ ngân sách, đặc biệt khi tăng tỷ trọng AI và hạ tầng số.",
            "⚠️",
        )

        section("Hướng mở rộng nghiên cứu", "🔬")
        future_research = pd.DataFrame(
            {
                "Hướng mở rộng": [
                    "Bài báo SCIE Q2-Q3",
                    "CGE hoặc DSGE-AI",
                    "Dữ liệu thời gian thực",
                    "Multi-Agent Reinforcement Learning",
                ],
                "Mô tả": [
                    "Chọn một use case cụ thể như ĐBSCL hoặc ngành chế biến chế tạo để viết bài nghiên cứu định lượng.",
                    "Mở rộng từ mô hình tối ưu cục bộ sang cân bằng tổng thể để đánh giá tác động liên ngành.",
                    "Kết nối Open Data Portal, Vietstock, hải quan, dữ liệu lao động để cập nhật theo tháng/quý.",
                    "Mở rộng Q-learning thành môi trường nhiều agent đại diện cho bộ, ngành, vùng và doanh nghiệp.",
                ],
                "Giá trị học thuật": [
                    "Tạo sản phẩm nghiên cứu có thể công bố.",
                    "Tăng độ chặt chẽ kinh tế học và mô phỏng chính sách vĩ mô.",
                    "Biến AIDEOM-VN thành policy intelligence cập nhật liên tục.",
                    "Mô phỏng xung đột lợi ích và phối hợp chính sách giữa nhiều tác nhân.",
                ],
            }
        )
        st.dataframe(future_research, use_container_width=True)

        section("Kết luận", "💗")
        st.success(
            """
            Bài 12 đã xây dựng nguyên mẫu **AIDEOM-VN Policy Lab**, tích hợp các kỹ thuật từ Bài 1 đến Bài 11
            thành một hệ thống hỗ trợ ra quyết định. Hệ thống gồm M1 dự báo kinh tế vĩ mô, M2 đánh giá sẵn sàng số,
            M3 phân bổ ngân sách, M4 chính sách theo 9 ngành, M5 lao động và rủi ro, M6 AIDEOM Score và AI Policy Agent.

            Điểm mạnh của AIDEOM-VN là không tạo ra một đáp án duy nhất, mà cho phép so sánh kịch bản,
            nhận diện đánh đổi, xác định ngành ưu tiên, cảnh báo rủi ro và đưa ra khuyến nghị chính sách.
            Kết quả nhấn mạnh Việt Nam cần cân bằng giữa tăng trưởng, chuyển đổi số, AI, nhân lực số,
            an sinh lao động và kiểm soát rủi ro.
            """
        )
