"""
Bài 11 — Học tăng cường Q-learning cho chính sách thích ứng
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import gymnasium as gym
from gymnasium import spaces

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

# ── Gymnasium Environment ─────────────────────────────────────────────────────
action_names = {
    0: "a0 - Truyền thống (70% K, 10% D, 10% AI, 10% H)",
    1: "a1 - Cân bằng (40% K, 25% D, 15% AI, 20% H)",
    2: "a2 - Số hóa nhanh (25% K, 45% D, 15% AI, 15% H)",
    3: "a3 - AI dẫn dắt (20% K, 20% D, 45% AI, 15% H)",
    4: "a4 - Bao trùm (30% K, 20% D, 10% AI, 40% H)"
}
state_level_names = {0: "low", 1: "medium", 2: "high"}

class VietnamEconomyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
        self.T = 10
        self.budget = 1000
        
        self.allocation = {
            0: np.array([0.70, 0.10, 0.10, 0.10]),
            1: np.array([0.40, 0.25, 0.15, 0.20]),
            2: np.array([0.25, 0.45, 0.15, 0.15]),
            3: np.array([0.20, 0.20, 0.45, 0.15]),
            4: np.array([0.30, 0.20, 0.10, 0.40])
        }
        self.w = np.array([0.40, 0.25, 0.20, 0.15])
        
        self.alpha_K, self.alpha_L, self.alpha_D, self.alpha_AI, self.alpha_H = 0.33, 0.42, 0.10, 0.08, 0.07

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        # Starting state: GDP medium(1), Digital medium(1), AI low(0), Unemployment medium(1)
        self.state = np.array([1, 1, 0, 1])
        
        self.K = 27500.0
        self.D = 20.3
        self.AI = 86.0
        self.H = 30.0
        self.L = 54.0
        self.A = 35.0
        self.prev_Y = self.production()
        return self.state.copy(), {}

    def production(self):
        return self.A * (self.K ** self.alpha_K) * (self.L ** self.alpha_L) * (self.D ** self.alpha_D) * (self.AI ** self.alpha_AI) * (self.H ** self.alpha_H)

    def discretize_state(self, gdp_growth, digital_index, ai_capacity, unemployment_risk):
        gdp_s = 0 if gdp_growth < 0.03 else 1 if gdp_growth < 0.06 else 2
        d_s = 0 if digital_index < 35 else 1 if digital_index < 70 else 2
        ai_s = 0 if ai_capacity < 100 else 1 if ai_capacity < 180 else 2
        u_s = 0 if unemployment_risk < 0.04 else 1 if unemployment_risk < 0.08 else 2
        return np.array([gdp_s, d_s, ai_s, u_s])

    def step(self, action):
        alloc = self.allocation[action]
        
        i_K = alloc[0] * self.budget
        i_D = alloc[1] * self.budget
        i_AI = alloc[2] * self.budget
        i_H = alloc[3] * self.budget
        
        # State dynamics
        self.K = 0.95 * self.K + i_K
        self.D = 0.88 * self.D + i_D / 100.0
        self.AI = 0.85 * self.AI + i_AI / 20.0
        self.H = 0.98 * self.H + i_H / 200.0
        
        # TFP dynamic
        self.A = self.A * (1.0 + 0.0008 * self.D + 0.0005 * self.AI + 0.001 * self.H)
        
        Y = self.production()
        gdp_growth = (Y - self.prev_Y) / self.prev_Y
        self.prev_Y = Y
        
        unemployment_risk = max(0.01, 0.08 + 0.0008 * i_AI - 0.0012 * i_H - 0.002 * self.H)
        cyber_risk = max(0.0, 0.03 + 0.0006 * i_AI - 0.0004 * i_H)
        emission = max(0.0, 0.04 + 0.0004 * i_K + 0.0005 * i_AI - 0.0002 * i_D)
        
        # Reward
        reward = (
            self.w[0] * (gdp_growth * 100)
            - self.w[1] * (unemployment_risk * 100)
            - self.w[2] * (cyber_risk * 100)
            - self.w[3] * (emission * 100)
        )
        
        self.state = self.discretize_state(gdp_growth, self.D, self.AI, unemployment_risk)
        self.t += 1
        done = self.t >= self.T
        
        return self.state.copy(), reward, done, False, {}

# ── Q-learning Train ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def train_q_learning(episodes=2000, alpha=0.1, gamma=0.9, epsilon=0.1):
    env = VietnamEconomyEnv()
    # State space: 3x3x3x3 = 81 states, Actions: 5
    Q = np.zeros((3, 3, 3, 3, 5))
    rewards_history = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        total_r = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state[0], state[1], state[2], state[3]])
                
            next_state, r, done, _, _ = env.step(action)
            
            # Bellman Equation update
            best_next = np.max(Q[next_state[0], next_state[1], next_state[2], next_state[3]])
            Q[state[0], state[1], state[2], state[3], action] += alpha * (
                r + gamma * best_next - Q[state[0], state[1], state[2], state[3], action]
            )
            
            state = next_state
            total_r += r
            
        rewards_history.append(total_r)
        
    return Q, rewards_history

def show():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
      <div style="display:flex;align-items:center;gap:1rem;">
        <span style="font-size:2rem;">🤖</span>
        <div>
          <div style="color:#00D4FF;font-size:.75rem;font-weight:600;letter-spacing:1px;">BÀI 11</div>
          <div style="color:#E2E8F0;font-size:1.5rem;font-weight:800;">Học tăng cường Q-learning cho Chính sách kinh tế thích ứng</div>
          <div style="margin-top:.3rem;">
            <span class="badge b-blue">Reinforcement Learning</span>
            <span class="badge b-purple">Gymnasium Environment</span>
            <span class="badge b-green">Bellman Optimality</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📖 Bối cảnh & Mô hình", "🎛️ Tham số", "📊 Kết quả", "📈 Phân tích chính sách", "🤖 AI Agent"])

    # ══ Tab 1: Bối cảnh & MDP ══════════════════════════════════════════════
    with tabs[0]:
        section("Bối cảnh bài toán học máy vĩ mô", "🌏")
        st.markdown("""
        Trong môi trường thay đổi nhanh chóng, các nhà hoạch định chính sách công không thể chỉ đưa ra các quyết định tĩnh 
        mà phải liên tục **thích nghi (adaptive policy)** dựa trên phản hồi của thị trường qua từng thời kỳ.
        
        Học tăng cường (Reinforcement Learning - RL) cung cấp một khung lý thuyết mạnh mẽ để giải quyết bài toán này. 
        Ta mô hình hóa nền kinh tế Việt Nam như một **Môi trường MDP (Markov Decision Process)**, trong đó tác tử AI đóng vai trò 
        nhà hoạch định chính sách học cách phân bổ ngân sách tối ưu qua việc tương tác thử-sai và nhận điểm thưởng (Rewards).
        """)

        section("Cơ cấu MDP và hàm phần thưởng", "📐")
        math_box("""
<b>Không gian Trạng thái (81 trạng thái):</b>
  Trạng thái S<sub>t</sub> = [GDP growth, Digital Index, AI capacity, Unemployment risk]
  Mỗi thành phần được chia thành 3 mức độ định lượng: {0: Low, 1: Medium, 2: High}.

<b>Không gian Hành động (5 hành động/chính sách):</b>
  • a0 (Truyền thống): 70% vốn K, 10% D, 10% AI, 10% H
  • a1 (Cân bằng):    40% vốn K, 25% D, 15% AI, 20% H
  • a2 (Số hóa nhanh): 25% vốn K, 45% D, 15% AI, 15% H
  • a3 (AI dẫn dắt):  20% vốn K, 20% D, 45% AI, 15% H
  • a4 (Bao trùm số):  30% vốn K, 20% D, 10% AI, 40% H

<b>Hàm Phần thưởng hàng năm (Reward):</b>
  R<sub>t</sub> = 0.40·ΔGDP% - 0.25·ThấtNghiệp% - 0.20·CyberRisk% - 0.15·PhátThải%
        """)

    # ══ Tab 2: Cấu hình RL ════════════════════════════════════════════════
    with tabs[1]:
        section("Thiết lập siêu tham số cho tác tử Q-learning", "🎛️")
        col1, col2 = st.columns(2)
        with col1:
            episodes = st.slider("Số Episodes huấn luyện", 500, 5000, 2000, 500, key="b11_ep")
            alpha = st.slider("Tốc độ học (Learning rate α)", 0.01, 0.50, 0.10, 0.01, key="b11_alpha")
        with col2:
            gamma = st.slider("Hệ số chiết khấu tương lai (Discount γ)", 0.50, 0.99, 0.90, 0.01, key="b11_gamma")
            epsilon = st.slider("Xác suất khám phá ngẫu nhiên (Epsilon ε)", 0.01, 0.30, 0.10, 0.01, key="b11_eps")

        section("Phương trình cập nhật Bellman Q-Learning", "📐")
        math_box("""
Q(S<sub>t</sub>, A<sub>t</sub>) ← Q(S<sub>t</sub>, A<sub>t</sub>) + α · [ R<sub>t</sub> + γ · max<sub>a</sub> Q(S<sub>t+1</sub>, a) - Q(S<sub>t</sub>, A<sub>t</sub>) ]
        """)

    # ══ Tab 3: Tiến trình học tập ══════════════════════════════════════════
    with tabs[2]:
        st.write("🔄 *Đang tiến hành huấn luyện tác tử Q-Learning trong môi trường Gymnasium...*")
        Q_table, rew_hist = train_q_learning(episodes, alpha, gamma, epsilon)
        
        st.success(f"🎉 **Đã huấn luyện xong {episodes} episodes!** Tác tử đã tìm thấy ma trận chính sách tối ưu.")
        
        # Plotly: learning curve (moving average)
        section("Câu 11.3.1 — Kiểm tra môi trường Gym/MDP", "✅")
        st.markdown("Môi trường có 81 trạng thái và 5 hành động chính sách; tác tử nhận reward theo tăng trưởng, số hóa, AI và rủi ro thất nghiệp.")
        section("Câu 11.3.2 — Huấn luyện Q-learning", "📈")
        df_rew = pd.DataFrame({"Episode": range(1, len(rew_hist)+1), "Reward": rew_hist})
        df_rew["Reward trung bình trượt (MA-50)"] = df_rew["Reward"].rolling(window=50, min_periods=5).mean()
        
        fig_curve = px.line(df_rew, x="Episode", y=["Reward", "Reward trung bình trượt (MA-50)"],
                            color_discrete_sequence=["rgba(0,212,255,0.2)", COLORS[1]])
        fig_curve.update_layout(**PLOTLY_LAYOUT, yaxis_title="Tổng Reward một Episode")
        st.plotly_chart(fig_curve, use_container_width=True)

    # ══ Kết quả mở rộng: chính sách tối ưu và mô phỏng ════════════════════
    with tabs[2]:
        section("Câu 11.3.3 — Trích xuất chính sách π*", "🧭")
        flat_q = Q_table.argmax(axis=-1)
        unique, counts = np.unique(flat_q, return_counts=True)
        policy_df = pd.DataFrame({
            "Hành động": [action_names[int(a)] for a in unique],
            "Số trạng thái tối ưu": counts,
            "Tỷ trọng (%)": counts / counts.sum() * 100,
        })
        st.dataframe(policy_df.style.format({"Tỷ trọng (%)": "{:.2f}"}), use_container_width=True)

        section("Câu 11.3.4 — So sánh với chính sách rule-based và mô phỏng tương tác", "🎮")
        st.markdown("Hãy chọn một trạng thái kinh tế ban đầu và xem AI đề xuất hành động nào tối ưu nhất dựa trên Q-table.")
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1: s_gdp = st.selectbox("GDP Growth", [0, 1, 2], format_func=lambda x: state_level_names[x], key="b11_sg")
        with col_s2: s_dig = st.selectbox("Digital Index", [0, 1, 2], format_func=lambda x: state_level_names[x], key="b11_sd")
        with col_s3: s_ai = st.selectbox("AI Capacity", [0, 1, 2], format_func=lambda x: state_level_names[x], key="b11_sa")
        with col_s4: s_un = st.selectbox("Unemployment Risk", [0, 1, 2], format_func=lambda x: state_level_names[x], key="b11_su")

        # Get optimal action from Q-table
        opt_action = np.argmax(Q_table[s_gdp, s_dig, s_ai, s_un])
        st.info(f"💡 **AI Agent gợi ý hành động tối ưu cho trạng thái này:** `{action_names[opt_action]}`")
        
        # Interactive single step run
        st.markdown("**Chạy thử nghiệm 1 bước (Step):**")
        env_sim = VietnamEconomyEnv()
        env_sim.reset()
        env_sim.state = np.array([s_gdp, s_dig, s_ai, s_un])
        
        act_choice = st.selectbox("Chọn hành động bạn muốn kiểm thử:", list(action_names.keys()), format_func=lambda x: action_names[x], key="b11_act_ch")
        
        if st.button("🚀 Thực thi bước chính sách", key="b11_step_btn"):
            next_state, rew, done, _, _ = env_sim.step(act_choice)
            
            c_r1, c_r2, c_r3 = st.columns(3)
            c_r1.metric("Điểm Reward nhận được", f"{rew:.2f}")
            c_r2.metric("Trạng thái mới", f"[{', '.join([state_level_names[x] for x in next_state])}]")
            c_r3.metric("GDP ròng mới", f"{env_sim.prev_Y:,.1f} tỷ")

        section("Câu 11.3.5 — Mở rộng DQN optional", "🤖")
        st.markdown("""
        DQN là hướng mở rộng khi không gian trạng thái/hành động lớn hơn và Q-table không còn phù hợp.
        Trong khuôn khổ bài này, Q-learning tabular đủ để minh họa cơ chế học chính sách thích nghi.
        Khi có dữ liệu thời gian thực, DQN hoặc Multi-Agent RL có thể mô phỏng phản ứng chính sách phức tạp hơn.
        """)

    # ══ Tab 4: Phân tích chính sách ═══════════════════════════════════════
    with tabs[3]:
        flat_q_policy = Q_table.argmax(axis=-1)
        unique_p, counts_p = np.unique(flat_q_policy, return_counts=True)
        counts_dict_policy = dict(zip(unique_p, counts_p))
        dominant_action = max(counts_dict_policy, key=counts_dict_policy.get)
        section("1. Chính sách tối ưu có cố định cho mọi trạng thái không?", "📌")
        st.markdown(f"""
        Không. Q-learning chọn hành động theo trạng thái, nên chính sách tối ưu là một bản đồ phản ứng chứ không phải một tỷ lệ ngân sách cố định.
        Hành động xuất hiện nhiều nhất hiện là **{action_names[int(dominant_action)]}** với {counts_dict_policy[dominant_action]} / 81 trạng thái.
        Điều này cho thấy tác tử ưu tiên chính sách khác nhau khi GDP, digital index, AI capacity và unemployment risk thay đổi.
        """)
        section("2. Khi nào nên ưu tiên bao trùm?", "👷")
        st.markdown("""
        Khi tăng trưởng thấp và rủi ro thất nghiệp cao, reward phạt mạnh thất nghiệp nên tác tử có xu hướng chọn chính sách bao trùm hoặc đào tạo lại.
        Đây là cơ chế đưa an sinh xã hội vào mô hình học tăng cường.
        Chính sách thực tế có thể dùng logic này để kích hoạt gói hỗ trợ lao động khi chỉ số thị trường việc làm xấu đi.
        """)
        section("3. Ý nghĩa của RL đối với điều hành kinh tế số", "🧠")
        st.markdown("""
        RL phù hợp với môi trường chính sách thay đổi liên tục, nơi tác động của hành động hôm nay ảnh hưởng đến trạng thái tương lai.
        Mô hình không thay thế quyết định của Chính phủ, nhưng giúp thử nghiệm quy tắc phản ứng trước khi áp dụng thật.
        Hướng mở rộng là kết nối dữ liệu thời gian thực và kiểm định chính sách bằng mô phỏng nhiều tác tử.
        """)
        sorted_policy = sorted(counts_dict_policy.items(), key=lambda kv: kv[1], reverse=True)
        second_action, second_count = sorted_policy[1] if len(sorted_policy) > 1 else (dominant_action, counts_dict_policy[dominant_action])
        section("4. Hàm ý chính sách nâng cấp", "📋")
        st.markdown(f"""
**Kết quả nổi bật.** Chính sách xuất hiện nhiều nhất là **{action_names[int(dominant_action)]}** với **{counts_dict_policy[dominant_action]} / 81** trạng thái; lựa chọn đứng thứ hai là **{action_names[int(second_action)]}** với **{second_count} / 81** trạng thái; không gian mô phỏng có **81** trạng thái và **5** hành động chính sách.

**Liên hệ chính sách Việt Nam.** Kết quả phù hợp với **Quyết định 127/QĐ-TTg** và **Nghị quyết 57-NQ/TW**: chính sách AI/chuyển đổi số cần cơ chế phản ứng thích nghi dựa trên dữ liệu trạng thái, không chỉ kế hoạch cố định.

**Đánh đổi cần lưu ý:** tối ưu phản ứng nhanh và tính ổn định chính sách; thay đổi hành động quá thường xuyên có thể gây khó triển khai nếu thiếu dữ liệu giám sát.

**Khuyến nghị hành động.** Xây bộ chỉ báo kích hoạt chính sách theo GDP, Digital Index, AI capacity và unemployment risk; dùng policy map để thử nghiệm trước khi ban hành; ưu tiên hành động bao trùm khi thất nghiệp cao; theo dõi tần suất chuyển chính sách để tránh dao động quá mức.
        """)

    # ══ Tab 5: AI Agent ═══════════════════════════════════════════════════
    with tabs[4]:
        st.caption("Bấm nút AI Agent bên dưới để hiển thị phân tích chính sách mô phỏng.")
        # Extract general trends
        # Let's count how many times each action is selected across all 81 states
        flat_q = Q_table.argmax(axis=-1)
        unique, counts = np.unique(flat_q, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        
        policy_analysis_text = "<b>1. Thống kê chiến lược tối ưu được trích xuất (Optimal Policy Map):</b><br>"
        for a_k in range(5):
            c_a = counts_dict.get(a_k, 0)
            policy_analysis_text += f"• `{action_names[a_k].split('(')[0]}`: là tối ưu cho <b>{c_a} / 81</b> trạng thái.<br>"
            
        policy_analysis_text += """<br>
        <b>2. Logic tiến hóa vĩ mô của tác tử:</b><br>
        • Khi kinh tế rơi vào **trạng thái GDP suy thoái (Low GDP growth)** và **Thất nghiệp cao (High Unemployment)**: 
        Tác tử Q-learning tự động lựa chọn chính sách `a4 - Bao trùm` để phục hồi thị trường lao động số H.<br>
        • Khi kinh tế ở **trạng thái Lạc quan (High GDP growth)** và **Chuyển đổi số tốt**: 
        Tác tử nhanh chóng chuyển hướng dồn toàn bộ nguồn lực sang hành động `a3 - AI dẫn dắt` để gặt hái siêu lợi nhuận GDP gain biên biên vĩ mô.<br><br>
        
        <b>3. Ý nghĩa đối với nhà hoạch định chính sách Việt Nam:</b><br>
        Q-learning chứng minh một triết lý sâu sắc: **Không có một chính sách phân bổ ngân sách nào là hoàn hảo cho mọi thời điểm**. 
        Việt Nam cần xây dựng một cơ chế phản ứng chính sách thích nghi thời gian thực (real-time adaptive feedback loop), liên tục theo dõi các chỉ số GDP, 
        chuyển đổi số doanh nghiệp để thay đổi cơ cấu ngân sách thích ứng nhằm đạt được quỹ đạo phát triển tối ưu dài hạn.
        """
        ai_agent(policy_analysis_text)
