"""
价值投资股票筛选器 · Streamlit 多页应用
位于 pages/ 下，会自动出现在 growth_valuation.py 的侧边栏
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import screener_data as sd
import screener_logic as sl


st.set_page_config(page_title="价值股筛选器", page_icon="🔎", layout="wide")
st.title("🔎 价值股筛选器")
st.caption("第一性原理：质量 → 行业 → 护城河 → 安全边际")


# ────────────────────────────────────────────────────────────
# 侧边栏控制
# ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 筛选设置")
    mode = st.radio(
        "严格度",
        ["strict", "standard", "loose"],
        index=1,
        format_func=lambda x: {"strict": "严格 (仅A档)",
                                "standard": "标准 (A+B档)",
                                "loose": "宽松 (A+B+C档)"}[x],
    )
    min_mv = st.number_input("最小市值（亿）", 0, 100000, 100, step=50)
    max_pe = st.number_input("最大 PE（0=不限）", 0, 500, 60, step=5)
    top_n_industry = st.slider("快速筛预过滤：每行业取市值前 N", 5, 50, 20)

    st.divider()
    st.subheader("数据更新")
    if st.button("🔄 刷新全市场快照", width="stretch"):
        status = st.empty()
        try:
            with st.spinner("拉取 A 股快照..."):
                df = sd.refresh_snapshot(force=True, status_cb=lambda m: status.info(m))
            warn = df.attrs.get("fallback_warning")
            src = df.attrs.get("source", "?")
            status.empty()
            if warn:
                st.warning(f"⚠️ 实时源全部失败，已使用：{src}\n详情：{warn}")
            else:
                st.success(f"快照已更新 ｜ 数据源：{src} ｜ {len(df)} 只")
            st.cache_data.clear()
        except Exception as e:
            status.empty()
            st.error(f"快照拉取失败：{e}")
    if st.button("🏭 刷新行业映射（首次/月度）", width="stretch"):
        status = st.empty()
        try:
            with st.spinner("拉取行业归属..."):
                sd.refresh_industry_map(force=True, status_cb=lambda m: status.info(m))
            status.empty()
            st.success("行业映射已更新")
            st.cache_data.clear()
        except Exception as e:
            status.empty()
            st.error(f"行业映射拉取失败：{e}")


# ────────────────────────────────────────────────────────────
# 加载基础数据（不自动拉，避免页面打开就失败）
# ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_base():
    return sd.load_snapshot(), sd.load_industry_map()


snap_df, ind_df = load_base()
if snap_df.empty:
    st.warning("📭 本地无快照数据。请点击左侧「🔄 刷新全市场快照」")
    st.stop()
if ind_df.empty:
    st.warning("📭 本地无行业映射。请点击左侧「🏭 刷新行业映射」（首次约 2-3 分钟）")
    st.stop()

st.info(f"快照日期：{snap_df['snap_date'].iloc[0]} ｜ 全市场 {len(snap_df)} 只 ｜ 行业映射 {len(ind_df)} 只")

wl = sl.load_whitelist()
with st.expander("📋 当前行业白名单"):
    col1, col2, col3 = st.columns(3)
    col1.markdown("**上升期**\n\n" + "\n".join(f"- {x}" for x in wl["rising"]))
    col2.markdown("**成熟稳定**\n\n" + "\n".join(f"- {x}" for x in wl["stable"]))
    col3.markdown("**黑名单**\n\n" + "\n".join(f"- {x}" for x in wl["blacklist"]))
    st.caption("编辑 industry_whitelist.yaml 后刷新页面生效")


# ────────────────────────────────────────────────────────────
# 预过滤：用快照层先砍到几百只，再去拉财务
# ────────────────────────────────────────────────────────────

df = snap_df.merge(ind_df, on="code", how="left")
df = df[df["industry"].notna()]
df = df[df["mv"] >= min_mv]
if max_pe > 0:
    df = df[(df["pe_ttm"] > 0) & (df["pe_ttm"] <= max_pe)]

allowed_inds = set(wl["rising"]) | set(wl["stable"])
df = df[df["industry"].isin(allowed_inds)]

# 行业内市值排名（用于龙头判断）
df["ind_mv_rank"] = df.groupby("industry")["mv"].rank(method="min", ascending=False).astype(int)

# 每行业取前 N
df = df[df["ind_mv_rank"] <= top_n_industry].copy()

st.write(f"**预过滤后候选**：{len(df)} 只（将逐个拉取 5 年财务，约 {len(df) * 0.3:.0f} 秒）")


# ────────────────────────────────────────────────────────────
# 启动评估
# ────────────────────────────────────────────────────────────

if "screener_results" not in st.session_state:
    st.session_state.screener_results = None

col_run, col_clear = st.columns([3, 1])
run_btn = col_run.button("🚀 开始筛选", type="primary", width="stretch")
if col_clear.button("清空结果", width="stretch"):
    st.session_state.screener_results = None

if run_btn:
    progress = st.progress(0.0, "拉取财务数据...")
    results = []
    codes = df["code"].tolist()
    snap_lookup = df.set_index("code").to_dict("index")

    for i, code in enumerate(codes):
        try:
            fin = sd.fetch_financials(code)
            snap = snap_lookup[code]
            res = sl.evaluate(
                snap=snap, fin=fin,
                industry=snap["industry"],
                industry_mv_rank=int(snap["ind_mv_rank"]),
                wl=wl, mode=mode,
            )
            results.append(res)
        except Exception as e:
            results.append({"code": code, "name": snap_lookup[code].get("name"),
                            "error": str(e), "passed": False, "final_score": 0})
        progress.progress((i + 1) / len(codes), f"已处理 {i+1}/{len(codes)}: {code}")

    progress.empty()
    st.session_state.screener_results = results


# ────────────────────────────────────────────────────────────
# 展示结果
# ────────────────────────────────────────────────────────────

results = st.session_state.screener_results
if results:
    res_df = pd.DataFrame(results)

    pass_df = res_df[res_df["passed"]].sort_values("final_score", ascending=False)
    near_df = res_df[~res_df["passed"]].sort_values("final_score", ascending=False)

    tab1, tab2, tab3 = st.tabs([
        f"✅ 入选池 ({len(pass_df)})",
        f"⚠️ 近门槛 ({len(near_df)})",
        "📊 全部结果",
    ])

    show_cols = [
        "code", "name", "industry", "tier", "final_score",
        "moat_score", "margin_score", "price", "pe_ttm", "pb", "mv",
        "tier_reason", "industry_reason", "distress_reversal",
    ]

    def _render(d: pd.DataFrame):
        if d.empty:
            st.info("暂无")
            return
        cols = [c for c in show_cols if c in d.columns]
        st.dataframe(d[cols], width="stretch", height=500,
                     column_config={
                         "final_score": st.column_config.ProgressColumn(
                             "综合分", min_value=0, max_value=100, format="%.1f"),
                         "moat_score": st.column_config.ProgressColumn(
                             "护城河", min_value=0, max_value=100, format="%.0f"),
                         "margin_score": st.column_config.ProgressColumn(
                             "安全边际", min_value=0, max_value=100, format="%.0f"),
                         "mv": st.column_config.NumberColumn("市值(亿)", format="%.0f"),
                         "pe_ttm": st.column_config.NumberColumn("PE", format="%.1f"),
                         "pb": st.column_config.NumberColumn("PB", format="%.2f"),
                         "price": st.column_config.NumberColumn("价格", format="%.2f"),
                     })

    with tab1:
        _render(pass_df)
        if not pass_df.empty:
            st.download_button("⬇️ 下载入选池 CSV",
                               pass_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="screener_pass.csv")
    with tab2:
        st.caption("未入选但分数靠前的，可作为观察名单")
        _render(near_df.head(50))
    with tab3:
        _render(res_df.sort_values("final_score", ascending=False))

    # 详情面板
    st.divider()
    st.subheader("🔬 单只详情")
    if not res_df.empty:
        codes_label = [f"{r['code']} {r['name']} (score={r['final_score']})"
                       for _, r in res_df.sort_values("final_score", ascending=False).iterrows()]
        sel = st.selectbox("选择查看", codes_label)
        if sel:
            sel_code = sel.split()[0]
            r = next(x for x in results if x["code"] == sel_code)
            c1, c2, c3 = st.columns(3)
            c1.metric("综合分", r.get("final_score", 0))
            c2.metric("质量档", r.get("tier", "-"))
            c3.metric("是否入选", "✅ 是" if r.get("passed") else "❌ 否")

            st.markdown(f"- **行业判断**：{r.get('industry_reason','')}")
            st.markdown(f"- **质量理由**：{r.get('tier_reason','')}")
            if r.get("distress_reversal"):
                st.markdown("- 🔥 **困境反转候选**")

            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("**护城河得分构成**")
                st.json(r.get("moat_parts", {}))
            with mc2:
                st.markdown("**安全边际构成**")
                st.json(r.get("margin_parts", {}))

            st.info(f"想做深度估值？复制代码 `{sel_code}` 到 [成长股估值] 页面")
else:
    st.caption("点上面的「开始筛选」按钮启动")
