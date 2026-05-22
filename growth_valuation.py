"""
成长股估值分析工具
框架：反向DCF + 三情景估值 + PEG + FCF + Gemini AI 解读
依赖：pip install streamlit akshare plotly google-generativeai
运行：streamlit run growth_valuation.py
"""

import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import warnings
import traceback
from pathlib import Path

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
# Key 持久化（~/.config/growth_valuation/config.env）
# ════════════════════════════════════════════════════════════

_CONFIG_PATH = Path.home() / ".config" / "growth_valuation" / "config.env"


def load_saved_key() -> str:
    try:
        if _CONFIG_PATH.exists():
            for line in _CONFIG_PATH.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


def save_key(key: str):
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(f"GEMINI_API_KEY={key}\n")
    except Exception:
        pass


def delete_key():
    try:
        if _CONFIG_PATH.exists():
            _CONFIG_PATH.unlink()
    except Exception:
        pass


st.set_page_config(
    page_title="成长股估值分析",
    page_icon="📈",
    layout="wide",
)

# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

def safe_fetch(name, func, retries=3, delay=3):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            result = func()
            if result is not None and (not isinstance(result, pd.DataFrame) or len(result) > 0):
                return result
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(delay * attempt)
    # 把最后一次错误存到 session_state 供诊断
    if last_err is not None:
        errs = st.session_state.get("_fetch_errors", {})
        errs[name] = f"{type(last_err).__name__}: {last_err}"
        st.session_state["_fetch_errors"] = errs
    return None


# ── 多源回退：当东财不可用时，自动切换到雪球/腾讯/新浪 ──

def _xq_symbol(symbol):
    if symbol.startswith(("6", "9")):
        return "SH" + symbol
    if symbol.startswith(("4", "8")):
        return "BJ" + symbol
    return "SZ" + symbol


def _sina_symbol(symbol):
    if symbol.startswith(("6", "9")):
        return "sh" + symbol
    if symbol.startswith(("4", "8")):
        return "bj" + symbol
    return "sz" + symbol


def _stock_info_xq(symbol):
    """雪球 basic + spot 组装成兼容东财 stock_individual_info_em 的 item/value 长表"""
    from datetime import datetime
    sym = _xq_symbol(symbol)
    basic = ak.stock_individual_basic_info_xq(symbol=sym)
    spot = ak.stock_individual_spot_xq(symbol=sym)

    def _get(df, key):
        if df is None or df.empty:
            return None
        v = df.loc[df["item"] == key, "value"].values
        return v[0] if len(v) else None

    rows = []
    for src_key, dst_key in [
        ("现价", "最新"),
        ("基金份额/总股本", "总股本"),
        ("流通股", "流通股"),
    ]:
        v = _get(spot, src_key)
        if v is not None:
            rows.append({"item": dst_key, "value": v})

    name = _get(basic, "org_short_name_cn")
    if name:
        rows.append({"item": "股票简称", "value": name})

    listed = _get(basic, "listed_date")
    if listed is not None:
        try:
            d = datetime.fromtimestamp(int(listed) / 1000).strftime("%Y%m%d")
            rows.append({"item": "上市时间", "value": d})
        except Exception:
            pass

    return pd.DataFrame(rows) if rows else None


def _hist_price_tx(symbol):
    df = ak.stock_zh_a_hist_tx(
        symbol=_sina_symbol(symbol),
        start_date="20200101", end_date="20271231", adjust="qfq",
    )
    return df.rename(columns={"date": "日期", "close": "收盘"}) if df is not None else None


def _hist_price_sina(symbol):
    df = ak.stock_zh_a_daily(
        symbol=_sina_symbol(symbol),
        start_date="20200101", end_date="20271231", adjust="qfq",
    )
    return df.rename(columns={"date": "日期", "close": "收盘"}) if df is not None else None


def fetch_with_fallback(key, sources, retries=2, delay=2):
    """sources=[(source_name, fn), ...]; 按序尝试，第一个非空即返回 (df, source_name)。
    每个源最多 retries 次。所有源失败时，把每个源的最后一次错误写入 _fetch_errors。"""
    for source_name, fn in sources:
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                df = fn()
                if df is not None and (not isinstance(df, pd.DataFrame) or len(df) > 0):
                    return df, source_name
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(delay)
        if last_err is not None:
            errs = st.session_state.get("_fetch_errors", {})
            errs[f"{key}[{source_name}]"] = f"{type(last_err).__name__}: {last_err}"
            st.session_state["_fetch_errors"] = errs
    return None, None


def extract_annual(df, metric, n=6):
    """从 financial_abstract 宽表取最近 n 个年报值，返回 {year: value(元)}"""
    if df is None or df.empty:
        return {}
    annual_cols = sorted(
        [c for c in df.columns if str(c).endswith("1231")],
        reverse=True,
    )
    row = df[df["指标"].astype(str).str.strip() == metric]
    if row.empty:
        return {}
    result = {}
    for col in annual_cols[:n]:
        try:
            val = float(row[col].values[0])
            if not np.isnan(val):
                result[int(str(col)[:4])] = val
        except Exception:
            pass
    return dict(sorted(result.items()))


def extract_annual_from_wide(df, keyword, item_col=None, n=6):
    """从行=科目、列=期间的宽表中按关键词匹配行，返回 {year: value}"""
    if df is None or df.empty:
        return {}
    if item_col is None:
        item_col = df.columns[0]
    year_cols = sorted(
        [c for c in df.columns if str(c).strip()[:4].isdigit()],
        reverse=True,
    )[:n]
    row = df[df[item_col].astype(str).str.contains(keyword, na=False)]
    if row.empty:
        return {}
    result = {}
    for col in year_cols:
        try:
            val = float(str(row[col].values[0]).replace(",", ""))
            if not np.isnan(val):
                result[int(str(col).strip()[:4])] = val
        except Exception:
            pass
    return dict(sorted(result.items()))


def reverse_dcf(current_mv, base_value, terminal_multiple, discount_rate, years):
    future_needed = current_mv * (1 + discount_rate) ** years
    terminal_needed = future_needed / terminal_multiple
    return (terminal_needed / base_value) ** (1 / years) - 1


def scenario_valuation(base_value, cagr, terminal_multiple, discount_rate, years, current_mv):
    future_val = base_value * (1 + cagr) ** years
    terminal_mv = future_val * terminal_multiple
    fair_mv_today = terminal_mv / (1 + discount_rate) ** years
    annual_return = (terminal_mv / current_mv) ** (1 / years) - 1
    return {
        "future_val": future_val,
        "terminal_mv": terminal_mv,
        "fair_mv_today": fair_mv_today,
        "annual_return": annual_return,
    }


def growth_label(g, bear, base, bull):
    if g < bear:
        return "🟢", "市场定价极度悲观，存在明显低估"
    if g < base:
        return "🟡", "市场定价偏保守，有一定安全边际"
    if g < bull:
        return "🟠", "市场定价合理，需增速超预期才有超额收益"
    return "🔴", "市场定价乐观，已超过乐观情景，上行空间有限"


# ════════════════════════════════════════════════════════════
# 数据拉取（主流程中逐步拉取，session_state 缓存当次会话）
# ════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════
# 财务计算
# ════════════════════════════════════════════════════════════

def compute_financials(data, symbol, forecast_profit, bear, base, bull,
                        terminal_pe, discount_rate, forecast_years,
                        prob_bear, prob_base, prob_bull):
    fa = data["financial_abstract"]
    stock_info = data["stock_info"]
    cf_yearly = data["cf_yearly"]
    cf_sina = data["cf_sina"]

    result = {}

    # ── 价格 & 股本 ──
    try:
        result["price"] = float(stock_info[stock_info["item"] == "最新"]["value"].values[0])
        result["shares"] = float(stock_info[stock_info["item"] == "总股本"]["value"].values[0]) / 1e8
        result["mv"] = result["price"] * result["shares"]
        result["industry"] = stock_info[stock_info["item"] == "行业"]["value"].values[0] if "行业" in stock_info["item"].values else "—"
        result["list_date"] = str(stock_info[stock_info["item"] == "上市时间"]["value"].values[0])[:4] if "上市时间" in stock_info["item"].values else "—"
    except Exception:
        result["price"] = result["shares"] = result["mv"] = 0
        result["industry"] = result["list_date"] = "—"

    # ── 历史净利润 ──
    raw_profit = extract_annual(fa, "归母净利润", n=6)
    result["hist_profit"] = {y: v / 1e8 for y, v in raw_profit.items()}

    # ── 历史 OCF ──
    raw_ocf = extract_annual(fa, "经营现金流量净额", n=6)
    result["hist_ocf"] = {y: v / 1e8 for y, v in raw_ocf.items()}

    # ── 历史 Capex（四级降级）──
    hist_capex = {}
    capex_source = "无"

    if cf_yearly is not None:
        _item = cf_yearly.columns[0]
        _cx = extract_annual_from_wide(cf_yearly, "购建固定资产", item_col=_item, n=6)
        if _cx:
            hist_capex = {y: abs(v) / 1e8 for y, v in _cx.items()}
            capex_source = "历年报表（购建固定资产）"

    if not hist_capex and cf_sina is not None:
        _cx2 = extract_annual_from_wide(cf_sina, "购建固定资产", n=6)
        if _cx2:
            hist_capex = {y: abs(v) / 1e8 for y, v in _cx2.items()}
            capex_source = "新浪财经（购建固定资产）"

    if not hist_capex and fa is not None:
        for kw in ["资本支出", "购建固定"]:
            _cx3 = extract_annual(fa, kw, n=6)
            if _cx3:
                hist_capex = {y: abs(v) / 1e8 for y, v in _cx3.items()}
                capex_source = f"财务摘要（{kw}）"
                break

    result["hist_capex"] = hist_capex
    result["capex_source"] = capex_source

    # ── FCF = OCF - Capex ──
    result["hist_fcf"] = {
        y: max(ocf - hist_capex.get(y, 0), 0)
        for y, ocf in result["hist_ocf"].items()
    }

    # ── FCF/净利润 ──
    fcf_margin = {}
    for y in sorted(set(result["hist_profit"]) & set(result["hist_fcf"])):
        p = result["hist_profit"][y]
        if p > 0:
            fcf_margin[y] = result["hist_fcf"][y] / p
    result["fcf_margin"] = fcf_margin
    result["avg_fcf_margin"] = float(np.mean(list(fcf_margin.values()))) if fcf_margin else 1.0

    # ── 派生指标 ──
    hp = result["hist_profit"]
    hv = list(hp.values())
    hy = list(hp.keys())
    result["hist_growth"] = [(hv[i] / hv[i-1] - 1) * 100 for i in range(1, len(hv))]
    result["avg_cagr"] = (hv[-1] / hv[0]) ** (1 / (len(hv) - 1)) - 1 if len(hv) > 1 else 0
    result["hist_years"] = hy
    result["hist_values"] = hv

    mv = result["mv"]
    result["pe_ttm"] = mv / hv[-1] if hv else 0
    result["pe_2026"] = mv / forecast_profit.get(2026, hv[-1] * 1.2) if hv else 0
    result["pe_2027"] = mv / forecast_profit.get(2027, hv[-1] * 1.4) if hv else 0
    result["peg"] = result["pe_2026"] / (base * 100) if base else 0

    # ── 反向 DCF ──
    latest_profit = hv[-1] if hv else 0
    latest_fcf = result["hist_fcf"].get(max(result["hist_fcf"]), latest_profit * result["avg_fcf_margin"]) if result["hist_fcf"] else latest_profit
    result["latest_profit"] = latest_profit
    result["latest_fcf"] = latest_fcf

    result["implied_pe"] = reverse_dcf(mv, latest_profit, terminal_pe, discount_rate, forecast_years) if latest_profit > 0 else float("nan")
    result["implied_fcf"] = reverse_dcf(mv, latest_fcf, terminal_pe, discount_rate, forecast_years) if latest_fcf > 0 else float("nan")

    # ── 三情景 ──
    cagr_map = {"悲观": bear, "基准": base, "乐观": bull}
    weights = {"悲观": prob_bear, "基准": prob_base, "乐观": prob_bull}

    scenarios_pe = {
        n: scenario_valuation(latest_profit, g, terminal_pe, discount_rate, forecast_years, mv)
        for n, g in cagr_map.items()
    }
    weighted_return_pe = sum(scenarios_pe[s]["annual_return"] * weights[s] for s in scenarios_pe)
    weighted_mv_pe = sum(scenarios_pe[s]["fair_mv_today"] * weights[s] for s in scenarios_pe)

    scenarios_fcf = {}
    weighted_return_fcf = weighted_mv_fcf = None
    if latest_fcf > 0:
        scenarios_fcf = {
            n: scenario_valuation(latest_fcf, g, terminal_pe, discount_rate, forecast_years, mv)
            for n, g in cagr_map.items()
        }
        weighted_return_fcf = sum(scenarios_fcf[s]["annual_return"] * weights[s] for s in scenarios_fcf)
        weighted_mv_fcf = sum(scenarios_fcf[s]["fair_mv_today"] * weights[s] for s in scenarios_fcf)

    result["cagr_map"] = cagr_map
    result["weights"] = weights
    result["scenarios_pe"] = scenarios_pe
    result["weighted_return_pe"] = weighted_return_pe
    result["weighted_mv_pe"] = weighted_mv_pe
    result["scenarios_fcf"] = scenarios_fcf
    result["weighted_return_fcf"] = weighted_return_fcf
    result["weighted_mv_fcf"] = weighted_mv_fcf

    # 决策
    wr = weighted_return_pe
    if wr >= 0.18:
        result["decision"] = "强烈推荐建仓，赔率极佳"
        result["dsymbol"] = "🟢🟢"
    elif wr >= 0.12:
        result["decision"] = "可以建仓，赔率合理"
        result["dsymbol"] = "🟢"
    elif wr >= 0.08:
        result["decision"] = "观望为主，等待更好买入点"
        result["dsymbol"] = "🟡"
    else:
        result["decision"] = "不建议买入，赔率不足"
        result["dsymbol"] = "🔴"

    return result


# ════════════════════════════════════════════════════════════
# Gemini AI 分析（三个独立模块）
# ════════════════════════════════════════════════════════════

def _build_model(api_key: str, grounding: bool = False):
    """初始化 Gemini 模型，grounding=True 时启用 Google Search"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = None
    if grounding:
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    return {"client": client, "model": "gemini-2.5-flash", "config": config}


def _call(model, prompt: str) -> str:
    try:
        resp = model["client"].models.generate_content(
            model=model["model"],
            contents=prompt,
            config=model["config"],
        )
        text = resp.text
        try:
            chunks = resp.candidates[0].grounding_metadata.grounding_chunks
            if chunks:
                sources = "\n".join(
                    f"- [{c.web.title}]({c.web.uri})"
                    for c in chunks if hasattr(c, "web")
                )
                text += f"\n\n---\n**信息来源**\n{sources}"
        except Exception:
            pass
        return text
    except Exception as e:
        return f"❌ 调用失败：{e}"


def _data_summary(r, forecast_profit, bear, base, bull,
                  terminal_pe, discount_rate, forecast_years) -> str:
    """生成传入 prompt 的量化数据摘要（三个模块共用）"""
    fcf_str = ", ".join(f"{y}年:{v:.2f}" for y, v in r["fcf_margin"].items()) or "数据不足"
    profit_str = ", ".join(f"{y}年:{v:.2f}亿" for y, v in r["hist_profit"].items())
    growth_str = ", ".join(f"{r['hist_years'][i+1]}年:{g:+.1f}%" for i, g in enumerate(r["hist_growth"]))
    fp_str = ", ".join(f"{y}E:{v:.2f}亿" for y, v in forecast_profit.items())
    return f"""
股票：{r.get('_name','该公司')}，行业：{r['industry']}，上市：{r['list_date']}年
当前股价：{r['price']:.2f}元，市值：{r['mv']:.1f}亿元
PE(TTM)：{r['pe_ttm']:.1f}x | 2026E PE：{r['pe_2026']:.1f}x | PEG：{r['peg']:.2f}
历史净利润：{profit_str}
历史增速：{growth_str}  历史CAGR：{r['avg_cagr']*100:.1f}%
机构预测：{fp_str}
FCF/净利润：{fcf_str}（均值：{r['avg_fcf_margin']:.2f}）
最新净利润：{r['latest_profit']:.2f}亿，FCF：{r['latest_fcf']:.2f}亿
反向DCF隐含增速（净利润）：{r['implied_pe']*100:.1f}%  （FCF）：{r['implied_fcf']*100:.1f}%
三情景加权年化回报：{r['weighted_return_pe']*100:.1f}%，加权合理市值：{r['weighted_mv_pe']:.0f}亿
悲观{bear*100:.0f}%→{r['scenarios_pe']['悲观']['annual_return']*100:+.1f}% | 基准{base*100:.0f}%→{r['scenarios_pe']['基准']['annual_return']*100:+.1f}% | 乐观{bull*100:.0f}%→{r['scenarios_pe']['乐观']['annual_return']*100:+.1f}%
折现率：{discount_rate*100:.0f}%，预测年限：{forecast_years}年，成熟期PE：{terminal_pe}x"""


# ── 模块一：估值解读 ──
def ai_valuation(api_key, stock_name, symbol, r, forecast_profit,
                 bear, base, bull, terminal_pe, discount_rate, forecast_years):
    try:
        model = _build_model(api_key, grounding=False)
    except Exception as e:
        return f"❌ Gemini 初始化失败：{e}"

    r["_name"] = stock_name
    data = _data_summary(r, forecast_profit, bear, base, bull,
                         terminal_pe, discount_rate, forecast_years)
    prompt = f"""你是一位专业的A股价值投资分析师。请基于以下量化数据对{stock_name}（{symbol}）进行估值解读。

## 量化数据
{data}

## 请完成（每项3-5句，专业但不晦涩，避免废话）

### 1. 估值水位
当前PE/PEG处于什么水位？市场隐含增速与你的基准假设相比是高还是低？给出明确判断。

### 2. 利润质量
FCF/净利润比反映了什么？结合行业特性（重资产扩张 vs 轻资产）解读这个数字的意义。

### 3. 增速假设合理性
历史CAGR、机构预测、基准假设{base*100:.0f}%三者是否一致？哪个最可信？

### 4. 赔率判断
加权年化{r['weighted_return_pe']*100:.1f}%在当前无风险利率环境下是否有吸引力？胜率和赔率如何？

### 5. 一句话结论
现在这个价位，值得买吗？

⚠️ 本分析仅供参考，不构成投资建议。"""
    return _call(model, prompt)


# ── 模块二：护城河与质地分析 ──
def ai_moat(api_key, stock_name, symbol, r):
    try:
        model = _build_model(api_key, grounding=False)
    except Exception as e:
        return f"❌ Gemini 初始化失败：{e}"

    prompt = f"""你是一位专注于A股基本面研究的分析师，擅长分析公司竞争优势和商业模式质量。

请对{stock_name}（{symbol}，{r['industry']}行业）进行深度护城河分析。

## 参考财务特征
- 历史净利润CAGR：{r['avg_cagr']*100:.1f}%
- FCF/净利润均值：{r['avg_fcf_margin']:.2f}
- ROE参考（从PE和增速推断）：中高水平
- 上市年份：{r['list_date']}，经历完整行业周期

## 请从以下维度分析（每项3-5句，要有具体依据，不要泛泛而谈）

### 1. 商业模式
这家公司怎么赚钱？收入结构是什么？有没有重复消费/订阅属性？

### 2. 核心护城河
从品牌、规模效应、网络效应、转换成本、牌照壁垒五个维度，这家公司的护城河宽度如何？哪条最强？

### 3. 行业竞争格局
行业集中度如何？主要竞争对手是谁？这家公司的市场地位是在强化还是被侵蚀？

### 4. 管理层与治理
创始人/管理层背景、股权结构、资本配置历史，有没有让你担心的地方？

### 5. 长期风险
未来5-10年最可能摧毁这家公司护城河的因素是什么？政策、技术替代、竞争、人口变化？

### 6. 质地打分
综合以上，给这家公司的生意质地打分（1-10分）并说明理由。

⚠️ 本分析基于公开信息和AI推断，仅供参考。"""
    return _call(model, prompt)


# ── 模块三：近期新闻与催化剂（启用 Google Search grounding）──
def ai_news(api_key, stock_name, symbol, r):
    try:
        model = _build_model(api_key, grounding=True)
    except Exception as e:
        return f"❌ Gemini 初始化失败：{e}"

    prompt = f"""请使用 Google Search 搜索{stock_name}（A股代码{symbol}）最近1个月的重要新闻和公告。

重点搜索以下内容：
1. 最新财报/业绩预告/业绩快报
2. 重大合同、并购、战略合作公告
3. 管理层变动
4. 监管政策变化（医疗/行业相关）
5. 机构调研、评级调整
6. 负面舆情（医疗事故、诉讼、处罚等）

## 行业背景
{r['industry']}行业，市值{r['mv']:.0f}亿元

## 请输出

### 近期重要事件（按时间倒序，列出5-8条）
每条格式：**[日期] 事件标题** — 1-2句说明对股价/基本面的影响

### 催化剂评估
近期有没有值得关注的正面催化剂或负面风险？对估值的影响方向是什么？

### 信息差提示
有没有市场可能尚未充分定价的信息？

⚠️ 新闻来自 Google Search 实时搜索，请注意核实原始来源。"""
    return _call(model, prompt)


# ── 模块四：AI 建议增速（结构化输出）──
def ai_suggest_cagr(api_key, stock_name, symbol, hist_profit, hist_cagr,
                    forecast_profit, industry) -> dict:
    """
    返回 {"bear": float, "base": float, "bull": float, "reason": str}
    失败时返回 None
    """
    try:
        model = _build_model(api_key, grounding=False)
    except Exception:
        return None

    profit_str = ", ".join(f"{y}:{v:.2f}亿" for y, v in hist_profit.items())
    fp_str = ", ".join(f"{y}E:{v:.2f}亿" for y, v in forecast_profit.items())

    prompt = f"""你是一位专业的A股研究员，请为{stock_name}（{symbol}，{industry}行业）给出三情景利润增速建议。

## 已知数据
- 历史净利润：{profit_str}
- 历史CAGR：{hist_cagr*100:.1f}%
- 机构一致预期：{fp_str}

## 任务
基于历史增速、行业景气度、机构预期，给出未来{len(forecast_profit)+2}年的三情景年化利润增速建议。

## 严格按以下 JSON 格式输出，不要输出任何其他内容：
{{
  "bear": <悲观增速，0到1之间的小数，如0.08>,
  "base": <基准增速，0到1之间的小数，如0.18>,
  "bull": <乐观增速，0到1之间的小数，如0.28>,
  "reason": "<50字以内的中文理由，说明三个数字的依据>"
}}"""

    raw = _call(model, prompt)
    try:
        import json, re
        # 容错：提取第一个 JSON 对象
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "bear": float(data["bear"]),
                "base": float(data["base"]),
                "bull": float(data["bull"]),
                "reason": str(data.get("reason", "")),
            }
    except Exception:
        pass
    return None


# ════════════════════════════════════════════════════════════
# 图表
# ════════════════════════════════════════════════════════════

def plot_heatmap(base_profit, cagr_range, pe_range, current_mv, forecast_years, base_cagr, terminal_pe):
    z = []
    for pe in pe_range:
        row = []
        for g in cagr_range:
            term_mv = base_profit * (1 + g) ** forecast_years * pe
            ann_ret = (term_mv / current_mv) ** (1 / forecast_years) - 1
            row.append(round(ann_ret * 100, 1))
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"{g*100:.0f}%" for g in cagr_range],
        y=[f"{pe}x" for pe in pe_range],
        colorscale=[
            [0.0, "#d73027"], [0.3, "#f46d43"],
            [0.5, "#fee08b"], [0.65, "#a6d96a"],
            [0.8, "#1a9850"], [1.0, "#006837"],
        ],
        zmid=12,
        text=[[f"{v}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 11},
        colorbar=dict(title="年化回报%"),
    ))

    bx = int(min(range(len(cagr_range)), key=lambda i: abs(cagr_range[i] - base_cagr)))
    by = pe_range.index(terminal_pe) if terminal_pe in pe_range else None
    if by is not None:
        fig.add_shape(
            type="rect",
            x0=bx - 0.5, x1=bx + 0.5,
            y0=by - 0.5, y1=by + 0.5,
            line=dict(color="blue", width=3),
        )

    fig.update_layout(
        xaxis_title="利润年化增速（CAGR）",
        yaxis_title="成熟期终值PE",
        height=380,
        margin=dict(t=20),
    )
    return fig


def plot_profit_trend(hist_profit, forecast_profit, hist_fcf, hist_ocf,
                      bear, base, bull, forecast_years):
    last_hist = max(hist_profit)
    last_forecast = max(forecast_profit)
    pivot = forecast_profit[last_forecast]
    ext_count = forecast_years - len(forecast_profit)
    ext_years = list(range(last_forecast + 1, last_forecast + ext_count + 1))

    fig = go.Figure()

    # 历史利润
    fig.add_trace(go.Scatter(
        x=list(hist_profit.keys()), y=list(hist_profit.values()),
        mode="lines+markers+text", name="历史净利润",
        text=[f"{v:.1f}亿" for v in hist_profit.values()],
        textposition="top center",
        line=dict(color="#2196F3", width=2.5), marker=dict(size=8),
    ))

    # 历史 OCF
    common_years = sorted(set(hist_ocf) & set(hist_profit))
    if common_years:
        fig.add_trace(go.Scatter(
            x=common_years, y=[hist_ocf[y] for y in common_years],
            mode="lines+markers", name="经营现金流",
            line=dict(color="#9C27B0", width=1.5, dash="dot"), marker=dict(size=6),
        ))

    # 历史 FCF
    if hist_fcf:
        fcf_years = sorted(hist_fcf)
        fig.add_trace(go.Scatter(
            x=fcf_years, y=[hist_fcf[y] for y in fcf_years],
            mode="lines+markers", name="FCF",
            line=dict(color="#FF9800", width=1.5, dash="dot"), marker=dict(size=6, symbol="diamond"),
        ))

    # 机构预测
    fig.add_trace(go.Scatter(
        x=[last_hist] + list(forecast_profit.keys()),
        y=[hist_profit[last_hist]] + list(forecast_profit.values()),
        mode="lines+markers+text", name="机构预测",
        text=[None] + [f"{v:.1f}亿" for v in forecast_profit.values()],
        textposition="top center",
        line=dict(color="#2196F3", width=2, dash="dash"), marker=dict(size=7, symbol="diamond"),
    ))

    # 三情景延伸
    for label, cagr_val, color in [("悲观", bear, "#FF5252"), ("基准", base, "#4CAF50"), ("乐观", bull, "#FF9800")]:
        ext_profits = [pivot * (1 + cagr_val) ** i for i in range(1, ext_count + 1)]
        fig.add_trace(go.Scatter(
            x=[last_forecast] + ext_years, y=[pivot] + ext_profits,
            mode="lines", name=f"{label}（{cagr_val*100:.0f}%）",
            line=dict(color=color, width=1.5, dash="dot"),
        ))

    fig.update_layout(
        xaxis_title="年份", yaxis_title="金额（亿元）",
        height=420, legend=dict(x=0.02, y=0.98),
        margin=dict(t=20),
    )
    return fig


def plot_price_history(hist_price, stock_name):
    if hist_price is None or hist_price.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_price["日期"], y=hist_price["收盘"],
        mode="lines", name="收盘价（前复权）",
        line=dict(color="#2196F3", width=1.5),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.08)",
    ))
    fig.update_layout(
        xaxis_title="日期", yaxis_title="股价（元）",
        height=300, margin=dict(t=20),
    )
    return fig


# ════════════════════════════════════════════════════════════
# 侧边栏
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ 参数设置")

    st.subheader("📌 标的")
    symbol = st.text_input("股票代码", value="300015", max_chars=6).strip()

    # 检测 symbol 变化，自动清空上一只股票的状态
    if st.session_state.get("_last_symbol") != symbol:
        for key in ["ai_valuation", "ai_moat", "ai_news",
                    "cagr_reason", "bear_slider", "base_slider", "bull_slider",
                    "_analysis_started"]:
            st.session_state.pop(key, None)
        # 清空上一只股票的数据缓存
        old_sym = st.session_state.get("_last_symbol", "")
        st.session_state.pop(f"raw_data_{old_sym}", None)
        st.session_state["_last_symbol"] = symbol

    # 股票名称：自动填充（分析后显示），也可手动修改
    _auto_name = st.session_state.get("_auto_name", "") if st.session_state.get("_auto_name_symbol") == symbol else ""
    stock_name = st.text_input(
        "股票名称",
        value=_auto_name,
        placeholder="点击「开始分析」后自动填充",
        help="自动从 AKShare 读取，也可手动修改",
    )

    st.subheader("📋 机构预测利润（亿元）")
    fp_2026 = st.number_input("2026E", value=14.46, step=0.1)
    fp_2027 = st.number_input("2027E", value=16.82, step=0.1)
    fp_2028 = st.number_input("2028E", value=19.43, step=0.1)
    forecast_profit = {2026: fp_2026, 2027: fp_2027, 2028: fp_2028}

    st.subheader("📐 增速假设")

    # 初始化默认值（只在首次运行时设置，之后由滑块自己维护）
    # 如果上一轮 AI 建议写入了待应用值，在 widget 创建前应用
    for _src, _dst in [("_pending_bear", "bear_slider"),
                       ("_pending_base", "base_slider"),
                       ("_pending_bull", "bull_slider")]:
        if _src in st.session_state:
            st.session_state[_dst] = st.session_state.pop(_src)

    if "bear_slider" not in st.session_state:
        st.session_state["bear_slider"] = 8
    if "base_slider" not in st.session_state:
        st.session_state["base_slider"] = 20
    if "bull_slider" not in st.session_state:
        st.session_state["bull_slider"] = 32
    if "cagr_reason" not in st.session_state:
        st.session_state["cagr_reason"] = ""

    st.button("🤖 AI 建议增速", key="suggest_btn",
              help="先点「开始分析」拉取数据，再点此按钮获取 AI 建议")

    bear_cagr = st.slider("悲观 CAGR", 0, 50, key="bear_slider") / 100
    base_cagr = st.slider("基准 CAGR", 0, 50, key="base_slider") / 100
    bull_cagr = st.slider("乐观 CAGR", 0, 60, key="bull_slider") / 100

    if st.session_state["cagr_reason"]:
        st.caption(f"🤖 {st.session_state['cagr_reason']}")

    st.subheader("🧮 估值假设")
    terminal_pe = st.number_input("成熟期 PE", value=20, step=1)
    discount_rate = st.slider("折现率", 8, 20, 12) / 100
    forecast_years = st.slider("预测年限", 3, 10, 5)

    st.subheader("⚖️ 情景概率")
    prob_bear = st.slider("悲观概率", 0, 100, 20) / 100
    prob_base = st.slider("基准概率", 0, 100, 55) / 100
    prob_bull = st.slider("乐观概率", 0, 100, 25) / 100

    st.caption(f"概率合计：{(prob_bear+prob_base+prob_bull)*100:.0f}%")
    if abs(prob_bear + prob_base + prob_bull - 1.0) > 0.01:
        st.warning("概率之和须等于 100%")

    st.subheader("🤖 Gemini AI 解读")

    _saved_key = load_saved_key()
    _key_source = "已从本地配置读取" if _saved_key else ""

    gemini_key = st.text_input(
        "Gemini API Key",
        value=_saved_key,
        type="password",
        placeholder="AIza...",
        help="免费获取：https://aistudio.google.com",
    )

    # Key 变化时自动保存或清除
    if gemini_key and gemini_key != _saved_key:
        save_key(gemini_key)
        st.caption("✅ Key 已保存到本地，下次启动自动读取")
    elif not gemini_key and _saved_key:
        delete_key()
        st.caption("🗑️ 已清除本地保存的 Key")
    elif _key_source:
        st.caption(f"✅ {_key_source}")

    # 缓存 key 供侧边栏 AI 建议按钮使用
    st.session_state["_gemini_key_cache"] = gemini_key

    enable_ai = st.checkbox("启用 AI 分析", value=bool(gemini_key))

    st.divider()
    run_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

# ════════════════════════════════════════════════════════════
# 主界面
# ════════════════════════════════════════════════════════════

st.title(f"📈 成长股估值分析")
st.caption("框架：反向DCF + 三情景 + FCF + PEG | 数据源：AKShare（东方财富/新浪）")

if run_btn:
    st.session_state["_analysis_started"] = True

if not st.session_state.get("_analysis_started"):
    st.info("👈 在左侧填入股票代码和参数，点击「开始分析」")
    st.markdown("""
**使用说明**

| 必填参数 | 说明 | 来源 |
|---------|------|------|
| 股票代码 | 6位数字，不带市场前缀 | — |
| 机构预测利润 | 未来3年一致预期 | 同花顺 F10 → 盈利预测 |
| 悲观/基准/乐观 CAGR | 你对公司的增速判断 | 你的研究 |
| 成熟期 PE | 行业参考：制造15-20，消费20-25，科技25-30 | 行业经验 |

**价格/股本/历史利润/OCF/Capex 全部自动拉取，无需手动填写。**
    """)
    st.stop()

# ── 拉取数据 ──
_cache_key = f"raw_data_{symbol}"
if _cache_key in st.session_state:
    # 命中 session 缓存，直接用
    raw_data = st.session_state[_cache_key]
else:
    # 真正拉取，逐步显示进度。每个 key 配置多源回退（东财 → 雪球/腾讯/新浪）。
    _plan = [
        ("财务摘要（历史利润/OCF）", "financial_abstract", [
            ("新浪", lambda: ak.stock_financial_abstract(symbol=symbol)),
        ]),
        ("实时价格与股本", "stock_info", [
            ("东财", lambda: ak.stock_individual_info_em(symbol=symbol)),
            ("雪球", lambda: _stock_info_xq(symbol)),
        ]),
        ("历史股价行情", "hist_price", [
            ("东财", lambda: ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date="20200101", end_date="20271231", adjust="qfq",
            )),
            ("腾讯", lambda: _hist_price_tx(symbol)),
            ("新浪", lambda: _hist_price_sina(symbol)),
        ]),
        ("历年资本开支", "cf_yearly", [
            ("东财", lambda: ak.stock_cash_flow_sheet_by_yearly_em(symbol=symbol)),
        ]),
        ("新浪历年现金流", "cf_sina", [
            ("新浪", lambda: ak.stock_financial_report_sina(
                stock=_sina_symbol(symbol), symbol="现金流量表",
            )),
        ]),
    ]

    st.markdown(f"#### ⏳ 正在拉取 **{symbol}** 数据...")
    _bar  = st.progress(0)
    _stat = st.empty()
    _results = {}
    _sources = {}

    for i, (label, key, sources) in enumerate(_plan):
        _stat.caption(f"正在拉取：{label}...")
        _bar.progress(int((i + 0.5) / len(_plan) * 100))
        df, src = fetch_with_fallback(key, sources)
        _results[key] = df
        _sources[key] = src
        _bar.progress(int((i + 1) / len(_plan) * 100))

    _results["_sources"] = _sources

    # 提取股票名称
    _results["stock_name_auto"] = ""
    try:
        si = _results["stock_info"]
        if si is not None:
            name_row = si[si["item"] == "股票简称"]
            if not name_row.empty:
                _results["stock_name_auto"] = str(name_row["value"].values[0])
    except Exception:
        pass

    _stat.empty()
    _bar.empty()

    raw_data = _results
    st.session_state[_cache_key] = raw_data   # 存入 session 缓存

if raw_data["financial_abstract"] is None or raw_data["stock_info"] is None:
    st.error(f"数据拉取失败，请检查股票代码是否正确，或稍后重试。")
    st.markdown("**各接口状态：**")
    sources = raw_data.get("_sources", {})
    for k, v in raw_data.items():
        if k in ("stock_name_auto", "_sources"):
            continue
        if v is None:
            st.markdown(f"- ❌ `{k}`")
        else:
            try:
                src = sources.get(k, "?")
                st.markdown(f"- ✅ `{k}`（{len(v)} 行，来源：{src}）")
            except Exception:
                st.markdown(f"- ✅ `{k}`")
    fetch_errors = st.session_state.get("_fetch_errors", {})
    if fetch_errors:
        st.markdown("**错误详情：**")
        for name, err in fetch_errors.items():
            st.code(f"{name}: {err}")
    st.markdown("**常见原因：**\n- 股票代码输入有误（需6位纯数字，如 `300015`）\n- 网络问题或 AKShare 接口临时故障，稍后重试\n- 部分接口需要 A 股交易时段才能访问")
    st.stop()

# ── 股票名称：自动 > 用户手动输入 ──
_auto_name_from_data = raw_data.get("stock_name_auto", "")
if _auto_name_from_data:
    # 如果用户没有手动填名称，或名称还是上一只股票的，用自动名称
    if not stock_name or st.session_state.get("_auto_name_symbol") != symbol:
        stock_name = _auto_name_from_data
    st.session_state["_auto_name"] = _auto_name_from_data
    st.session_state["_auto_name_symbol"] = symbol

# ── 计算 ──
with st.spinner("计算中..."):
    r = compute_financials(
        raw_data, symbol, forecast_profit,
        bear_cagr, base_cagr, bull_cagr,
        terminal_pe, discount_rate, forecast_years,
        prob_bear, prob_base, prob_bull,
    )

# ── 顶部摘要卡片 ──
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("当前股价", f"¥{r['price']:.2f}")
col2.metric("总市值", f"{r['mv']:.0f} 亿")
col3.metric("PE(TTM)", f"{r['pe_ttm']:.1f}x")
col4.metric("2026E PE", f"{r['pe_2026']:.1f}x")
col5.metric("PEG", f"{r['peg']:.2f}", delta=f"{'低估' if r['peg']<1 else '合理' if r['peg']<1.5 else '偏贵'}")

# ── AI 建议增速（数据加载后处理，直接写入滑块 session_state key）──
if st.session_state.get("suggest_btn"):
    if not gemini_key:
        st.warning("请先在侧边栏填入 Gemini API Key")
    else:
        with st.spinner("AI 推算增速中..."):
            _sug = ai_suggest_cagr(
                gemini_key, stock_name, symbol,
                r["hist_profit"], r["avg_cagr"],
                forecast_profit, r["industry"],
            )
        if _sug:
            # widget 已实例化，不能直接写 bear_slider；存到 _pending_*，下次 rerun 顶部应用
            st.session_state["_pending_bear"] = int(round(_sug["bear"] * 100))
            st.session_state["_pending_base"] = int(round(_sug["base"] * 100))
            st.session_state["_pending_bull"] = int(round(_sug["bull"] * 100))
            st.session_state["cagr_reason"] = _sug["reason"]
            st.rerun()
        else:
            st.warning("AI 增速建议失败，请手动设置或稍后重试")

st.divider()

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs(["📊 数据总览", "📐 估值分析", "🌡️ 敏感性", "🤖 AI 解读"])

# ─────────────────────────────────────
# Tab 1：数据总览
# ─────────────────────────────────────
with tab1:
    st.subheader(f"{stock_name}（{symbol}）基础数据")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**公司信息**")
        info_df = pd.DataFrame({
            "项目": ["股票代码", "行业", "上市年份", "总股本"],
            "数值": [symbol, r["industry"], r["list_date"], f"{r['shares']:.2f} 亿股"],
        })
        st.dataframe(info_df, hide_index=True, use_container_width=True)

        st.markdown("**历史股价走势**")
        price_fig = plot_price_history(raw_data["hist_price"], stock_name)
        if price_fig:
            st.plotly_chart(price_fig, use_container_width=True)

    with c2:
        st.markdown("**历史财务数据（亿元）**")
        all_years = sorted(set(r["hist_profit"]) | set(r["hist_ocf"]) | set(r["hist_fcf"]))
        fin_rows = []
        for y in all_years:
            p = r["hist_profit"].get(y)
            ocf = r["hist_ocf"].get(y)
            fcf = r["hist_fcf"].get(y)
            cap = r["hist_capex"].get(y)
            fm = r["fcf_margin"].get(y)
            fin_rows.append({
                "年份": y,
                "归母净利润": f"{p:.2f}" if p else "—",
                "经营现金流": f"{ocf:.2f}" if ocf else "—",
                "资本开支": f"{cap:.2f}" if cap else "—",
                "FCF": f"{fcf:.2f}" if fcf is not None else "—",
                "FCF/净利润": f"{fm:.2f}" if fm else "—",
            })
        fin_df = pd.DataFrame(fin_rows)
        st.dataframe(fin_df, hide_index=True, use_container_width=True)
        st.caption(f"Capex 数据来源：{r['capex_source']}")

        # 利润 vs OCF vs FCF 趋势图
        st.markdown("**利润与现金流趋势**")
        trend_fig = plot_profit_trend(
            r["hist_profit"], forecast_profit,
            r["hist_fcf"], r["hist_ocf"],
            bear_cagr, base_cagr, bull_cagr, forecast_years,
        )
        st.plotly_chart(trend_fig, use_container_width=True)


# ─────────────────────────────────────
# Tab 2：估值分析
# ─────────────────────────────────────
with tab2:
    st.subheader("反向 DCF：市场隐含增速")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**净利润路径**")
        if not np.isnan(r["implied_pe"]):
            em, label = growth_label(r["implied_pe"], bear_cagr, base_cagr, bull_cagr)
            st.metric("隐含增速", f"{r['implied_pe']*100:.1f}%/年")
            st.write(f"{em} {label}")
        else:
            st.write("数据不足")

    with c2:
        st.markdown("**FCF 路径**")
        if not np.isnan(r["implied_fcf"]):
            em2, label2 = growth_label(r["implied_fcf"], bear_cagr, base_cagr, bull_cagr)
            st.metric("隐含增速", f"{r['implied_fcf']*100:.1f}%/年")
            st.write(f"{em2} {label2}")
            diff = r["implied_fcf"] - r["implied_pe"]
            if not np.isnan(diff):
                if diff > 0:
                    st.caption("FCF隐含增速 > 净利润：市场对现金流要求更高，capex 压力受关注")
                else:
                    st.caption("FCF隐含增速 < 净利润：FCF优于净利润，现金质量好")
        else:
            st.write("FCF 数据不足")

    st.markdown(f"参考区间：悲观 **{bear_cagr*100:.0f}%** | 基准 **{base_cagr*100:.0f}%** | 乐观 **{bull_cagr*100:.0f}%** | 历史CAGR **{r['avg_cagr']*100:.1f}%**")

    st.divider()
    st.subheader("三情景正向估值")

    for path_name, scenarios, weighted_return, weighted_mv in [
        ("净利润路径（PE终值）", r["scenarios_pe"], r["weighted_return_pe"], r["weighted_mv_pe"]),
        ("FCF路径（P/FCF终值）", r["scenarios_fcf"], r["weighted_return_fcf"], r["weighted_mv_fcf"]),
    ]:
        if not scenarios:
            continue
        st.markdown(f"**{path_name}**")
        rows = []
        prob_map = {"悲观": prob_bear, "基准": prob_base, "乐观": prob_bull}
        cagr_disp = {"悲观": bear_cagr, "基准": base_cagr, "乐观": bull_cagr}
        for name, s in scenarios.items():
            rows.append({
                "情景": name,
                "增速": f"{cagr_disp[name]*100:.0f}%",
                "概率": f"{prob_map[name]*100:.0f}%",
                f"{forecast_years}年后值（亿）": f"{s['future_val']:.1f}",
                "终值市值（亿）": f"{s['terminal_mv']:.1f}",
                "折现合理市值（亿）": f"{s['fair_mv_today']:.1f}",
                "年化回报": f"{s['annual_return']*100:+.1f}%",
            })
        rows.append({
            "情景": "加权期望",
            "增速": "—",
            "概率": "—",
            f"{forecast_years}年后值（亿）": "—",
            "终值市值（亿）": "—",
            "折现合理市值（亿）": f"{weighted_mv:.1f}",
            "年化回报": f"{weighted_return*100:+.1f}%",
        })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        updown = (weighted_mv / r["mv"] - 1) * 100
        st.caption(f"当前市值 {r['mv']:.1f}亿 → 加权合理市值 {weighted_mv:.1f}亿（{updown:+.1f}%）")

    st.divider()
    st.subheader("合理买入价（基准情景，净利润路径）")
    base_terminal_mv = r["scenarios_pe"]["基准"]["terminal_mv"]
    buy_rows = []
    for target_r in [0.10, 0.12, 0.15, 0.20, 0.25]:
        buy_mv = base_terminal_mv / (1 + target_r) ** forecast_years
        buy_price = buy_mv / r["shares"]
        disc = (buy_price / r["price"] - 1) * 100
        buy_rows.append({
            "目标年化回报": f"{target_r*100:.0f}%",
            "对应买入市值（亿）": f"{buy_mv:.1f}",
            "对应买入价（元）": f"{buy_price:.2f}",
            "较当前折价": f"{disc:+.1f}%",
            "": "← 当前在此区间" if abs(disc) < 5 else "",
        })
    st.dataframe(pd.DataFrame(buy_rows), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader(f"{r['dsymbol']} 综合决策")
    st.markdown(f"### {r['decision']}")
    st.markdown(f"加权年化预期回报（净利润路径）：**{r['weighted_return_pe']*100:.1f}%**")
    if r["weighted_return_fcf"] is not None:
        st.markdown(f"加权年化预期回报（FCF路径）：**{r['weighted_return_fcf']*100:.1f}%**")
    st.warning("⚠️ 本工具仅做辅助分析，不构成投资建议")


# ─────────────────────────────────────
# Tab 3：敏感性热力图
# ─────────────────────────────────────
with tab3:
    st.subheader("增速 × 终值PE 敏感性分析")
    st.caption("每个格子 = 对应假设下买入当前市值的年化回报率。蓝框 = 当前基准假设。")

    cagr_range = np.arange(0.05, 0.45, 0.05)
    pe_range_list = [12, 15, 18, 20, 25, 30]

    heatmap_fig = plot_heatmap(
        r["latest_profit"], cagr_range, pe_range_list,
        r["mv"], forecast_years, base_cagr, terminal_pe,
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)
    st.caption("绿色（>15%）= 值得买入；黄色（8-15%）= 勉强；红色（<8%）= 不划算")

    if r["latest_fcf"] > 0:
        st.subheader("FCF 路径敏感性")
        heatmap_fcf = plot_heatmap(
            r["latest_fcf"], cagr_range, pe_range_list,
            r["mv"], forecast_years, base_cagr, terminal_pe,
        )
        st.plotly_chart(heatmap_fcf, use_container_width=True)


# ─────────────────────────────────────
# Tab 4：AI 解读
# ─────────────────────────────────────
with tab4:
    st.subheader("🤖 Gemini AI 投研解读")

    if not enable_ai or not gemini_key:
        st.info(
            "在左侧侧边栏填入 Gemini API Key 并勾选「启用 AI 分析」，即可获得 AI 投研解读。\n\n"
            "免费 Key 获取：https://aistudio.google.com"
        )
    else:
        ai_tab1, ai_tab2, ai_tab3 = st.tabs(["📊 估值解读", "🏰 护城河分析", "📰 近期新闻"])

        with ai_tab1:
            st.caption("基于量化数据的估值水位、利润质量、增速假设与赔率判断")
            if st.button("生成估值解读", key="btn_valuation"):
                with st.spinner("分析中..."):
                    out = ai_valuation(
                        gemini_key, stock_name, symbol, r, forecast_profit,
                        bear_cagr, base_cagr, bull_cagr,
                        terminal_pe, discount_rate, forecast_years,
                    )
                st.session_state["ai_valuation"] = out
            if "ai_valuation" in st.session_state:
                st.markdown(st.session_state["ai_valuation"])

        with ai_tab2:
            st.caption("商业模式、护城河宽度、竞争格局、管理层质量、长期风险（基于AI知识库）")
            if st.button("生成护城河分析", key="btn_moat"):
                with st.spinner("分析中..."):
                    out = ai_moat(gemini_key, stock_name, symbol, r)
                st.session_state["ai_moat"] = out
            if "ai_moat" in st.session_state:
                st.markdown(st.session_state["ai_moat"])

        with ai_tab3:
            st.caption("通过 Google Search 实时抓取近1个月重要新闻、公告、机构评级变化")
            if st.button("抓取近期新闻", key="btn_news"):
                with st.spinner("正在搜索最新资讯..."):
                    out = ai_news(gemini_key, stock_name, symbol, r)
                st.session_state["ai_news"] = out
            if "ai_news" in st.session_state:
                st.markdown(st.session_state["ai_news"])

        st.divider()
        st.caption("⚠️ AI 内容仅供参考，不构成投资建议。新闻来自 Google Search，请核实原始来源。")
