"""
股票筛选器 - 四层漏斗逻辑
1. 质量门槛（ROE 三档 + 现金流 + 负债）
2. 行业判断（白名单 / 龙头）
3. 护城河 & 趋势打分
4. 安全边际（PE 分位 + DCF 简化版）
最终输出综合评分。
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Literal

import yaml

WHITELIST_PATH = Path(__file__).parent / "industry_whitelist.yaml"

Tier = Literal["A", "B", "C", "X"]
StrictMode = Literal["strict", "standard", "loose"]


def load_whitelist() -> dict:
    if not WHITELIST_PATH.exists():
        return {"rising": [], "stable": [], "blacklist": []}
    with open(WHITELIST_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {
        "rising": data.get("rising", []) or [],
        "stable": data.get("stable", []) or [],
        "blacklist": data.get("blacklist", []) or [],
    }


# ────────────────────────────────────────────────────────────
# 第 1 层：质量门槛
# ────────────────────────────────────────────────────────────

def quality_tier(fin: dict) -> tuple[Tier, str]:
    """
    返回 (tier, reason)
    - A: 5年ROE全 ≥15%，标准差<5%
    - B: 5年ROE全 ≥12%，且至少3年≥15%
    - C: 5年ROE全 ≥10%，趋势平稳或上行
    - X: 不合格（含困境反转待评估的也归 X，由调用方再判断）
    """
    roe = fin.get("roe_hist") or {}
    if len(roe) < 3:
        return "X", "ROE 历史不足 3 年"

    years = sorted(roe.keys(), reverse=True)[:5]
    vals = [roe[y] for y in years]
    if any(v is None for v in vals):
        return "X", "ROE 数据缺失"

    mn = min(vals)
    if mn < 10:
        return "X", f"近 {len(vals)} 年最低 ROE={mn:.1f}% < 10%"

    cfo_ratio = fin.get("cfo_to_profit")
    if cfo_ratio is not None and cfo_ratio < 0.6:
        return "X", f"经营现金流/净利润={cfo_ratio:.2f} < 0.6"

    debt = fin.get("debt_ratio")
    if debt is not None and debt > 70:
        return "X", f"资产负债率={debt:.1f}% > 70%"

    if mn >= 15:
        try:
            sd = statistics.stdev(vals) if len(vals) > 1 else 0
        except statistics.StatisticsError:
            sd = 0
        if sd < 5:
            return "A", f"ROE 全 ≥15%，σ={sd:.1f}"

    if mn >= 12 and sum(1 for v in vals if v >= 15) >= 3:
        return "B", f"ROE 全 ≥12%，{sum(1 for v in vals if v >= 15)} 年 ≥15%"

    if mn >= 10:
        recent3 = vals[:3]
        if recent3[0] >= recent3[-1] - 2:  # 平稳或上行（容差 2pct）
            return "C", f"ROE 全 ≥10%，近 3 年趋势 OK"

    return "X", f"ROE min={mn:.1f}% 未达分档"


def is_distress_reversal(fin: dict, snap: dict) -> bool:
    """困境反转豁免：10年ROE中位数≥12% & 当前 PB 在历史低位"""
    roe = fin.get("roe_hist") or {}
    vals = [v for v in roe.values() if v is not None]
    if len(vals) < 5:
        return False
    if statistics.median(vals) < 12:
        return False
    pb = snap.get("pb")
    return pb is not None and 0 < pb < 2.5


# ────────────────────────────────────────────────────────────
# 第 2 层：行业判断
# ────────────────────────────────────────────────────────────

def industry_check(industry: str, mv: float, industry_mv_rank: int | None,
                   wl: dict) -> tuple[bool, str, str]:
    """返回 (通过, 类别, 理由)，类别 ∈ {rising, stable_leader, none}"""
    if not industry:
        return False, "none", "无行业归属"
    if industry in wl["blacklist"]:
        return False, "none", f"行业 {industry} 在黑名单"
    if industry in wl["rising"]:
        return True, "rising", f"上升期行业：{industry}"
    if industry in wl["stable"]:
        if industry_mv_rank is not None and industry_mv_rank <= 3:
            return True, "stable_leader", f"成熟行业 {industry} 龙头（市值 #{industry_mv_rank}）"
        return False, "none", f"成熟行业 {industry} 但非龙头（市值排名 {industry_mv_rank}）"
    return False, "none", f"行业 {industry} 不在白名单"


# ────────────────────────────────────────────────────────────
# 第 3 层：护城河 & 趋势打分（0-100）
# ────────────────────────────────────────────────────────────

def moat_score(fin: dict) -> tuple[float, dict]:
    parts = {}

    gm = fin.get("gross_margin") or {}
    gm_vals = [v for v in gm.values() if v is not None]
    if gm_vals:
        avg_gm = sum(gm_vals) / len(gm_vals)
        parts["毛利率"] = min(100, max(0, (avg_gm - 10) * 2))
    else:
        parts["毛利率"] = 0

    roe = fin.get("roe_hist") or {}
    roe_vals = [v for v in roe.values() if v is not None]
    if roe_vals:
        avg_roe = sum(roe_vals) / len(roe_vals)
        parts["ROE 中枢"] = min(100, max(0, (avg_roe - 8) * 5))
    else:
        parts["ROE 中枢"] = 0

    rev = fin.get("revenue_hist") or {}
    yrs = sorted(rev.keys())
    if len(yrs) >= 3:
        try:
            old, new = rev[yrs[0]], rev[yrs[-1]]
            n = len(yrs) - 1
            cagr = (new / old) ** (1 / n) - 1 if old > 0 and new > 0 else 0
            parts["营收增速"] = min(100, max(0, (cagr * 100 - 5) * 4))
        except Exception:
            parts["营收增速"] = 0
    else:
        parts["营收增速"] = 0

    cfo_r = fin.get("cfo_to_profit")
    if cfo_r is not None:
        parts["现金流质量"] = min(100, max(0, (cfo_r - 0.5) * 100))
    else:
        parts["现金流质量"] = 50

    weights = {"毛利率": 0.3, "ROE 中枢": 0.3, "营收增速": 0.25, "现金流质量": 0.15}
    score = sum(parts[k] * w for k, w in weights.items())
    return round(score, 1), parts


# ────────────────────────────────────────────────────────────
# 第 4 层：安全边际
# ────────────────────────────────────────────────────────────

def margin_score(snap: dict, fin: dict) -> tuple[float, dict]:
    """简化版：PE 与行业经验值对比 + 隐含增速反推"""
    parts = {}
    pe = snap.get("pe_ttm")
    pb = snap.get("pb")

    if pe is None or pe <= 0:
        parts["PE"] = 0
    else:
        # PE 越低越好；20x 为基准
        parts["PE"] = min(100, max(0, (40 - pe) * 4))

    if pb is None or pb <= 0:
        parts["PB"] = 0
    else:
        parts["PB"] = min(100, max(0, (8 - pb) * 14))

    # 增速 vs PE：PEG
    rev = fin.get("revenue_hist") or {}
    yrs = sorted(rev.keys())
    peg = None
    if len(yrs) >= 3 and pe and pe > 0:
        try:
            old, new = rev[yrs[0]], rev[yrs[-1]]
            n = len(yrs) - 1
            cagr = (new / old) ** (1 / n) - 1 if old > 0 and new > 0 else 0
            if cagr > 0:
                peg = pe / (cagr * 100)
        except Exception:
            pass
    if peg is None:
        parts["PEG"] = 30
    else:
        parts["PEG"] = min(100, max(0, (2 - peg) * 50))

    weights = {"PE": 0.4, "PB": 0.25, "PEG": 0.35}
    score = sum(parts[k] * w for k, w in weights.items())
    return round(score, 1), parts


# ────────────────────────────────────────────────────────────
# 综合评分 & 主入口
# ────────────────────────────────────────────────────────────

TIER_THRESHOLD = {"strict": ("A",), "standard": ("A", "B"), "loose": ("A", "B", "C")}


def evaluate(snap: dict, fin: dict, industry: str,
             industry_mv_rank: int | None, wl: dict,
             mode: StrictMode = "standard") -> dict:
    tier, qreason = quality_tier(fin)
    distress = False
    if tier == "X":
        if is_distress_reversal(fin, snap):
            tier = "C"
            distress = True
            qreason = "困境反转豁免：" + qreason

    ind_pass, ind_cat, ind_reason = industry_check(industry, snap.get("mv") or 0,
                                                   industry_mv_rank, wl)

    moat, moat_parts = moat_score(fin)
    margin, margin_parts = margin_score(snap, fin)

    # 质量分（按档转分数）
    qscore = {"A": 100, "B": 80, "C": 60, "X": 0}[tier]
    iscore = 100 if ind_cat == "rising" else (80 if ind_cat == "stable_leader" else 0)

    final = qscore * 0.4 + iscore * 0.2 + moat * 0.2 + margin * 0.2

    allowed = tier in TIER_THRESHOLD[mode] and ind_pass
    return {
        "code": snap.get("code"),
        "name": snap.get("name"),
        "industry": industry,
        "price": snap.get("price"),
        "pe_ttm": snap.get("pe_ttm"),
        "pb": snap.get("pb"),
        "mv": snap.get("mv"),
        "tier": tier,
        "tier_reason": qreason,
        "distress_reversal": distress,
        "industry_pass": ind_pass,
        "industry_category": ind_cat,
        "industry_reason": ind_reason,
        "moat_score": moat,
        "moat_parts": moat_parts,
        "margin_score": margin,
        "margin_parts": margin_parts,
        "final_score": round(final, 1),
        "passed": allowed,
    }
