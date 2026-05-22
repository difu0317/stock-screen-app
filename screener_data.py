"""
股票筛选器 - 数据层
- akshare 拉全市场数据
- SQLite 本地缓存（按日期）
- 分两步：snapshot（快，全市场） + financials（慢，按需）
"""

from __future__ import annotations

import sqlite3
import time
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import akshare as ak
import pandas as pd

DB_PATH = Path(__file__).parent / "screener_cache.db"


# ────────────────────────────────────────────────────────────
# DB 初始化
# ────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS snapshot (
            snap_date TEXT,
            code      TEXT,
            name      TEXT,
            price     REAL,
            pe_ttm    REAL,
            pb        REAL,
            mv        REAL,
            PRIMARY KEY (snap_date, code)
        );
        CREATE TABLE IF NOT EXISTS industry_map (
            code     TEXT PRIMARY KEY,
            industry TEXT
        );
        CREATE TABLE IF NOT EXISTS financials (
            code        TEXT,
            fetch_date  TEXT,
            payload     TEXT,
            PRIMARY KEY (code, fetch_date)
        );
        """)


def _retry(fn, *args, tries: int = 5, sleep: float = 2.0, **kwargs):
    """简单重试，应对 akshare 偶发连接断开（指数退避 + 抖动）"""
    import random
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            time.sleep(sleep * (2 ** i) + random.uniform(0, 1))
    raise last


# ────────────────────────────────────────────────────────────
# 全市场快照（每日刷新一次）
# ────────────────────────────────────────────────────────────

def _fetch_snapshot_em() -> pd.DataFrame:
    """东方财富源（首选，含 PE/PB/市值）"""
    df = ak.stock_zh_a_spot_em()
    df = df.rename(columns={
        "代码": "code", "名称": "name", "最新价": "price",
        "市盈率-动态": "pe_ttm", "市净率": "pb", "总市值": "mv",
    })
    df = df[["code", "name", "price", "pe_ttm", "pb", "mv"]].copy()
    df["mv"] = df["mv"] / 1e8
    return df


def _fetch_snapshot_em_by_market() -> pd.DataFrame:
    """东方财富分市场拼接（沪/深/北分别拉，规避全市场接口限流）"""
    parts = []
    for fn_name in ("stock_sh_a_spot_em", "stock_sz_a_spot_em", "stock_bj_a_spot_em"):
        fn = getattr(ak, fn_name, None)
        if fn is None:
            continue
        try:
            d = fn()
            parts.append(d)
            time.sleep(1.0)
        except Exception:
            continue
    if not parts:
        raise RuntimeError("分市场接口全部失败")
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns={
        "代码": "code", "名称": "name", "最新价": "price",
        "市盈率-动态": "pe_ttm", "市净率": "pb", "总市值": "mv",
    })
    keep = [c for c in ["code", "name", "price", "pe_ttm", "pb", "mv"] if c in df.columns]
    df = df[keep].drop_duplicates("code").copy()
    if "mv" in df.columns:
        df["mv"] = df["mv"] / 1e8
    for col in ("pe_ttm", "pb", "mv"):
        if col not in df.columns:
            df[col] = None
    return df[["code", "name", "price", "pe_ttm", "pb", "mv"]]


def _fetch_snapshot_sina() -> pd.DataFrame:
    """新浪源（备用，无 PE/PB/市值，需要后续补）"""
    df = ak.stock_zh_a_spot()
    df = df.rename(columns={
        "代码": "code", "名称": "name", "最新价": "price",
    })
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
    for col in ("pe_ttm", "pb", "mv"):
        if col not in df.columns:
            df[col] = None
    return df[["code", "name", "price", "pe_ttm", "pb", "mv"]].copy()


def _fetch_snapshot_tencent() -> pd.DataFrame:
    """腾讯源（最后兜底，部分版本 akshare 提供）"""
    fn = getattr(ak, "stock_zh_a_alerts_cls", None)  # 仅占位，主要用下方 ths
    fn2 = getattr(ak, "stock_a_all_pb", None)
    if fn2 is not None:
        try:
            df = fn2()
            return pd.DataFrame()  # 字段口径不同，跳过
        except Exception:
            pass
    raise RuntimeError("无可用腾讯接口")


def _fetch_snapshot_with_fallback(status_cb=None) -> tuple[pd.DataFrame, str]:
    """多层 fallback：东方财富全市场 → 东方财富分市场 → 新浪"""
    sources = [
        ("东方财富(全)", _fetch_snapshot_em, 3),
        ("东方财富(分市场)", _fetch_snapshot_em_by_market, 2),
        ("新浪", _fetch_snapshot_sina, 3),
    ]
    errors = []
    for name, fn, tries in sources:
        if status_cb:
            status_cb(f"尝试 {name} ...")
        try:
            df = _retry(fn, tries=tries, sleep=2.0)
            if df is not None and len(df) > 100:
                return df, name
            errors.append(f"{name}: 返回数据过少({0 if df is None else len(df)})")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise RuntimeError("所有数据源均失败 → " + " | ".join(errors))


def refresh_snapshot(force: bool = False, status_cb=None) -> pd.DataFrame:
    """
    拉取 A 股全市场实时快照，多层 fallback：
    1) 东方财富全市场 → 2) 东方财富分市场拼接 → 3) 新浪 → 4) 旧快照兜底
    缓存当日数据
    """
    init_db()
    today = date.today().isoformat()
    with _conn() as c:
        if not force:
            n = c.execute(
                "SELECT COUNT(*) FROM snapshot WHERE snap_date=?", (today,)
            ).fetchone()[0]
            if n > 1000:
                return load_snapshot(today)

    try:
        df, source = _fetch_snapshot_with_fallback(status_cb=status_cb)
    except Exception as e:
        # 最终兜底：用最近一次本地快照
        old = load_snapshot()
        if not old.empty:
            old.attrs["source"] = f"本地缓存({old['snap_date'].iloc[0]})"
            old.attrs["fallback_warning"] = str(e)
            return old
        raise

    df["snap_date"] = today
    df = df.dropna(subset=["price"])

    with _conn() as c:
        c.execute("DELETE FROM snapshot WHERE snap_date=?", (today,))
        df[["snap_date", "code", "name", "price", "pe_ttm", "pb", "mv"]].to_sql(
            "snapshot", c, if_exists="append", index=False
        )
    df.attrs["source"] = source
    return df


def load_snapshot(snap_date: str | None = None) -> pd.DataFrame:
    init_db()
    if snap_date is None:
        with _conn() as c:
            row = c.execute("SELECT MAX(snap_date) FROM snapshot").fetchone()
            snap_date = row[0] if row and row[0] else None
    if snap_date is None:
        return pd.DataFrame()
    with _conn() as c:
        return pd.read_sql(
            "SELECT * FROM snapshot WHERE snap_date=?", c, params=(snap_date,)
        )


# ────────────────────────────────────────────────────────────
# 行业归属（一次性拉，结果稳定）
# ────────────────────────────────────────────────────────────

def refresh_industry_map(force: bool = False, status_cb=None) -> pd.DataFrame:
    """
    刷新行业映射，多层 fallback：
    1) 板块列表 + 逐板块成分（最完整）
    2) 全市场快照 + 单股 stock_individual_info_em（兜底，慢但稳）
    3) 已有的本地缓存
    """
    init_db()
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM industry_map").fetchone()[0]
        if n > 1000 and not force:
            return pd.read_sql("SELECT * FROM industry_map", c)

    rows: list[tuple[str, str]] = []
    try:
        if status_cb:
            status_cb("尝试板块成分接口...")
        boards = _retry(ak.stock_board_industry_name_em, tries=3, sleep=2.0)
        name_col = "板块名称" if "板块名称" in boards.columns else boards.columns[0]
        ok_boards = 0
        for ind in boards[name_col].tolist():
            try:
                cons = _retry(ak.stock_board_industry_cons_em, symbol=ind, tries=2, sleep=1.0)
                code_col = "代码" if "代码" in cons.columns else cons.columns[1]
                for code in cons[code_col].astype(str).tolist():
                    rows.append((code.zfill(6), ind))
                ok_boards += 1
                time.sleep(0.2)
            except Exception:
                continue
        if ok_boards < 5:
            raise RuntimeError(f"板块成分仅取到 {ok_boards} 个，触发兜底")
    except Exception as e:
        if status_cb:
            status_cb(f"板块接口失败({e})，切换到单股查询...")
        rows = []
        snap = load_snapshot()
        if snap.empty:
            try:
                snap, _ = _fetch_snapshot_with_fallback(status_cb=status_cb)
            except Exception:
                pass
        if not snap.empty:
            codes = snap["code"].astype(str).tolist()
            for i, code in enumerate(codes):
                if status_cb and i % 50 == 0:
                    status_cb(f"单股查询行业 {i}/{len(codes)}")
                try:
                    info = ak.stock_individual_info_em(symbol=code)
                    val_col = "value" if "value" in info.columns else info.columns[1]
                    item_col = "item" if "item" in info.columns else info.columns[0]
                    ind_row = info[info[item_col] == "行业"]
                    if not ind_row.empty:
                        rows.append((code.zfill(6), str(ind_row[val_col].iloc[0])))
                    time.sleep(0.1)
                except Exception:
                    continue

    if not rows:
        # 最终兜底：用旧数据
        with _conn() as c:
            old = pd.read_sql("SELECT * FROM industry_map", c)
        if not old.empty:
            return old
        raise RuntimeError("行业映射拉取失败且无本地缓存")

    df = pd.DataFrame(rows, columns=["code", "industry"]).drop_duplicates("code")
    with _conn() as c:
        c.execute("DELETE FROM industry_map")
        df.to_sql("industry_map", c, if_exists="append", index=False)
    return df


def load_industry_map() -> pd.DataFrame:
    init_db()
    with _conn() as c:
        return pd.read_sql("SELECT * FROM industry_map", c)


# ────────────────────────────────────────────────────────────
# 单股财务历史（慢，按需调用，缓存 7 天）
# ────────────────────────────────────────────────────────────

def fetch_financials(code: str, max_age_days: int = 7) -> dict:
    """
    返回 {
      'roe_hist': {年份: ROE%}, 'profit_hist': {年份: 净利润亿},
      'revenue_hist': {年份: 营收亿}, 'gross_margin': {年份: 毛利率%},
      'cfo_to_profit': float, 'debt_ratio': float,
    }
    """
    import json
    init_db()
    today = date.today().isoformat()
    with _conn() as c:
        row = c.execute(
            "SELECT fetch_date, payload FROM financials WHERE code=? "
            "ORDER BY fetch_date DESC LIMIT 1", (code,)
        ).fetchone()
    if row:
        age = (datetime.fromisoformat(today) - datetime.fromisoformat(row[0])).days
        if age <= max_age_days:
            return json.loads(row[1])

    payload = _fetch_financials_raw(code)
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO financials VALUES (?, ?, ?)",
            (code, today, json.dumps(payload, ensure_ascii=False)),
        )
    return payload


def _fetch_financials_raw(code: str) -> dict:
    out: dict = {
        "roe_hist": {}, "profit_hist": {}, "revenue_hist": {},
        "gross_margin": {}, "cfo_to_profit": None, "debt_ratio": None,
    }
    try:
        df = ak.stock_financial_abstract(symbol=code)
        if df is None or df.empty:
            return out

        idx_col = df.columns[0]
        df = df.set_index(idx_col)
        year_cols = [c for c in df.columns if str(c).startswith("20")]
        year_cols = sorted(year_cols, reverse=True)[:6]

        def _series(keyword: str) -> dict:
            for k in df.index:
                if keyword in str(k):
                    s = df.loc[k, year_cols]
                    return {str(y)[:4]: _to_float(v) for y, v in s.items()
                            if _to_float(v) is not None}
            return {}

        out["roe_hist"] = _series("净资产收益率")
        profit = _series("净利润")
        out["profit_hist"] = {y: v / 1e8 for y, v in profit.items()}
        rev = _series("营业总收入") or _series("营业收入")
        out["revenue_hist"] = {y: v / 1e8 for y, v in rev.items()}
        out["gross_margin"] = _series("毛利率") or _series("销售毛利率")

        cfo = _series("经营活动产生的现金流量净额") or _series("经营现金流")
        if profit and cfo:
            recent_y = sorted(set(profit) & set(cfo), reverse=True)[:3]
            if recent_y:
                p = sum(profit[y] for y in recent_y)
                c_ = sum(cfo[y] for y in recent_y)
                if p > 0:
                    out["cfo_to_profit"] = round(c_ / p, 2)

        debt = _series("资产负债率")
        if debt:
            latest_y = max(debt)
            out["debt_ratio"] = debt[latest_y]
    except Exception:
        pass
    return out


def _to_float(x) -> float | None:
    if x is None:
        return None
    s = str(x).strip().replace(",", "").replace("%", "")
    if s in ("", "--", "nan", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_financials_batch(codes: Iterable[str], progress_cb=None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    codes = list(codes)
    for i, code in enumerate(codes):
        out[code] = fetch_financials(code)
        if progress_cb:
            progress_cb(i + 1, len(codes))
        time.sleep(0.15)
    return out
