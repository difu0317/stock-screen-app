# 价值股筛选器 · 使用说明

价值投资视角的全市场筛选工具，与 `growth_valuation.py` 同一 Streamlit 应用。

## 安装

```bash
source venv/bin/activate
pip install pyyaml      # 新增依赖
# akshare、streamlit 等若已装可跳过
```

## 启动

```bash
streamlit run growth_valuation.py
```

启动后侧边栏会出现两个页面：

- **growth_valuation**（主页，单股深度估值）
- **🔎 价值股筛选器**（多股筛选）

## 首次使用

进入筛选器页面后，按顺序点击侧边栏：

1. **🔄 刷新全市场快照**（约 10 秒，每天点一次即可）
2. **🏭 刷新行业映射**（约 2-3 分钟，首次必须；之后每月点一次）
3. 设置严格度 / 市值 / PE 过滤器，点 **🚀 开始筛选**

筛选时会逐个拉取候选股票的 5 年财务（缓存 7 天），200 只约 1 分钟。

## 筛选逻辑（四层漏斗）

| 层 | 内容 | 实现位置 |
| --- | --- | --- |
| 1. 质量门槛 | ROE 三档 (A/B/C) + 现金流 + 负债 | `screener_logic.quality_tier` |
| 2. 行业判断 | 上升期白名单 / 成熟行业龙头 | `industry_whitelist.yaml` |
| 3. 护城河打分 | 毛利率、ROE 中枢、营收 CAGR、现金流质量 | `screener_logic.moat_score` |
| 4. 安全边际 | PE / PB / PEG | `screener_logic.margin_score` |

**ROE 三档**

- A 档（严格）：5 年 ROE 全 ≥ 15%，σ < 5%
- B 档（标准）：5 年 ROE 全 ≥ 12%，且至少 3 年 ≥ 15%
- C 档（宽松）：5 年 ROE 全 ≥ 10%，趋势平稳或上行

**困境反转豁免**：10 年 ROE 中位数 ≥ 12% 且 PB < 2.5 时，自动归入 C 档并标记。

**综合评分**：质量 40% + 行业 20% + 护城河 20% + 安全边际 20%

## 自定义行业白名单

编辑 `industry_whitelist.yaml`：

```yaml
rising:    # 上升期行业，全部通过行业层
  - 半导体
  - ...
stable:    # 成熟稳定行业，仅市值前 3 龙头通过
  - 白酒
  - ...
blacklist: # 直接淘汰
  - 教育
```

行业名称要和 akshare `stock_board_industry_name_em` 的命名一致。修改后刷新页面即可。

## 文件结构

```
stock_project/
├── growth_valuation.py            # 主入口（单股估值，原文件）
├── pages/
│   └── 1_🔎_价值股筛选器.py        # 筛选页
├── screener_data.py               # akshare 数据 + SQLite 缓存
├── screener_logic.py              # 四层漏斗逻辑
├── industry_whitelist.yaml        # 行业白名单（你可改）
└── screener_cache.db              # 自动生成的本地缓存
```

## 使用流程建议

1. 在筛选器里跑出"入选池"，下载 CSV
2. 关注综合分 ≥ 70 且安全边际 ≥ 60 的标的
3. 对感兴趣的标的，复制代码到主页 `growth_valuation` 做反向 DCF 深挖
4. 关注列表定期（每周/每月）重跑筛选器，观察分数变化
