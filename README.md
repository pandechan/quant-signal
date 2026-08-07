# quant-signal

美股技术分析提醒工具。定时拉取关注股票的 15 分钟 / 1 小时 K 线，计算技术指标，
生成买卖信号，通过 Server酱 推送到微信。运行在 GitHub Actions 上，零服务器成本。

## 功能

- 多股票、多周期（15m / 60m）技术分析
- 指标：EMA、MACD、RSI、KDJ、布林带、ADX、量比
- 规则引擎生成买卖信号，带置信度分级
- 仅在美股交易时段触发，自动处理节假日与夏令时
- 信号去重，避免重复骚扰
- 预留机器学习信号生成接口

## 快速开始

### 1. Fork 或克隆本仓库

### 2. 配置 Server酱 SendKey

在 GitHub 仓库 → Settings → Secrets and variables → Actions → New repository secret：

- Name: `SERVERCHAN_SENDKEY`
- Value: 你的 Server酱 SendKey（https://sct.ftqq.com/ 申请）

### 3. 编辑关注列表

修改 `config/watchlist.yaml`，添加你关注的股票代码。

### 4. 手动测试

在仓库 Actions 页面选择 "美股信号扫描" workflow → Run workflow。

## 本地运行

```bash
pip install -r requirements.txt
export SERVERCHAN_SENDKEY=你的key
python src/main.py
```

## 配置说明

- `config/watchlist.yaml`：关注股票池
- `config/strategy.yaml`：指标参数与规则阈值

## 信号说明

| 方向 | 含义 |
|------|------|
| BUY | 买入信号 |
| SELL | 卖出信号 |
| WATCH | 关注，置信度不足以操作 |

## 免责声明

本工具生成的信号仅基于技术指标，不构成任何投资建议。投资有风险，决策需谨慎。
