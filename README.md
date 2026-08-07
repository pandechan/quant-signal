# quant-signal

美股技术分析提醒工具。定时拉取关注股票的 15 分钟 / 1 小时 K 线，计算技术指标，
生成买卖信号，通过飞书机器人推送。运行在 GitHub Actions 上，零服务器成本。

## 功能

- 多股票、多周期（15m / 60m）技术分析
- 数据源：**Alpaca 实时数据**（优先），未配置时自动回退 yfinance
- 指标：EMA、MACD、RSI、KDJ、布林带、ADX、量比（纯 pandas 实现）
- 规则引擎生成买卖信号，带置信度分级
- 仅在美股交易时段触发，自动处理节假日与夏令时
- 信号去重，避免重复骚扰
- 飞书交互式卡片推送，买入绿/卖出红颜色区分
- 预留机器学习信号生成接口

## 快速开始

### 1. Fork 或克隆本仓库

### 2. 配置 Alpaca 数据源（实时数据，推荐）

1. 注册 Alpaca paper trading 账号：https://app.alpaca.markets/
2. 进入 API Keys 页面，创建一个新的 API Key
3. 复制 **API Key ID** 和 **Secret Key**

在 GitHub 仓库 → Settings → Secrets and variables → Actions → New repository secret：

- `ALPACA_API_KEY` = API Key ID
- `ALPACA_SECRET_KEY` = Secret Key

> 未配置 Alpaca 时会自动回退 yfinance（有 15-20 分钟延迟），不影响运行。

### 3. 配置飞书机器人

1. 在飞书中创建一个群（或用现有群）
2. 群设置 → 群机器人 → 添加机器人 → 选择「自定义机器人」
3. 复制 webhook 地址（形如 `https://open.feishu.cn/open-apis/bot/v2/hook/xxxx`）
4. （可选）启用「签名校验」，记下 secret

在 GitHub Secrets 添加：

- `FEISHU_WEBHOOK` = webhook 完整地址（必填）
- `FEISHU_SECRET` = 签名密钥（启用加签才填，否则可不配）

### 4. 编辑关注列表

修改 `config/watchlist.yaml`，添加你关注的股票代码。

### 5. 手动测试

仓库 Actions 页面 →「美股信号扫描」→ Run workflow → 勾选「跳过交易时段检查」。

## 本地运行

```bash
pip install -r requirements.txt

# 数据源（二选一）
export ALPACA_API_KEY=你的key_id        # 推荐，实时数据
export ALPACA_SECRET_KEY=你的secret

# 推送
export FEISHU_WEBHOOK=你的webhook地址
export FEISHU_SECRET=你的签名密钥        # 可选

python src/main.py --force
```

`--force` 跳过交易时段检查，方便非交易时段测试。

## 配置说明

- `config/watchlist.yaml`：关注股票池
- `config/strategy.yaml`：指标参数与规则阈值

## 信号说明

| 方向 | 含义 | 卡片颜色 |
|------|------|---------|
| BUY | 买入信号 | 绿色 |
| SELL | 卖出信号 | 红色 |
| WATCH | 关注，置信度不足以操作 | 蓝色 |

## 免责声明

本工具生成的信号仅基于技术指标，不构成任何投资建议。投资有风险，决策需谨慎。
