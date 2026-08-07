# quant-signal

美股技术分析提醒工具。定时拉取关注股票的 15 分钟 / 1 小时 K 线，计算技术指标，
生成买卖信号，通过飞书机器人推送。运行在 GitHub Actions 上，零服务器成本。

## 功能

- 多股票、多周期（15m / 60m）技术分析
- 指标：EMA、MACD、RSI、KDJ、布林带、ADX、量比
- 规则引擎生成买卖信号，带置信度分级
- 仅在美股交易时段触发，自动处理节假日与夏令时
- 信号去重，避免重复骚扰
- 飞书交互式卡片推送，买入绿/卖出红颜色区分
- 预留机器学习信号生成接口

## 快速开始

### 1. Fork 或克隆本仓库

### 2. 配置飞书机器人

1. 在飞书中创建一个群（或用现有群）
2. 群设置 → 群机器人 → 添加机器人 → 选择「自定义机器人」
3. 复制 webhook 地址（形如 `https://open.feishu.cn/open-apis/bot/v2/hook/xxxx`）
4. （可选）启用「签名校验」，记下 secret

在 GitHub 仓库 → Settings → Secrets and variables → Actions → New repository secret：

- Name: `FEISHU_WEBHOOK`　Value: webhook 完整地址（必填）
- Name: `FEISHU_SECRET`　Value: 签名密钥（启用加签才填，否则可不配）

### 3. 编辑关注列表

修改 `config/watchlist.yaml`，添加你关注的股票代码。

### 4. 手动测试

在仓库 Actions 页面选择「美股信号扫描」workflow → Run workflow → 勾选「跳过交易时段检查」即可立即验证推送。

## 本地运行

```bash
pip install -r requirements.txt
export FEISHU_WEBHOOK=你的webhook地址
export FEISHU_SECRET=你的签名密钥    # 可选，未启用加签则不设
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
