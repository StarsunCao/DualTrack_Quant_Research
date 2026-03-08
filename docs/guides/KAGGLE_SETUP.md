# Kaggle API 配置指南

## 步骤 1：注册 Kaggle 账号

1. 访问 https://www.kaggle.com/
2. 点击右上角 "Register" 注册账号
3. 填写信息并验证邮箱

## 步骤 2：生成 API Token

1. 登录 Kaggle 后，点击右上角头像
2. 选择 "My Account"（我的账号）
3. 滚动到 "API" 部分
4. 点击 "Create New API Token"（创建新 API 令牌）
5. 自动下载 `kaggle.json` 文件

## 步骤 3：配置 API Token

在终端中执行以下命令：

```bash
# 创建 .kaggle 目录
mkdir -p ~/.kaggle

# 移动 kaggle.json 到正确位置
mv ~/Downloads/kaggle.json ~/.kaggle/

# 设置权限（必须）
chmod 600 ~/.kaggle/kaggle.json
```

## 步骤 4：验证配置

```bash
# 测试 Kaggle API
kaggle datasets list
```

如果看到数据集列表，说明配置成功！

## 数据集下载

配置完成后，运行以下命令下载数据：

```bash
# 下载开源美股新闻数据
python scripts/download_open_source_news.py

# 清洗和合并数据
python scripts/clean_and_merge_news.py

# 测试数据质量
python tests/test_us_news_data.py
```

## 故障排除

### 错误：403 Forbidden

- 检查 `kaggle.json` 文件权限是否正确（应为 600）
- 确认 API Token 未过期

### 错误：kaggle command not found

- 确认已安装 kaggle：`pip show kaggle`
- 检查 PATH 环境变量

### 错误：No such file or directory

- 确认 `kaggle.json` 文件在 `~/.kaggle/` 目录下
- 检查文件名是否正确（应为 `kaggle.json`，不是 `kaggle (1).json` 等）

## 参考链接

- Kaggle API 官方文档：https://github.com/Kaggle/kaggle-api
- 数据集 1：https://www.kaggle.com/datasets/notlucasp/financial-news-headlines
- 数据集 2：https://www.kaggle.com/datasets/zeroshot/twitter-financial-news-sentiment
- 数据集 3：https://www.kaggle.com/datasets/TheFinAI/esg-news