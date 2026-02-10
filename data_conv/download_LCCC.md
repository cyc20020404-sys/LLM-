# 下载 LCCC 数据集到 data_conv

LCCC (Large-scale Cleaned Chinese Conversation) 为清华大学 CoAI 发布的大规模中文对话数据。

## 方式一：OpenXLab 命令行（推荐国内，替代已弃用的 odl）

**说明**：`odl get LCCC` 已弃用，且常出现 `0it` 实际未下载。请改用 **openxlab**（与 OpenDataLab 同账号，可用原账号密码在 openxlab 网站登录后获取 AK/SK）。

1. **安装并登录**（二选一）：
   - 交互式：`pip install openxlab` 后执行 `openxlab login`，按提示输入 **Access Key ID** 与 **Secret Access Key**（在 [OpenXLab 个人中心-密钥](https://sso.openxlab.org.cn/usercenter?tab=secret) 用原 OpenDataLab 账号登录后获取）。
   - 非交互：设置环境变量 `OPENXLAB_AK`、`OPENXLAB_SK` 后直接执行下载命令，无需 `openxlab login`。

2. **下载到 data_conv/LCCC**（任选其一）：
   ```bash
   cd /root/autodl-tmp/emention_bot
   # 命令行
   openxlab dataset get -r OpenDataLab/LCCC -t data_conv
   # 或 Python 脚本（同样会写到 data_conv/LCCC）
   python3 data_conv/download_lccc_openxlab.py
   ```
   下载完成后数据在 `data_conv/LCCC/` 下（含 `raw/LCCC-base-split.zip`、`raw/LCCC-large.zip` 等）。

## 方式二：Hugging Face（需可访问 hf.co 或镜像）

在可访问外网或已配置 HF 镜像的环境下：

```bash
cd /root/autodl-tmp/emention_bot
python3 data_conv/download_lccc_hf.py
```

会从 `silver/lccc` 拉取并写入 `data_conv/LCCC/`。

## 方式三：清华云盘直链（手动，链接可能失效）

若以下链接仍有效，可在浏览器中下载后解压到 `data_conv/LCCC/`：

- LCCC-base：<https://cloud.tsinghua.edu.cn/f/f131a4d259184566a29c/>
- LCCC-large：<https://cloud.tsinghua.edu.cn/f/8424e7b9454c4e628c24/>
