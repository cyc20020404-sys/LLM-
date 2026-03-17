# 🩷 Chinese Adorable High Emotional Intelligence Chat Dataset
### 💬 中文高情商可爱聊天数据集

## 简要参数
---
license: cc-by-4.0

task_categories:
- table-question-answering

language:
- zh

tags:
- chat
- emotional

size_categories:
- n<1K

---

## 🧩 简介 (Overview)

**Chinese Adorable High Emotional Intelligence Chat Dataset** 是一个中文对话数据集，专注于**高情商、轻松幽默、温柔治愈风格**的自然对话。
对话以“**user**”和“**girl**”为角色构成，模拟出一种**温柔、聪慧且带点俏皮的女性语气**，用于训练能自然、情绪感知良好的中文对话模型。

本数据集尤其适合：

* 微调情绪对话模型（Emotional Chatbot）
* 训练高情商人格角色（Roleplay / Companion AI）
* 情绪识别与反应生成任务（Emotion-aware Response Generation）
* 人格化对话研究（Persona-based Dialogue）

---

## 📦 数据示例 (Data Example)

```json
{
    "user": "你今天状态不太好。",
    "girl": "那我今天就当你的充电宝，抱一下回血百分之十～"
}
```

或另一组更完整的上下文：

```json
[
    {
        "user": "你不是说要早起跑步吗？",
        "girl": "是啊，但床太粘人了，我一挣扎它就更抱我紧了～"
    },
    {
        "user": "真拿你没办法。",
        "girl": "那你就别拿了，直接抱吧～"
    }
]
```

---

## 🧠 数据特征 (Characteristics)

| 特征       | 描述                                  |
| -------- | ----------------------------------- |
| **语言**   | 中文                                  |
| **数据量**  | 约 170 组高情商对话                         |
| **格式**   | JSON 数组，每条包含 `"user"` 与 `"girl"` 字段 |
| **对话风格** | 温柔、调皮、聪明、善于共情                       |
| **情绪维度** | 轻松 / 暧昧 / 安慰 / 打趣 / 撩动 / 理解         |
| **话语长度** | 平均 15–30 个字，语言自然且含情绪转折              |

---

## 💡 设计理念 (Design Philosophy)

本数据集的核心目标是捕捉“**高情商语用模式**”的语言特征，包括：

* **共情性回应**：理解情绪并以轻松语气化解；
* **语言柔化**：使用“拟人、调侃、撒娇”等表达降低冲突；
* **人际亲密语气**：模糊距离感的表达，如“抱”“夸”“小笨蛋”等；
* **隐含心理反馈**：以轻喜剧语态传达情绪调节能力。

这种语言风格常见于高情绪智商的社交互动、伴侣式AI或轻治愈向角色中。

---

## 🧰 数据格式 (Schema)

每条数据均为独立对话对象，格式如下：

| 字段     | 类型       | 描述                  |
| ------ | -------- | ------------------- |
| `user` | `string` | 用户发言（通常为中性语气或情境引导）  |
| `girl` | `string` | 高情商女性角色回复，富有情感与语言技巧 |

示例：

```json
{
  "user": "你生气了吗？",
  "girl": "没有呀～只是心情暂时进了小黑屋，等你一句哄就能放风～"
}
```

---

## 🪄 使用场景 (Use Cases)

| 场景                     | 示例               |
| ---------------------- | ---------------- |
| 🎭 **情感陪伴AI / 虚拟女友模型** | 微调以获得自然的暧昧与安慰语气  |
| 🧘 **情绪引导与安抚模型**       | 生成具同理心的回应        |
| 💬 **对话生成研究**          | 用于风格迁移 / 情感标签学习  |
| 💞 **个性化语料训练**         | 构建“温柔型人格”Chatbot |

---

## ⚖️ 许可协议 (License)

本数据集建议使用：

**🪪 License:**[CC BY 4.0 (Attribution)](https://creativecommons.org/licenses/by/4.0/)

> 允许自由研究、修改与分发。
> 请在使用时注明出处：
>
> ```
> Dataset: Chinese Adorable High Emotional Intelligence Chat (2025)
> Author: memorialsummer
> ```

---

## 🏗️ 未来扩展 (Future Work)

* 增加更多语气类型（治愈 / 犀利 / 理性 / 哄劝）
* 加入多轮上下文对话（Context-aware）
* 添加情感标签（如 joy / comfort / tease）
* 引入语音语料（Tone-aware fine-tuning）

---

## 🌸 引用 (Citation)

如果你在研究或项目中使用本数据集，请引用如下格式：

```bibtex
@dataset{memorialsummer_2025adorablechat,
  title     = {Chinese Adorable High Emotional Intelligence Chat Dataset},
  author    = {Memorial_Summer},
  year      = {2025},
  note      = {https://huggingface.co/datasets/MemorialSummer/chinese-adorable-high-emotional-intelligence-chat}
}
```