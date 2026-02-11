"""
生成集成模块 - 使用 DeepSeek API
API 密钥从环境变量 DEEPSEEK_API_KEY 读取，勿写入代码。
"""

import logging
import os
import time
from typing import List

from openai import OpenAI
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """生成集成模块 - 负责答案生成（DeepSeek API）"""

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置环境变量 DEEPSEEK_API_KEY（勿将 key 写入代码或提交仓库）")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )
        logger.info(f"生成模块初始化完成，模型: {model_name}")

    def generate_adaptive_answer(self, question: str, documents: List[Document]) -> str:
        """基于检索结果生成回答"""
        context_parts = []
        for doc in documents:
            content = doc.page_content.strip()
            if content:
                level = doc.metadata.get("retrieval_level", "")
                if level:
                    context_parts.append(f"[{level.upper()}] {content}")
                else:
                    context_parts.append(content)
        context = "\n\n".join(context_parts)

        prompt = f"""
作为一位专业的烹饪助手，请基于以下信息回答用户的问题。

检索到的相关信息：
{context}

用户问题：{question}

请提供准确、实用的回答。根据问题的性质：
- 如果是询问多个菜品，请提供清晰的列表
- 如果是询问具体制作方法，请提供详细步骤
- 如果是一般性咨询，请提供综合性回答

回答：
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def generate_adaptive_answer_stream(
        self, question: str, documents: List[Document], max_retries: int = 3
    ):
        """流式生成回答（带重试）"""
        context_parts = []
        for doc in documents:
            content = doc.page_content.strip()
            if content:
                level = doc.metadata.get("retrieval_level", "")
                if level:
                    context_parts.append(f"[{level.upper()}] {content}")
                else:
                    context_parts.append(content)
        context = "\n\n".join(context_parts)

        prompt = f"""
作为一位专业的烹饪助手，请基于以下信息回答用户的问题。

检索到的相关信息：
{context}

用户问题：{question}

请提供准确、实用的回答。根据问题的性质：
- 如果是询问多个菜品，请提供清晰的列表
- 如果是询问具体制作方法，请提供详细步骤
- 如果是一般性咨询，请提供综合性回答

回答：
"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=60,
                )
                if attempt == 0:
                    print("开始流式生成回答...\n")
                else:
                    print(f"第{attempt + 1}次尝试流式生成...\n")
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                logger.warning(f"流式生成第{attempt + 1}次尝试失败: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"⚠️ 连接中断，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error("流式生成完全失败，尝试非流式后备")
                    print("⚠️ 流式生成失败，切换到标准模式...")
                    try:
                        fallback = self.generate_adaptive_answer(question, documents)
                        yield fallback
                    except Exception as fe:
                        yield f"抱歉，生成失败：{str(fe)}"
                    return
