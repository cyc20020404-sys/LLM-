"""
基于图RAG的智能烹饪助手 - 主程序（DeepSeek API + Milvus + Neo4j）
"""

import os
import sys
import time
import logging
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from .config import DEFAULT_CONFIG, GraphRAGConfig
# 使用相对导入，避免与 emention_bot 顶层 rag_modules 冲突（Streamlit 先加载后者会占 sys.modules['rag_modules']）
from .rag_modules import (
    GraphDataPreparationModule,
    MilvusIndexConstructionModule,
    GenerationIntegrationModule,
)
from .rag_modules.hybrid_retrieval import HybridRetrievalModule
from .rag_modules.graph_rag_retrieval import GraphRAGRetrieval
from .rag_modules.intelligent_query_router import IntelligentQueryRouter, QueryAnalysis

load_dotenv()


class AdvancedGraphRAGSystem:
    """图RAG系统：智能路由 + 传统混合检索 + 图RAG检索，生成使用 DeepSeek API。"""

    def __init__(self, config: Optional[GraphRAGConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.generation_module = None
        self.traditional_retrieval = None
        self.graph_rag_retrieval = None
        self.query_router = None
        self.system_ready = False

    def initialize_system(self):
        """初始化系统"""
        logger.info("启动高级图RAG系统...")
        try:
            print("初始化数据准备模块...")
            self.data_module = GraphDataPreparationModule(
                uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database,
            )
            print("初始化Milvus向量索引...")
            self.index_module = MilvusIndexConstructionModule(
                host=self.config.milvus_host,
                port=self.config.milvus_port,
                collection_name=self.config.milvus_collection_name,
                dimension=self.config.milvus_dimension,
                model_name=self.config.embedding_model,
            )
            print("初始化生成模块（DeepSeek API）...")
            self.generation_module = GenerationIntegrationModule(
                model_name=self.config.llm_model,
                base_url=self.config.llm_base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            print("初始化传统混合检索...")
            self.traditional_retrieval = HybridRetrievalModule(
                config=self.config,
                milvus_module=self.index_module,
                data_module=self.data_module,
                llm_client=self.generation_module.client,
            )
            print("初始化图RAG检索引擎...")
            self.graph_rag_retrieval = GraphRAGRetrieval(
                config=self.config,
                llm_client=self.generation_module.client,
            )
            print("初始化智能查询路由器...")
            self.query_router = IntelligentQueryRouter(
                traditional_retrieval=self.traditional_retrieval,
                graph_rag_retrieval=self.graph_rag_retrieval,
                llm_client=self.generation_module.client,
                config=self.config,
            )
            print("✅ 高级图RAG系统初始化完成！")
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise

    def build_knowledge_base(self):
        """构建知识库"""
        print("\n检查知识库状态...")
        try:
            if self.index_module.has_collection():
                print("✅ 发现已存在的知识库，尝试加载...")
                if self.index_module.load_collection():
                    print("知识库加载成功！")
                    print("加载图数据以支持图检索...")
                    self.data_module.load_graph_data()
                    print("构建菜谱文档...")
                    self.data_module.build_recipe_documents()
                    print("进行文档分块...")
                    chunks = self.data_module.chunk_documents(
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap,
                    )
                    self._initialize_retrievers(chunks)
                    return
                else:
                    print("❌ 知识库加载失败，开始重建...")
            print("未找到已存在的集合，开始构建新的知识库...")
            print("从Neo4j加载图数据...")
            self.data_module.load_graph_data()
            print("构建菜谱文档...")
            self.data_module.build_recipe_documents()
            print("进行文档分块...")
            chunks = self.data_module.chunk_documents(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            print("构建Milvus向量索引...")
            if not self.index_module.build_vector_index(chunks):
                raise Exception("构建向量索引失败")
            self._initialize_retrievers(chunks)
            self._show_knowledge_base_stats()
            print("✅ 知识库构建完成！")
        except Exception as e:
            logger.error(f"知识库构建失败: {e}")
            raise

    def _initialize_retrievers(self, chunks: List = None):
        """初始化检索器"""
        print("初始化检索引擎...")
        if chunks is None:
            chunks = self.data_module.chunks or []
        self.traditional_retrieval.initialize(chunks)
        self.graph_rag_retrieval.initialize()
        self.system_ready = True
        print("✅ 检索引擎初始化完成！")

    def _show_knowledge_base_stats(self):
        """显示知识库统计"""
        print("\n知识库统计:")
        stats = self.data_module.get_statistics()
        print(f"   菜谱数量: {stats.get('total_recipes', 0)}")
        print(f"   食材数量: {stats.get('total_ingredients', 0)}")
        print(f"   烹饪步骤: {stats.get('total_cooking_steps', 0)}")
        print(f"   文档数量: {stats.get('total_documents', 0)}")
        print(f"   文本块数: {stats.get('total_chunks', 0)}")
        milvus_stats = self.index_module.get_collection_stats()
        print(f"   向量索引: {milvus_stats.get('row_count', 0)} 条记录")
        route_stats = self.query_router.get_route_statistics()
        print(f"   路由统计: 总查询 {route_stats.get('total_queries', 0)} 次")
        if stats.get("categories"):
            categories = list(stats["categories"].keys())[:10]
            print(f"   🏷️ 主要分类: {', '.join(categories)}")

    def ask_question_with_routing(
        self, question: str, stream: bool = False, explain_routing: bool = False
    ):
        """智能问答"""
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")
        print(f"\n❓ 用户问题: {question}")
        if explain_routing:
            explanation = self.query_router.explain_routing_decision(question)
            print(explanation)
        start_time = time.time()
        try:
            print("执行智能查询路由...")
            relevant_docs, analysis = self.query_router.route_query(
                question, self.config.top_k
            )
            strategy_icons = {
                "hybrid_traditional": "🔍",
                "graph_rag": "🕸️",
                "combined": "🔄",
            }
            strategy_icon = strategy_icons.get(
                analysis.recommended_strategy.value, "❓"
            )
            print(f"{strategy_icon} 使用策略: {analysis.recommended_strategy.value}")
            print(
                f"📊 复杂度: {analysis.query_complexity:.2f}, 关系密集度: {analysis.relationship_intensity:.2f}"
            )
            if relevant_docs:
                doc_info = []
                for doc in relevant_docs:
                    recipe_name = doc.metadata.get("recipe_name", "未知内容")
                    search_type = doc.metadata.get(
                        "search_type",
                        doc.metadata.get("route_strategy", "unknown"),
                    )
                    score = doc.metadata.get(
                        "final_score", doc.metadata.get("relevance_score", 0)
                    )
                    doc_info.append(f"{recipe_name}({search_type}, {score:.3f})")
                print(f"📋 找到 {len(relevant_docs)} 个相关文档: {', '.join(doc_info[:3])}")
                if len(doc_info) > 3:
                    print(f"    等 {len(relevant_docs)} 个结果...")
            else:
                return "抱歉，没有找到相关的烹饪信息。请尝试其他问题。", analysis
            print("🎯 智能生成回答...")
            if stream:
                try:
                    for chunk_text in self.generation_module.generate_adaptive_answer_stream(
                        question, relevant_docs
                    ):
                        print(chunk_text, end="", flush=True)
                    print("\n")
                    result = "流式输出完成"
                except Exception as stream_error:
                    logger.error(f"流式输出错误: {stream_error}")
                    print("\n⚠️ 流式输出中断，切换到标准模式...")
                    result = self.generation_module.generate_adaptive_answer(
                        question, relevant_docs
                    )
            else:
                result = self.generation_module.generate_adaptive_answer(
                    question, relevant_docs
                )
            end_time = time.time()
            print(f"\n⏱️ 问答完成，耗时: {end_time - start_time:.2f}秒")
            return result, analysis
        except Exception as e:
            logger.error(f"问答处理失败: {e}")
            return f"抱歉，处理问题时出现错误：{str(e)}", None

    def run_interactive(self):
        """交互式问答"""
        if not self.system_ready:
            print("❌ 系统未就绪，请先构建知识库")
            return
        print("\n欢迎使用图RAG烹饪助手（DeepSeek + Milvus + Neo4j）！")
        print("可用功能：")
        print("   - 'stats' : 查看系统统计")
        print("   - 'rebuild' : 重建知识库")
        print("   - 'quit' : 退出系统")
        print("\n" + "=" * 50)
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if not user_input:
                    continue
                if user_input.lower() == "quit":
                    break
                if user_input.lower() == "stats":
                    self._show_system_stats()
                    continue
                if user_input.lower() == "rebuild":
                    self._rebuild_knowledge_base()
                    continue
                print("\n回答:")
                result, analysis = self.ask_question_with_routing(
                    user_input, stream=True, explain_routing=False
                )
                if result and result != "流式输出完成":
                    print(f"{result}\n")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
                import traceback
                traceback.print_exc()
        print("\n👋 感谢使用！")
        self._cleanup()

    def _show_system_stats(self):
        """系统统计"""
        print("\n系统运行统计")
        print("=" * 40)
        route_stats = self.query_router.get_route_statistics()
        total_queries = route_stats.get("total_queries", 0)
        if total_queries > 0:
            print(f"总查询次数: {total_queries}")
            print(
                f"传统检索: {route_stats.get('traditional_count', 0)} ({route_stats.get('traditional_ratio', 0):.1%})"
            )
            print(
                f"图RAG检索: {route_stats.get('graph_rag_count', 0)} ({route_stats.get('graph_rag_ratio', 0):.1%})"
            )
            print(
                f"组合策略: {route_stats.get('combined_count', 0)} ({route_stats.get('combined_ratio', 0):.1%})"
            )
        else:
            print("暂无查询记录")
        self._show_knowledge_base_stats()

    def _rebuild_knowledge_base(self):
        """重建知识库"""
        print("\n准备重建知识库...")
        confirm = input(
            "⚠️  这将删除现有的向量数据并重新构建，是否继续？(y/N): "
        ).strip().lower()
        if confirm != "y":
            print("❌ 重建操作已取消")
            return
        try:
            print("删除现有的Milvus集合...")
            if self.index_module.delete_collection():
                print("✅ 现有集合已删除")
            else:
                print("删除集合时出现问题，继续重建...")
            print("开始重建知识库...")
            self.build_knowledge_base()
            print("✅ 知识库重建完成！")
        except Exception as e:
            logger.error(f"重建知识库失败: {e}")
            print(f"❌ 重建失败: {e}")

    def _cleanup(self):
        """清理资源"""
        if self.data_module:
            self.data_module.close()
        if self.traditional_retrieval:
            self.traditional_retrieval.close()
        if self.graph_rag_retrieval:
            self.graph_rag_retrieval.close()
        if self.index_module:
            self.index_module.close()


def main():
    try:
        print("启动高级图RAG系统...")
        rag_system = AdvancedGraphRAGSystem()
        rag_system.initialize_system()
        rag_system.build_knowledge_base()
        rag_system.run_interactive()
    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        import traceback
        traceback.print_exc()
        err_msg = str(e).lower()
        if "7687" in err_msg or "connection refused" in err_msg or "neo4j" in err_msg:
            print("\n❌ Neo4j 未连接。请先在本目录启动 Docker 服务：")
            print("   cd emention_bot/graph_rag_deepseek")
            print("   docker compose up -d")
            print("   等待约 1～2 分钟后重新运行 python main.py")
        elif "19530" in err_msg or "milvus" in err_msg:
            print("\n❌ Milvus 未连接。请先启动 Docker：docker compose up -d")
        else:
            print(f"\n❌ 系统错误: {e}")


if __name__ == "__main__":
    main()
