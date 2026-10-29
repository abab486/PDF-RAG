import unittest
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

from hello_agents import SimpleAgent, ToolRegistry
from hello_agents.core.database_config import (
    Neo4jConfig,
    QdrantConfig,
    db_config,
)
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.tools.builtin.memory_tool import MemoryTool
from hello_agents.tools.builtin.rag_tool import RAGTool

db_config.qdrant = QdrantConfig.from_env()
db_config.neo4j = Neo4jConfig.from_env()


class ObservedRAGTool(RAGTool):
    def __init__(self, *args, **kwargs):
        self.calls = []
        super().__init__(*args, **kwargs)

    def run(self, parameters):
        self.calls.append(parameters.copy())
        return super().run(parameters)


class ObservedMemoryTool(MemoryTool):
    def __init__(self, *args, **kwargs):
        self.calls = []
        super().__init__(*args, **kwargs)

    def run(self, parameters):
        self.calls.append(parameters.copy())
        return super().run(parameters)


class AgentToolDecisionTest(unittest.TestCase):
    """Observe and verify when the Agent selects RAG or Memory."""

    @classmethod
    def setUpClass(cls):
        tool_registry = ToolRegistry()
        cls.rag_tool = ObservedRAGTool(rag_namespace="pdf_demo_user")
        cls.memory_tool = ObservedMemoryTool(user_id="demo_user")
        tool_registry.register_tool(cls.rag_tool)
        tool_registry.register_tool(cls.memory_tool)

        system_prompt = (
            "你是一个学习助手。需要回答大语言模型、文档知识或概念比较问题时，"
            "使用 rag 工具搜索知识库，格式为 "
            "[TOOL_CALL:rag:search=查询内容]。用户询问之前的对话或记忆时，"
            "使用 memory 工具搜索记忆，格式为 "
            "[TOOL_CALL:memory:recall=查询内容]。简单算术、编写代码等不需要"
            "文档知识的问题直接回答，不要调用 rag。工具结果不足时如实说明。"
        )
        cls.agent = SimpleAgent(
            name="学习助手",
            llm=HelloAgentsLLM(),
            system_prompt=system_prompt,
            tool_registry=tool_registry,
        )

        cls.memory_tool.run({
            "action": "add",
            "content": "之前问过：大模型是什么？",
            "memory_type": "episodic",
            "importance": 0.9,
        })
        cls.memory_tool.calls.clear()

    def test_agent_tool_decisions(self):
        cases = [
            ("大模型是什么？", "rag"),
            ("1+1等于几？", None),
            ("我之前问过什么？", "memory"),
            ("帮我写一段Python代码", None),
            ("GPT和BERT的区别？", "rag"),
        ]

        for question, expected_tool in cases:
            with self.subTest(question=question):
                self.rag_tool.calls.clear()
                self.memory_tool.calls.clear()
                self.agent.clear_history()

                print(f"\n{'=' * 60}")
                print(f"问题：{question}")
                response = self.agent.run(question)

                called_tools = []
                if self.rag_tool.calls:
                    called_tools.append("rag")
                if self.memory_tool.calls:
                    called_tools.append("memory")

                print(f"实际调用：{called_tools or '无工具'}")
                print(f"回答：{response}")

                if expected_tool is None:
                    self.assertNotIn("rag", called_tools)
                    self.assertNotIn("memory", called_tools)
                else:
                    self.assertIn(expected_tool, called_tools)
                    self.assertEqual(called_tools, [expected_tool])


if __name__ == "__main__":
    unittest.main(verbosity=2)
