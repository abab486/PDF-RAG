import contextlib
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from assistant import PDFLearningAssistant
from hello_agents.core.llm import HelloAgentsLLM

EVAL_DIR = Path(__file__).resolve().parent
TEST_SET_PATH = EVAL_DIR / "routing_test_set.json"
RESULTS_PATH = EVAL_DIR / "routing_results.json"


def load_routing_test_set():
    with TEST_SET_PATH.open("r", encoding="utf-8") as f:
        test_set = json.load(f)

    if not isinstance(test_set, list) or not test_set:
        raise ValueError(f"路由测试集必须是非空 JSON 数组: {TEST_SET_PATH}")

    seen_ids = set()
    for index, item in enumerate(test_set):
        if not isinstance(item, dict):
            raise ValueError(f"路由测试集第 {index + 1} 条必须是 JSON 对象")
        if not {"id", "question", "expected_route"}.issubset(item):
            raise ValueError(f"路由测试集第 {index + 1} 条缺少必需字段")
        if item["expected_route"] not in {"RAG", "MEMORY"}:
            raise ValueError(f"测试题 {item['id']} 的 expected_route 必须是 RAG 或 MEMORY")
        if item["id"] in seen_ids:
            raise ValueError(f"路由测试集存在重复 ID: {item['id']}")
        seen_ids.add(item["id"])

    return test_set


class ObservedLLM:
    def __init__(self, llm):
        self.llm = llm
        self.last_response = None

    def invoke(self, messages, **kwargs):
        self.last_response = self.llm.invoke(messages, **kwargs)
        return self.last_response


def create_router_probe(llm):
    """调用真实路由方法，但不执行实际检索或记忆操作。"""
    probe = PDFLearningAssistant.__new__(PDFLearningAssistant)
    route = {}

    def fake_recall(question):
        route["selected"] = "MEMORY"
        return ""

    def fake_ask(question):
        route["selected"] = "RAG"
        return ""

    probe.recall = fake_recall
    probe.ask = fake_ask
    probe.rag_tool = SimpleNamespace(llm=ObservedLLM(llm))
    return probe, route


def evaluate_routing():
    test_set = load_routing_test_set()
    llm = HelloAgentsLLM()
    router_probe, selected_route = create_router_probe(llm)

    results = []
    for item in test_set:
        question = item["question"]
        expected = item["expected_route"]

        selected_route.clear()
        keyword_start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            router_probe.smart_ask(question)
        keyword_elapsed = time.perf_counter() - keyword_start
        route_kw = selected_route.get("selected")

        result = {
            "id": item["id"],
            "question": question,
            "expected_route": expected,
            "route_kw": route_kw,
            "correct_kw": route_kw == expected,
            "elapsed_kw": keyword_elapsed,
        }

        selected_route.clear()
        router_probe.rag_tool.llm.last_response = None
        llm_start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            router_probe.smart_ask_llm(question)
        fallback = getattr(router_probe, "_last_llm_route_fallback", False)
        route_llm = getattr(router_probe, "_last_llm_route", None)
        result.update({
            "route_llm": route_llm,
            "llm_response": router_probe.rag_tool.llm.last_response,
            "correct_llm": route_llm == expected and not fallback,
            "elapsed_llm": time.perf_counter() - llm_start,
            "llm_status": "failed" if fallback else "success",
        })
        if fallback:
            result["llm_error"] = getattr(
                router_probe,
                "_last_llm_route_error",
                "LLM 路由失败或返回格式无效，已降级到 RAG",
            )
            print(f"[LLM 路由失败] {item['id']}: {result['llm_error']}")

        results.append(result)

    total = len(results)
    keyword_correct = sum(result["correct_kw"] for result in results)
    llm_successes = [
        result for result in results if result["llm_status"] == "success"
    ]
    llm_correct = sum(result["correct_llm"] for result in llm_successes)

    print("=" * 100)
    print("智能路由准确率测试结果")
    print("=" * 100)
    print(
        f"{'ID':<5} | {'预期':<8} | {'关键词路由':<12} | "
        f"{'LLM路由':<12} | 问题"
    )
    print("-" * 100)
    for result in results:
        keyword_icon = "✅" if result["correct_kw"] else "❌"
        if result["llm_status"] == "success":
            llm_icon = "✅" if result["correct_llm"] else "❌"
            llm_route = result["route_llm"]
        else:
            llm_icon = "⚠️"
            llm_route = "ERROR"
        print(
            f"{result['id']:<5} | {result['expected_route']:<8} | "
            f"{keyword_icon} {str(result['route_kw']):<8} | "
            f"{llm_icon} {llm_route:<8} | {result['question'][:30]}"
        )

    print("-" * 100)
    print(
        f"关键词路由准确率: {keyword_correct}/{total} = "
        f"{keyword_correct / total * 100:.1f}%"
    )
    if llm_successes:
        print(
            f"LLM 路由准确率:  {llm_correct}/{len(llm_successes)} = "
            f"{llm_correct / len(llm_successes) * 100:.1f}% "
            f"(调用成功 {len(llm_successes)}/{total})"
        )
    else:
        print(f"LLM 路由准确率: 无成功调用 (0/{total})")

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存到: {RESULTS_PATH}")


if __name__ == "__main__":
    evaluate_routing()
