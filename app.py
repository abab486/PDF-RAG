import json
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))

from assistant import PDFLearningAssistant

# 全局助手实例
assistant = PDFLearningAssistant(user_id="demo_user")

def upload_pdf(file):
    if file is None:
        return "请上传 PDF 文件"
    result = assistant.load_document(file.name)
    return result["message"]


def chat(question, history):
    if not question:
        return history, ""
    
    answer = assistant.ask(question)
    
    # 确保 history 是列表
    if history is None:
        history = []
    
    # 用 messages 格式追加
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    
    return history, ""


def add_note(content):
    assistant.add_note(content)
    return "笔记已保存"


def recall(query):
    return assistant.recall(query)


def stats():
    s = assistant.get_stats()
    return "\n".join(f"{k}: {v}" for k, v in s.items())


def report():
    r = assistant.generate_report()
    return json.dumps(r, ensure_ascii=False, indent=2, default=str)


with gr.Blocks(title="智能文档问答助手") as demo:
    gr.Markdown("# 📚 智能文档问答助手")
    
    with gr.Tab("📄 文档加载"):
        file_input = gr.File(label="上传 PDF", file_types=[".pdf"])
        upload_btn = gr.Button("加载")
        upload_output = gr.Textbox(label="结果")
        upload_btn.click(upload_pdf, inputs=file_input, outputs=upload_output)
    
    with gr.Tab("💬 智能问答"):
        chatbot = gr.Chatbot(label="对话")
        question_input = gr.Textbox(label="提问")
        ask_btn = gr.Button("提问")
        ask_btn.click(chat, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
    
    with gr.Tab("📝 学习笔记"):
        note_input = gr.Textbox(label="笔记内容")
        note_btn = gr.Button("保存")
        note_output = gr.Textbox(label="结果")
        note_btn.click(add_note, inputs=note_input, outputs=note_output)
    
    with gr.Tab("🔍 学习回顾"):
        recall_input = gr.Textbox(label="查询")
        recall_btn = gr.Button("搜索")
        recall_output = gr.Textbox(label="结果")
        recall_btn.click(recall, inputs=recall_input, outputs=recall_output)
    
    with gr.Tab("📊 统计与报告"):
        stats_btn = gr.Button("查看统计")
        stats_output = gr.Textbox(label="统计")
        stats_btn.click(stats, outputs=stats_output)
        
        report_btn = gr.Button("生成报告")
        report_output = gr.Textbox(label="报告")
        report_btn.click(report, outputs=report_output)


if __name__ == "__main__":
    demo.launch()
    