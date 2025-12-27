# webui_research.py
import gradio as gr
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import base64
from io import BytesIO, StringIO
import os
import sys
from ATF.core.adaptoflux import AdaptoFlux

# === 2. 全局状态容器（研究用，简单）===
class GlobalState:
    def __init__(self):
        self.af = None  # AdaptoFlux 实例

state = GlobalState()

# === 3. 工具函数 ===

def plot_graph_to_html(G):
    if G is None or G.number_of_nodes() == 0:
        return "<p>图为空</p>"
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color="lightblue", font_size=8)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{img_b64}" />'

def parse_input_data(data_input, input_format):
    try:
        if input_format == "CSV":
            df = pd.read_csv(StringIO(data_input))
            values = df.iloc[:, :-1].values  # 假设最后一列是标签
            labels = df.iloc[:, -1].values
        elif input_format == "NumPy (space sep)":
            lines = [list(map(float, line.split())) for line in data_input.strip().split('\n')]
            arr = np.array(lines)
            values = arr[:, :-1]
            labels = arr[:, -1]
        else:  # Manual
            values = np.array(eval(data_input))
            labels = np.zeros(values.shape[0])  # dummy
        return values, labels, None
    except Exception as e:
        return None, None, str(e)

# === 4. Gradio 回调函数 ===

def load_methods(file_obj):
    if not file_obj:
        return "请上传方法文件", None
    state.af = AdaptoFlux(methods_path=file_obj.name)
    method_names = list(state.af.methods.keys())
    return f"加载 {len(method_names)} 个方法", gr.Dropdown(choices=method_names)

def init_graph(data_input, input_format, collapse_method="SUM"):
    if state.af is None:
        return "请先加载方法文件", None
    values, labels, err = parse_input_data(data_input, input_format)
    if err:
        return f"解析失败: {err}", None
    # 重建 AdaptoFlux（带数据）
    state.af = AdaptoFlux(
        values=values,
        labels=labels,
        methods_path=state.af.methods_path,
        collapse_method=getattr(CollapseMethod, collapse_method)
    )
    return "图初始化成功", plot_graph_to_html(state.af.graph)

def train_one_layer():
    if state.af is None:
        return "未初始化", None, ""
    try:
        result = state.af.process_random_method()
        state.af.append_nx_layer(result)
        acc = state.af.infer_with_graph(state.af.values)
        # 简化：假设返回的是预测标签，计算准确率
        pred_labels = (acc > 0.5).astype(int)  # 示例：二分类
        acc_score = np.mean(pred_labels == state.af.labels)
        state.af.metrics['accuracy'] = acc_score
        return f"添加新层，当前准确率: {acc_score:.4f}", plot_graph_to_html(state.af.graph), str(result)
    except Exception as e:
        return f"训练失败: {str(e)}", plot_graph_to_html(state.af.graph), traceback.format_exc()

def replace_node_method(node_id, new_method):
    if state.af is None:
        return "未初始化", None
    if node_id not in state.af.graph.nodes:
        return "节点不存在", plot_graph_to_html(state.af.graph)
    if new_method not in state.af.methods:
        return "方法不存在", plot_graph_to_html(state.af.graph)
    # 替换节点方法（需修改 graph_processor.graph 中节点属性）
    state.af.graph_processor.graph.nodes[node_id]['method_name'] = new_method
    return "替换成功", plot_graph_to_html(state.af.graph)

def infer_batch(data_input, input_format):
    if state.af is None:
        return "未初始化", ""
    values, _, err = parse_input_data(data_input, input_format)
    if err:
        return f"解析失败: {err}", ""
    try:
        preds = state.af.infer_with_graph(values)
        return "推理完成", str(preds.tolist())
    except Exception as e:
        return f"推理失败: {str(e)}", ""

def save_model():
    if state.af is None:
        return "未初始化"
    try:
        state.af.save_model(folder="saved_model")
        return "模型已保存到 ./saved_model"
    except Exception as e:
        return f"保存失败: {str(e)}"

# === 5. Gradio UI ===

with gr.Blocks(title="AdaptoFlux 研究 WebUI") as demo:
    gr.Markdown("## 🌊 AdaptoFlux 池流算法 - 研究用 WebUI")

    with gr.Tab("1. 加载方法"):
        method_file = gr.File(label="上传 methods.py", file_types=[".py"])
        load_btn = gr.Button("加载方法")
        load_status = gr.Textbox()
        method_dropdown = gr.Dropdown(label="可用方法（供替换用）", interactive=False)

        load_btn.click(load_methods, method_file, [load_status, method_dropdown])

    with gr.Tab("2. 初始化图"):
        with gr.Row():
            input_format = gr.Radio(["CSV", "NumPy (space sep)", "Manual (Python list)"], value="CSV", label="输入格式")
        data_input = gr.Textbox(label="输入数据", lines=5, value="x1,x2,y\n1,2,0\n3,4,1\n5,6,1")
        collapse_method = gr.Radio(["SUM", "MEAN", "MAX"], value="SUM", label="坍缩方法")
        init_btn = gr.Button("初始化图结构")
        init_status = gr.Textbox()
        graph_display = gr.HTML()

        init_btn.click(init_graph, [data_input, input_format, collapse_method], [init_status, graph_display])

    with gr.Tab("3. 训练"):
        train_btn = gr.Button("添加一层（训练）")
        train_status = gr.Textbox()
        train_graph = gr.HTML()
        train_log = gr.Textbox(label="Layer Result", lines=3)

        train_btn.click(train_one_layer, None, [train_status, train_graph, train_log])

    with gr.Tab("4. 替换节点方法"):
        node_id_input = gr.Textbox(label="节点ID（如 1_0_add）")
        replace_method = gr.Dropdown(label="新方法", choices=[])
        replace_btn = gr.Button("替换")
        replace_status = gr.Textbox()
        replace_graph = gr.HTML()

        # 动态更新方法下拉框
        demo.load(lambda: gr.Dropdown(choices=list(state.af.methods.keys()) if state.af else []), None, replace_method)
        replace_btn.click(replace_node_method, [node_id_input, replace_method], [replace_status, replace_graph])

    with gr.Tab("5. 推理"):
        infer_data = gr.Textbox(label="推理输入（同初始化格式）", lines=3)
        infer_format = gr.Radio(["CSV", "NumPy (space sep)", "Manual"], value="CSV")
        infer_btn = gr.Button("推理")
        infer_status = gr.Textbox()
        infer_result = gr.Textbox(label="结果", lines=3)

        infer_btn.click(infer_batch, [infer_data, infer_format], [infer_status, infer_result])

    with gr.Tab("6. 保存/加载"):
        save_btn = gr.Button("保存模型")
        save_status = gr.Textbox()
        save_btn.click(save_model, None, save_status)
        gr.Markdown("模型保存在 `./saved_model`，可手动加载（当前版本暂未实现加载UI，但支持 `af.load_model()`）")

# === 6. 启动 ===
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)