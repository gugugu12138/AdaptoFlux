# webui.py
import gradio as gr
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import ATF

# 全局状态（研究用，可接受）
method_pool = {}      # {name: func}
current_graph = None  # NetworkX DiGraph 或自定义图对象
trained_model = None

# 1. 导入方法池
def upload_method_file(file_obj):
    # 1. 动态加载 .py 文件（需定义 @method_profile 装饰器）
    # 2. 解析出所有带装饰器的方法
    # 3. 存入 method_pool
    return f"Loaded {len(method_pool)} methods."

# 2. 查看方法池
def show_method_pool():
    return "\n".join(method_pool.keys())

# 3. 构建/重置图结构
def build_graph(initial_layers=3):
    global current_graph
    # 调用你的 LayerGrow 逻辑，生成初始图
    current_graph = create_initial_graph(initial_layers)
    return plot_graph(current_graph)

# 4. 可视化图（Matplotlib → base64 图像）
def plot_graph(G):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return f'<img src="data:image/png;base64,{img_str}" />'

# 5. 替换图中某个节点的方法
def replace_node_method(node_id, new_method_name):
    if node_id in current_graph.nodes and new_method_name in method_pool:
        current_graph.nodes[node_id]['method'] = method_pool[new_method_name]
        return "Success", plot_graph(current_graph)
    return "Error", plot_graph(current_graph)

# 6. 训练
def train_graph(epochs=10):
    global trained_model
    # 调用你的 GraphEvo + 遗传算法训练逻辑
    trained_model = train_adaptoflux(current_graph, epochs)
    return "Training finished."

# 7. 推理
def inference(input_data):
    if trained_model is None:
        return "Model not trained!"
    result = trained_model.forward(input_data)
    # 可选：高亮推理路径，并返回带路径的图
    return str(result), plot_graph_with_path(current_graph, result.path)

# ===== Gradio UI 布局 =====
with gr.Blocks(title="AdaptoFlux WebUI") as demo:
    gr.Markdown("## AdaptoFlux: 池流算法研究 WebUI")

    with gr.Tab("方法池"):
        file_input = gr.File(label="上传方法文件 (.py)")
        upload_btn = gr.Button("导入方法")
        method_list = gr.Textbox(label="已加载方法", lines=5)
        upload_btn.click(upload_method_file, file_input, method_list)

    with gr.Tab("图结构"):
        layers = gr.Slider(1, 10, value=3, label="初始层数")
        build_btn = gr.Button("构建初始图")
        graph_display = gr.HTML()
        build_btn.click(build_graph, layers, graph_display)

        with gr.Group():
            node_id = gr.Textbox(label="节点ID（如 '1_0'）")
            method_name = gr.Dropdown(choices=[], label="新方法", interactive=True)
            replace_btn = gr.Button("替换方法")
            replace_status = gr.Textbox()
            # 动态更新方法下拉框
            demo.load(lambda: gr.Dropdown(choices=list(method_pool.keys())), None, method_name)
            replace_btn.click(replace_node_method, [node_id, method_name], [replace_status, graph_display])

    with gr.Tab("训练"):
        epochs = gr.Number(value=10, label="训练轮数")
        train_btn = gr.Button("开始训练")
        train_log = gr.Textbox()
        train_btn.click(train_graph, epochs, train_log)

    with gr.Tab("推理"):
        input_data = gr.Textbox(label="输入（JSON 或列表）")
        infer_btn = gr.Button("推理")
        output = gr.Textbox()
        infer_btn.click(inference, input_data, [output, graph_display])

# 启动
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)