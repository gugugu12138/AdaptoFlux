import os
import sys
from pathlib import Path
from cairosvg import svg2pdf

def convert_svg_to_pdf(input_dir: str):
    """
    将指定目录下所有 .svg 文件转换为同名 .pdf 文件。
    跳过已存在的 PDF 文件。
    """
    input_path = Path(input_dir).resolve()
    if not input_path.is_dir():
        raise ValueError(f"指定路径不是有效目录: {input_path}")

    svg_files = list(input_path.glob("*.svg"))
    if not svg_files:
        print(f"⚠️  在 {input_path} 中未找到任何 .svg 文件。")
        return

    print(f"📁 找到 {len(svg_files)} 个 SVG 文件，开始转换为 PDF...")
    
    for svg_file in svg_files:
        pdf_file = svg_file.with_suffix('.pdf')
        if pdf_file.exists():
            print(f"⏭️  跳过（PDF 已存在）: {pdf_file.name}")
            continue

        try:
            svg2pdf(url=str(svg_file), write_to=str(pdf_file))
            print(f"✅ 转换成功: {svg_file.name} → {pdf_file.name}")
        except Exception as e:
            print(f"❌ 转换失败: {svg_file.name} | 错误: {e}")

    print("🎉 所有转换任务完成！")

if __name__ == "__main__":
    # 默认转换当前目录下的 SVG
    default_dir = "experiments/PipelineParallel_exp2/results/flamegraph"
    
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = default_dir
        print(f"📢 未指定目录，使用默认路径: {target_dir}")
    
    convert_svg_to_pdf(target_dir)