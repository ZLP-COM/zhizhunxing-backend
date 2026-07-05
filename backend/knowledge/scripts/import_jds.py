"""职准星 - 一键导入JD脚本

从 docx 文件中提取JD并保存为txt文件

使用方法：
    python -m knowledge.scripts.import_jds <docx_file_or_dir>
"""

import os
import sys
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from docx import Document
except ImportError:
    logger.error("请先安装 python-docx: pip install python-docx")
    sys.exit(1)

from core import config


def extract_jds_from_docx(docx_path: str, output_dir: str = None) -> int:
    """
    从 docx 文件提取JD并保存为txt

    Args:
        docx_path: docx文件路径
        output_dir: 输出目录

    Returns:
        提取的JD数量
    """
    output_dir = output_dir or str(config.JDS_DIR)
    os.makedirs(output_dir, exist_ok=True)

    doc = Document(docx_path)
    all_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    # 按公司名称分割
    lines = all_text.split("\n")
    current_jd = []
    jd_count = 0
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".txt") and f.startswith("JD_")])

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("公司") and ("：" in line or ":" in line) and current_jd:
            content = "\n".join(current_jd)
            if len(content) > 200:
                jd_count += 1
                fname = f"JD_{existing + jd_count:02d}.txt"
                fpath = os.path.join(output_dir, fname)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"  已保存: {fname}")
            current_jd = [line]
        else:
            current_jd.append(line)

    if current_jd:
        content = "\n".join(current_jd)
        if len(content) > 200:
            jd_count += 1
            fname = f"JD_{existing + jd_count:02d}.txt"
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"  已保存: {fname}")

    return jd_count


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="职准星 - 一键导入JD脚本")
    parser.add_argument("source", nargs="?", help="docx文件路径或目录路径")
    parser.add_argument("--output", "-o", default=None, help="输出目录")
    args = parser.parse_args()

    source = args.source
    if not source:
        # 默认扫描当前目录和知识库目录
        source = os.path.join(os.path.dirname(os.path.dirname(config.KNOWLEDGE_DIR)), "..", "..", "..")
        logger.info(f"未指定源文件，扫描: {source}")

    output_dir = args.output or str(config.JDS_DIR)
    logger.info(f"输出目录: {output_dir}")

    total = 0

    if os.path.isfile(source):
        if source.endswith(".docx"):
            count = extract_jds_from_docx(source, output_dir)
            total += count
            logger.info(f"从 {os.path.basename(source)} 提取 {count} 份JD")
    elif os.path.isdir(source):
        for fname in sorted(os.listdir(source)):
            if fname.endswith(".docx") and "JD" in fname:
                fpath = os.path.join(source, fname)
                count = extract_jds_from_docx(fpath, output_dir)
                total += count
                logger.info(f"从 {fname} 提取 {count} 份JD")

    logger.info(f"共导入 {total} 份JD到 {output_dir}")


if __name__ == "__main__":
    main()
