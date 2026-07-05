"""职准星 - 知识库导入脚本

使用方法：
    python -m knowledge.scripts.import_kb

从 docx 源文件导入知识库到 ChromaDB
"""

import os
import sys
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.kb_manager import KnowledgeBaseManager, init_knowledge_base, kb_manager
from core import config


async def main():
    """主入口"""
    logger.info("=" * 50)
    logger.info("职准星 - 知识库导入脚本")
    logger.info("=" * 50)

    # 检查知识库状态
    if kb_manager.is_available():
        logger.info(f"ChromaDB 可用，路径: {config.CHROMA_DB_PATH}")
    else:
        logger.warning("ChromaDB 初始化失败，将使用文件降级方案")

    # 初始化知识库
    success = await init_knowledge_base(kb_manager)

    if success:
        stats = {
            "judgment_rules": kb_manager.count("judgment_rules"),
            "jds": kb_manager.count("jds"),
            "holland_guide": kb_manager.count("holland_guide"),
        }
        logger.info(f"知识库导入完成: {stats}")
    else:
        logger.warning("知识库导入失败或部分完成")

    logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
