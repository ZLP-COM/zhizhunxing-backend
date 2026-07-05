'''职准星 - PDF 解析模块（基于 PyMuPDF，零本地 RPC 服务依赖）

核心特性：
  1. 仅依赖 PyMuPDF（fitz），无外部 RPC/127.0.0.1 端口调用
  2. 统一入口 parse_pdf_bytes()，接收二进制 PDF 流
  3. 自动区分普通文字 PDF 与扫描图片 PDF
  4. 过滤乱码、换行冗余、空白行，输出干净简历纯文本
  5. 读取 config.MAX_FILE_SIZE 校验文件大小，拦截超大附件
  6. 捕获文件损坏/加密/空文件异常，返回友好兜底文案
'''

import io
import os
import re
import logging
from typing import Optional

import fitz  # PyMuPDF

from ..core import config

logger = logging.getLogger(__name__)


# ============================================================
# 自定义异常
# ============================================================

class PDFParseError(Exception):
    '''PDF 解析异常（含友好提示文本，上层 catch 后展示给用户）'''
    pass


# ============================================================
# 统一入口：接收二进制流
# ============================================================

def parse_pdf_bytes(pdf_bytes: bytes, filename: str = '') -> str:
    '''
    统一 PDF 解析入口函数，接收二进制文件流。

    内部流程：
      1. 文件大小校验（读取 config.MAX_FILE_SIZE）
      2. 文件头 magic number 校验
      3. PyMuPDF 解析
      4. 扫描件检测 → 返回降级提示
      5. 文本清理（过滤乱码/冗余换行/空白行）
      6. 长度限制

    Args:
        pdf_bytes: PDF 文件的二进制内容
        filename:  原文件名（仅用于日志）

    Returns:
        解析后的简历纯文本（干净、格式化）

    Raises:
        PDFParseError: 解析失败时抛出友好提示
    '''
    # ---- 1. 文件大小校验 ----
    file_size = len(pdf_bytes)
    if file_size > config.MAX_FILE_SIZE:
        raise PDFParseError(
            f'文件过大（{file_size / 1024 / 1024:.1f}MB，超过 {config.MAX_FILE_SIZE / 1024 / 1024:.0f}MB 限制），'
            '请压缩后上传或直接粘贴简历文本'
        )
    if file_size == 0:
        raise PDFParseError('文件为空，请重新上传或直接粘贴简历文本')

    # ---- 2. PDF 文件头校验 ----
    if not pdf_bytes.startswith(b'%PDF'):
        raise PDFParseError('文件格式不正确，目前仅支持 PDF 格式')

    # ---- 3. PyMuPDF 解析 ----
    try:
        text = _extract_text_from_bytes(pdf_bytes)
    except fitz.FileDataError:
        raise PDFParseError(
            'PDF 文件已损坏，无法解析，建议直接粘贴简历文本'
        )
    except Exception as e:
        error_msg = str(e)[:80]
        logger.error(f'PDF 解析异常: {error_msg}')
        raise PDFParseError(
            f'PDF 解析失败（{error_msg}），建议直接粘贴简历文本'
        )

    # ---- 4. 扫描件检测 ----
    if not text or len(text.strip()) < 50:
        # 尝试二次确认是否为扫描件
        if _is_scanned_pdf(pdf_bytes):
            raise PDFParseError(
                '检测到扫描图片型 PDF（无文字层），无法提取文本，请直接粘贴简历文本'
            )
        raise PDFParseError(
            'PDF 内容为空或文字过少，请确认 PDF 是否包含文字内容，或直接粘贴简历文本'
        )

    # ---- 5. 文本清理 ----
    text = _clean_text(text)

    # ---- 6. 长度限制 ----
    if len(text) > config.MAX_RESUME_LENGTH:
        logger.info(
            f'PDF 文本过长（{len(text)}字），截断至 {config.MAX_RESUME_LENGTH}'
        )
        text = text[:config.MAX_RESUME_LENGTH]

    logger.info(
        f'PDF 解析成功 | 文件={filename or "bytes"} | 大小={file_size / 1024:.1f}KB | 字数={len(text)}'
    )
    return text


# ============================================================
# 兼容旧接口：接收文件路径
# ============================================================

def safe_parse_pdf(file_path: str) -> str:
    '''
    兼容旧接口的 PDF 解析入口，接收文件路径。

    内部调用 parse_pdf_bytes()，先读取文件再解析。

    Args:
        file_path: PDF 文件的磁盘路径

    Returns:
        解析后的简历纯文本

    Raises:
        PDFParseError: 解析失败时抛出友好提示
    '''
    if not os.path.exists(file_path):
        raise PDFParseError('文件不存在，请重新上传')

    with open(file_path, 'rb') as f:
        pdf_bytes = f.read()

    filename = os.path.basename(file_path)
    return parse_pdf_bytes(pdf_bytes, filename=filename)


# ============================================================
# 内部解析：二进制流 → 原始文本
# ============================================================

def _extract_text_from_bytes(pdf_bytes: bytes) -> str:
    '''
    使用 PyMuPDF 从二进制流中提取文本。

    优先使用 page.get_text('text') 提取，
    若某页无文本则回退到 page.get_text('dict') 逐行提取。
    '''
    text_parts: list[str] = []
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')

    try:
        # 检查是否加密
        if doc.is_encrypted:
            raise PDFParseError('PDF 文件已加密，无法解析，请去除密码保护后重试')

        for page_num in range(len(doc)):
            page = doc[page_num]

            # 方式一：直接提取文本
            text = page.get_text('text')
            if text and text.strip():
                text_parts.append(text)
                continue

            # 方式二：dict 模式逐行提取（适配部分格式异常 PDF）
            blocks = page.get_text('dict').get('blocks', [])
            page_lines: list[str] = []
            for block in blocks:
                if block.get('type') == 0:  # 文字块
                    for line in block.get('lines', []):
                        line_text = ' '.join(
                            span.get('text', '') for span in line.get('spans', [])
                        )
                        if line_text.strip():
                            page_lines.append(line_text)
            if page_lines:
                text_parts.append('\n'.join(page_lines))
    finally:
        doc.close()

    return '\n'.join(text_parts)


# ============================================================
# 文本清理
# ============================================================

def _clean_text(text: str) -> str:
    '''
    清理提取出的原始文本，输出干净简历纯文本。

    清理规则：
      1. 移除控制字符（保留常见可见字符和换行）
      2. 压缩连续空行（>=3 → 2）
      3. 压缩连续空格/Tab 为单个空格
      4. 移除全空白行
      5. 首尾修剪
    '''
    # 移除控制字符（保留 \n \r \t 以及可见字符）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 压缩连续空行为最多 2 个换行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 压缩连续空格/Tab
    text = re.sub(r'[ \t]+', ' ', text)

    # 按行处理，移除全空白行
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]

    # 恢复段落间距：每个自然段之间保留空行
    cleaned: list[str] = []
    for i, line in enumerate(lines):
        if cleaned and not cleaned[-1]:
            cleaned.append(line)
        elif line.endswith(('。', '！', '？', ':', '：', ';', '；', '.', '!', '?')):
            cleaned.append(line)
            cleaned.append('')  # 句号后插入空行分段
        else:
            cleaned.append(line)

    return '\n'.join(cleaned).strip()


# ============================================================
# 扫描件检测
# ============================================================

def _is_scanned_pdf(pdf_bytes: bytes) -> bool:
    '''
    快速判断 PDF 是否为扫描图片型 PDF（无文字层）。

    判断逻辑：字符总数 < 页面数 × 50 且页面包含图片。
    '''
    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        try:
            total_chars = 0
            has_image = False
            for page in doc:
                text = page.get_text('text')
                total_chars += len(text.strip())
                # 检查页面是否包含图片
                images = page.get_images(full=True)
                if images:
                    has_image = True
            # 标准：字符极少 + 包含图片 = 扫描件
            return total_chars < len(doc) * 50 and has_image
        finally:
            doc.close()
    except Exception:
        return False


def is_scanned_pdf(file_path: str) -> bool:
    '''兼容旧接口：通过文件路径判断是否为扫描件'''
    try:
        with open(file_path, 'rb') as f:
            return _is_scanned_pdf(f.read())
    except Exception:
        return False


# ============================================================
# 工具函数
# ============================================================

def validate_pdf_extension(filename: str) -> bool:
    '''校验文件扩展名是否为 .pdf'''
    return filename.lower().endswith('.pdf')


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    '''获取 PDF 页数'''
    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        try:
            return len(doc)
        finally:
            doc.close()
    except Exception:
        return 0


# ============================================================
# 直接运行时测试
# ============================================================

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        try:
            text = safe_parse_pdf(filepath)
            print(f'解析成功，共 {len(text)} 字')
            print('--- 前 500 字 ---')
            print(text[:500])
        except PDFParseError as e:
            print(f'解析失败: {e}')
    else:
        print('用法: python pdf_parser.py <pdf_file_path>')
