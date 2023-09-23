import docx


def read_docx(file_path: str) -> list:
    """打开docx文件"""
    doc = docx.Document(file_path)

    # 创建一个空列表，用于存储段落内容
    paragraphs_list = []

    # 遍历文档中的段落，并将其内容添加到列表中
    for paragraph in doc.paragraphs:
        paragraphs_list.append(paragraph.text)

    return paragraphs_list
