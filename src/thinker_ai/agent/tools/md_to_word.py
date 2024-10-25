import os


def convert_md_to_docx(md_file_path):
    """
    将单个 Markdown 文件转换为 DOCX 文件。

    参数:
    md_file_path (str): Markdown 文件的完整路径。

    返回:
    None
    """
    try:
        # 确保传入的是一个文件路径
        if not os.path.isfile(md_file_path):
            print(f"错误: {md_file_path} 不是一个有效的文件路径。")
            return

        # 构建 DOCX 文件路径
        docx_file_path = os.path.splitext(md_file_path)[0] + '.docx'

        # 构建 pandoc 命令
        command = f'pandoc "{md_file_path}" -o "{docx_file_path}"'

        # 执行命令
        result = os.popen(command).read()

        # 检查转换结果
        if not result:
            print(f"已成功转换: {md_file_path} -> {docx_file_path}")
        else:
            print(f"转换 {md_file_path} 时出现问题: {result}")

    except Exception as e:
        print(f"转换 {md_file_path} 时发生异常: {e}")


def batch_convert_md_to_docx(directory):
    """
    遍历指定目录及其子目录中的所有 Markdown 文件，并将其转换为 DOCX 文件。

    参数:
    directory (str): 需要遍历的目录路径。

    返回:
    None
    """
    # 存储所有 Markdown 文件的完整路径
    md_files = []

    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.md'):
                full_path = os.path.join(root, file)
                md_files.append(full_path)

    print(f"找到 {len(md_files)} 个 Markdown 文件。")

    # 逐个转换
    for md_file in md_files:
        convert_md_to_docx(md_file)


# 示例用法
if __name__ == "__main__":
    # 或者单独转换一个文件
    single_md_file = "/Users/wangli/Desktop/禅修项目运营.md"
    convert_md_to_docx(single_md_file)

