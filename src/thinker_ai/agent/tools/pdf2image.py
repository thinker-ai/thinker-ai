import fitz  # PyMuPDF
import os


def pdf_to_jpg(pdf_path, output_folder):
    # 打开 PDF 文件
    pdf_document = fitz.open(pdf_path)

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历每一页
    for page_num in range(len(pdf_document)):
        # 获取页面
        page = pdf_document.load_page(page_num)

        # 将页面渲染为 pixmap
        pix = page.get_pixmap()

        # 定义输出图像文件路径
        output_image_path = os.path.join(output_folder, f'page_{page_num + 1}.jpg')

        # 将 pixmap 保存为 JPG 文件
        pix.save(output_image_path)

    print(f'PDF 文件已成功导出到 {output_folder}')

if __name__ == '__main__':
    # 示例调用
    pdf_path = '/Users/wangli/Library/Mobile Documents/com~apple~CloudDocs/注册公司/赢睿科技（海南）有限公司章程/章程.pdf'
    output_folder = '/Users/wangli/Library/Mobile Documents/com~apple~CloudDocs/注册公司/赢睿科技（海南）有限公司章程/章程/'
    pdf_to_jpg(pdf_path, output_folder)
