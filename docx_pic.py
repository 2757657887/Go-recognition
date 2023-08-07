from docx import Document
import os

# 指定包含Word文档的文件夹路径
input_folder = './1-300'

# 指定保存提取的图片的文件夹路径
output_folder = './pic_all'

# 遍历文件夹中的每个Word文档
for index,filename in enumerate(os.listdir(input_folder)):
    print(filename)
    if filename.endswith('.docx'):
        doc_path = os.path.join(input_folder, filename)
        doc = Document(doc_path)

        # 遍历文档中的每个图像
        for rel in doc.part.rels:
            if "image" in doc.part.rels[rel].target_ref:
                image_part = doc.part.rels[rel].target_part
                image_bytes = image_part.blob

                # 创建输出路径并保存图片
                output_path = os.path.join(output_folder,
                                           f'{filename[:3]}_{rel}_{index}.png')
                with open(output_path, 'wb') as f:
                    f.write(image_bytes)