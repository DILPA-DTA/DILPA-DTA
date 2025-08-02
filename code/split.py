import os


def rename_files_in_directory(directory_path):
    # 获取目录中的所有文件
    for filename in os.listdir(directory_path):
        # 确保文件是Word文件（.docx）
        # 分割文件名
        name_parts = filename.split('-')
        if len(name_parts) > 2:
            # 只保留第二个`-`之后的部分
            new_name = '-'.join(name_parts[2:])
            # 拼接新的文件路径
            old_file_path = os.path.join(directory_path, filename)
            new_file_path = os.path.join(directory_path, new_name)

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {filename} -> {new_name}')


# 使用该函数时传入目标文件夹路径
directory_path = 'C:/Users/yangj/Desktop/软件2308班web'  # 替换为实际路径
rename_files_in_directory(directory_path)
