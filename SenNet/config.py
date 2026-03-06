import os
from fileinput import filename
from os.path import join
from typing import Optional

# 参数设置。
max_cpu_cnt = 6
max_thread_cnt = 10
drop_out_rate = None # 控制nnUNet各个卷积层是否使用dropout

def get_base_folder():
    base_folder = join('/data', 'zr')  # .98 服务器。
    return base_folder

my_folder = get_base_folder()
predict_folder = join(my_folder, 'predict')  # 存放模型预测结果。
predict_train_folder = join(my_folder, 'predict_train')  # 存放模型预测结果。
val_id_set = []  # 测试集的 id。
test_id_set = []  # 测试集的 id。

# 任务设置相关
config_list = []

class BaseConfig:
    def __init__(self, task_name: str, task_patch_size: tuple, label_map: dict, folder_name : Optional[str] = None, region_label=False):
        self.task_name = task_name
        self.patch_size = task_patch_size
        self.label_map = label_map
        self.folder_name = folder_name if folder_name else task_name
        self.base_folder = join(my_folder, self.folder_name)  # 项目数据的根路径。
        self.region_label = region_label  # 一个实质占多个标签（如KiTS2023）。
        self.black_list = []
        config_list.append(self)

    def get_label_path(self, image_filename):
        raise NotImplementedError('gt_folder() is not implemented.')

class AblationConfig(BaseConfig):
    """
        消融区域的配置
    """
    def __init__(self):
        super().__init__(task_name='MyAblation', task_patch_size=(64, 128, 128), label_map={
            '0': 'background',
            '1': 'Ablation',
        })
        self.my_image_folder = join(self.base_folder, 'image')
        self.my_label_folder = join(self.base_folder, 'label')
        self.my_split_folder = join(self.base_folder, 'test_split')

    def get_label_path(self, image_filename):
        return join(self.my_label_folder, image_filename)



ablation_config = AblationConfig()
main_config: BaseConfig = ablation_config  # 主配置。

patch_size = main_config.patch_size
label_map = main_config.label_map


