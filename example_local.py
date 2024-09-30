# 使用本地数据进行可视化，在本地可视化
from Taiyi.visualize import LocalVisualization 

if __name__ == '__main__':
    project = 'BatchNorm2d'
    localvis = LocalVisualization(project=project)
    # 这个quantity_name就是example中要监测的指标的名称
    quantity_name = 'BatchNorm2d'
    localvis.show(quantity_name=quantity_name)