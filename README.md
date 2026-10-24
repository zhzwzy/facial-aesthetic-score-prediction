# 美学评分预测源码

本仓库整理了 50 项和 57 项测量特征的美学评分预测流程，包含数据构建、特征计算、模型训练、评估和可视化代码。仓库不包含任何原始或派生数据、图像、标注、评分、训练结果或模型文件。

## 运行环境

建议使用 Python 3.9，并安装依赖：

```powershell
pip install -r requirements.txt
```

## 本地数据结构

数据需要由使用者自行准备并放入 `inputs/`。该目录已加入 `.gitignore`，其中的文件不会上传。

```text
inputs/
  base_data/
    female.xlsx
    male.xlsx
  landmarks/
    prompt_generation/
      pre.csv
      post.csv
    core/
      pre.csv
      post.csv
      rotation.csv
  new_male_annotations/
    <annotation files>.json
  labels/
    supplementary_dataset.csv
```

新增 JSON 标注应包含标志点 `points` 和自然头位旋转角 `rotation`；评分表应包含姓名、阶段和评分字段。原始数据不随本仓库提供。

## 构建与训练

在仓库根目录运行：

```powershell
python code/build_dataset.py
python code/train_models.py --features all
```

构建脚本会在本地生成 50 项和 57 项特征数据；训练脚本分别运行两组特征。也可以用 `--features 50` 或 `--features 57` 单独运行。生成的派生数据和训练输出均保存在本地，并由 `.gitignore` 排除。

Sn-UL 按自然头位下的有符号垂直距离计算：

```text
(y_UL - y_Sn) / (distance(R1, R2) / 10)
```

图像坐标的 y 轴向下为正，因此 UL 在 Sn 下方时该值为正。
