# Figure 2：枢纽节点示例

这个目录可整体复制到任意位置独立运行；所有运行输入、程序和输出均在本目录。公共实现使用各图内的独立副本，不导入其他 figure 或 `cache`。六个目录保持一致的命名规则。

## 常用命令

在本目录执行（也支持从其他目录使用脚本绝对路径）：

```sh
python -m pip install -r requirements.txt
python plot.py
python validate.py
```

`plot.py` 只读取已有数据，生成 `figure_2.pdf`、`figure_2.png` 与 `build.json`，保持当前论文图形的颜色和版式。`validate.py` 核验本地依赖、构建哈希与对应数据。

## 文件组织

- `plot.py`：绘图入口；`build_record.py`：本图输入、源码和输出的哈希记录。
- `paths.py`：只解析本目录的路径；`requirements.txt`、`pyproject.toml`：依赖和统一格式设置。
- `data/`：原始网络（需要时）；`results/`：仿真和统计结果（需要时）。
- `reports/`：核验记录；`assets/`：需要直接复用的原始图件（需要时）。
- `README.md`：本图运行说明；所有 Python 文件直接放在本图目录。

## 原始示例图

`assets/original.pdf` 是提交稿原始矢量图。原程序没有保存该图特定随机实例的种子与边集，因此 `plot.py` 直接复用本地原图，完整保留原来的网络、排序和排版。它不会把后续生成的随机队列当作原图的数据。

原始生成程序已在项目旧版 `cache/code.zip` 的 `code/community/hub_node_experiment.py` 中定位，并原样归档到 `PREV/snapshots/before_submission_polish_20261011/original_generators/figure_2/`。该程序包含300次试验的生成与排名逻辑，含排名汇总函数并输出PDF；压缩包内未保存原队列的逐次排名、网络边集或完整随机状态。来源记录见 `PREV/snapshots/before_submission_polish_20261011/original_generator_provenance.json`。归档程序用于追溯原实验设计，当前绘图仍只依赖本图目录。

本图没有可恢复的原始数值数据，因此没有添加冒充原图复现的 `generate_data.py`。现存图件可以完全从本目录重建。

## 归档

本次整理前的完整项目代码保存在项目根目录的 `PREV/snapshots/before_figure_local_code_20261011/code/`；该路径仅用于人工追溯，运行本图不依赖它。原始投稿稿件现位于 `PREV/submitted_20260527/`，旧修订稿现位于 `cache/main_revised.tex`；归档稿件保持原样。

## 投稿前标题校对（2026-10-11）

绘图入口继续复用本目录的原始图件，仅将 CC 标题中的拼写 Clossness 修正为 Closeness，沿用图内嵌字体、字号与标题中心位置。assets/original.pdf 保持原样；validate.py 检查标题之外的渲染像素逐点一致，不改变网络、直方图、颜色或数据。

## 文稿图片同步

在本项目中，`python plot.py` 完成绘图后，会将文稿实际引用的 PDF 同步到 `main_final/figures/` 和 `response_letter/figures/`。只复制各文稿使用的图片；计算输入仍全部来自本图目录。单独复制本目录到其他位置运行时，不要求存在文稿目录。

只同步已有图片、不重新绘图时，执行 `python build_record.py`。图文摘要由 Figure 1 同步。
