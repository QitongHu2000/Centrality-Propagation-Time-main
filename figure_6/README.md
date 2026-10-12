# Figure 6：M 模型有限时响应

这个目录可整体复制到任意位置独立运行；所有运行输入、程序和输出均在本目录。公共实现使用各图内的独立副本，不导入其他 figure 或 `cache`。六个目录保持一致的命名规则。

## 常用命令

在本目录执行（也支持从其他目录使用脚本绝对路径）：

```sh
python -m pip install -r requirements.txt
python plot.py
python validate.py
```

`plot.py` 只读取已有数据，生成 `figure_6.pdf`、`figure_6.png` 与 `build.json`，保持当前论文图形的颜色和版式。`validate.py` 核验本地依赖、构建哈希与对应数据。

## 文件组织

- `plot.py`：绘图入口；`build_record.py`：本图输入、源码和输出的哈希记录。
- `paths.py`：只解析本目录的路径；`requirements.txt`、`pyproject.toml`：依赖和统一格式设置。
- `data/`：原始网络（需要时）；`results/`：仿真和统计结果（需要时）。
- `reports/`：核验记录；`assets/`：需要直接复用的原始图件（需要时）。
- `README.md`：本图运行说明；所有 Python 文件直接放在本图目录。

## 公共实现的本地副本

`centrality.py` 保留 PTC、LPP、SGM、HSI 实现；`run_experiments.py` 提供中心性、选择规则和稳态求解；`run_finite_time.py` 提供网络读取与有限时协议；`manuscript_assets.py` 保留原图样式。四个数值图使用相同副本；仅 `paths.py` 中的图号和模型配置不同。更改公共算法后，应同步四份副本。

`docs/recent_methods.md` 和 `docs/lpp_source_audit.md` 保留方法出处。`reports/provenance/` 中的旧元数据如有旧路径，只作为来源记录，不参与当前运行查找。

## 生成数据

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python generate_data.py empirical
python generate_data.py large
python plot.py
python validate.py
```

`empirical` 只重算本图显示的四个真实网络参数；`large` 只计算本图显示的 BA/WS 各 1,000 节点参数。两条命令均限定为 M 模型，共 30 组显示参数，其中 10 组为新增大网络实验。原有未显示的有限时扫描仍作为保存记录留在 `results/finite_time/`，刷新可见参数时保留其余行。大网络已有轨迹可以复用，并再次执行求解器交叉核验。

正文第4.3.1节工作点讨论与第4.4节耗时比较使用的相关实验也在本目录，按需运行：

```sh
python generate_data.py steady
python generate_data.py working-points
python generate_data.py timing
```

- `steady`：本模型 14 个网络的稳态工作点对照；两个模型使用相同的八个原始网络和六个 BA/WS 网络。
- `working-points`：本模型四个原始网络的工作点对照及保存轨迹分析。
- `timing`：BA/WS 至 2,000 节点的排序耗时，结果与模型无关，图4和图6保存相同副本；仅需选择其中一个目录运行。

这些命令不会额外绘制已从正文删除的图7。`plot_finite_time.py` 保持原来的曲线、框内图例和排名面板；`analyze_mechanisms.py` 负责工作点分析。详细实验参数见 `docs/reproducibility_report.md`。

## 归档

本次整理前的完整项目代码保存在项目根目录的 `PREV/snapshots/before_figure_local_code_20261011/code/`；该路径仅用于人工追溯，运行本图不依赖它。原始投稿稿件现位于 `PREV/submitted_20260527/`，旧修订稿现位于 `cache/main_revised.tex`；归档稿件保持原样。

## 文稿图片同步

在本项目中，`python plot.py` 完成绘图后，会将文稿实际引用的 PDF 同步到 `main_final/figures/` 和 `response_letter/figures/`。只复制各文稿使用的图片；计算输入仍全部来自本图目录。单独复制本目录到其他位置运行时，不要求存在文稿目录。

只同步已有图片、不重新绘图时，执行 `python build_record.py`。图文摘要由 Figure 1 同步。
