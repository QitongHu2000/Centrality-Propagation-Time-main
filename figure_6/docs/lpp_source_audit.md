# LPP 原始来源核查记录

核查日期：2026-10-05。**当前处理：用户确认当前 LPP 版本正确，沿用现有实现，本项收尾。** 助手尚未取得可逐式核对的原论文全文或作者源码，未完成独立外部一致性核验；用户确认与助手独立核验应区别记录。以下保留此前访问及核查历史，不再把这项列为本轮待解决工作。主稿、LPP 实现、默认参数及实验结果未因这次来源核查改变，也未新增参数敏感性实验。

## 已确认的来源与访问结果

论文为 Guiqiong Xu、Lei Meng，*A novel algorithm for identifying influential nodes in complex networks based on local propagation probability model*，Chaos, Solitons & Fractals 168 (2023), 113155，DOI [10.1016/j.chaos.2023.113155](https://doi.org/10.1016/j.chaos.2023.113155)。

| 入口 | 本次实际取得的内容 | 能支持的结论 |
| --- | --- | --- |
| [出版页](https://www.sciencedirect.com/science/article/pii/S0960077923000565) | 搜索索引中的摘要、引言和章节预览；直接下载入口返回 HTTP 403 | 确认三阶邻域，以及跨层传播、同层传播和层次信息三类组成；不足以核对具体公式和参数 |
| [Crossref 元数据](https://api.crossref.org/works/10.1016/j.chaos.2023.113155) | 书目信息和出版商接口链接 | 确认论文身份；未给出独立开放全文或作者代码链接 |
| [Elsevier 文章接口](https://api.elsevier.com/content/article/pii/S0960077923000565) | 默认请求只返回 `coredata`；`view=FULL` 返回 HTTP 401 | 此次访问未取得正文，不能把接口响应名称中的 `full-text` 当作取得全文的证据 |
| [OpenAlex 记录](https://openalex.org/W4320168495) | `is_oa=false`、`oa_status=closed`、`any_repository_has_fulltext=false`，无 PDF URL | 当前索引没有提供开放全文位置；不等于证明所有地方都不存在合法副本 |
| [Semantic Scholar 记录](https://www.semanticscholar.org/paper/301ac78c503f7714d1c405daf5e477bc7a820222) | `openAccessPdf.url` 为空 | 未取得开放 PDF 链接 |
| CORE 公共索引 | 按 DOI 查询返回 `totalHits=0` | 本次查询未找到记录 |
| [ResearchGate 条目](https://www.researchgate.net/publication/368888267_A_novel_algorithm_for_identifying_influential_nodes_in_complex_networks_based_on_local_propagation_probability_model) | 页面标示无可用全文，提供向作者申请入口 | 未取得全文，也未代用户向作者发送申请 |
| [第一作者大学主页](https://ms.shu.edu.cn/info/1264/15328.htm)及 GitHub 搜索 | 大学主页列出该论文；以论文短语、文章号和 LPP/influential 检索，未定位到可确认归属的作者实现 | 不能声称作者未发布代码；只能说明本次尚未找到 |

此前读取的 Crossref、OpenAlex、Semantic Scholar、Elsevier 默认接口和 CORE 响应属于历史来源核查材料；当前图目录保留此说明，运行不依赖旧 `results/lpp_source_audit/`。

## 拿到原文后需要核对的项目

| 项目 | 当前稿件及 `centrality.py` 的约定 | 尚缺的证据 |
| --- | --- | --- |
| 跨层递推 | 对目标节点的上一层邻居取乘积；第一层概率为 `gamma` | 原文方程、符号定义和例图是否一致 |
| 层内影响 | 当前层诱导子图度除以该层最大度；无内部边时取零 | 原文归一化式及零分母约定；若原文未定义，需明确这是本实现的补充约定 |
| 层次分数 | 使用标准 k-core / k-shell 数 | 原文是否采用相同分解及得分，而非其他层次细分 |
| 感染概率 | `gamma = omega * <d>/<d²>` | 原文阈值表达式和概率适用范围 |
| 默认参数 | 半径 `r=3`、`omega=0.5` | 三阶范围已由摘要确认；`omega=0.5` 的原文依据和调参规则尚未核实，不能称为原作者推荐最优值 |

当前代码通过的解析小图、有限值和重标号检查，说明其与**稿件列出的递推**相符，不证明它与尚未取得的原文或作者实现完全一致。现有 56/56 前三结论也应理解为针对已披露实现、11 方法比较池和规定末时刻的结果。

## 用户确认前的范围记录

- 保留 TODO 中“LPP 原文/作者实现外部交叉核对”的未完成状态。
- 不用参数扫描替代原文核验，也不因为暂时无法下载而猜测或改写算法。
- 参数敏感性属于可选补充，目前未运行。取得全文或作者代码后，先逐项核对；只有确有实现或参数差异时，才修正并重算受影响的比较。

## 再次核查的补充发现（2026-10-05）

按用户“确认已有 LPP 实现与原论文一致”的要求，再次检查项目文件和可访问来源。项目中仍未发现该论文全文。使用本机 Chrome 打开出版页后，页面停留在 `Just a moment...`，没有取得正文；没有处理验证挑战或改变浏览器/网络安全设置。

GitHub 代码检索找到一个[公开 LPP notebook](https://github.com/aqingfengmingyue/experience/blob/da47735171a1efe1d5793aa997c2119cb27db241/python/complex_networks/manners/local_propagation_probability.ipynb)，固定提交为 `da47735171a1efe1d5793aa997c2119cb27db241`。已静态阅读代码，未执行。该仓库没有提供足以确认其为 Xu、Meng 原作者实现的归属证据，所检索的完整目录树也没有原论文 PDF。此文件只能作为待核实线索，不能作为论文算法的权威依据。

这个 notebook 的 `yita` 使用节点在**整个图中的度**，按同一 BFS 层的最大值归一化；其传播概率直接使用 `<d>/<d²>`，相当于当前参数记号下的 `omega=1`。当前修订实现则使用**层内诱导子图的度**并取 `omega=0.5`。这些差异不能说明哪一个版本符合原论文，也不能据此替换当前实现。

**当时的独立核查结论为未确认。** 缺少的是原论文方法定义及参数部分，而不是更多内部测试。后来用户确认沿用当前实现；该确认结束本轮待办，但不改变此前实际取得的证据范围。本次未修改 `main.tex`、`main_revised.tex`、算法或实验结果。
