# Img2Answer

Img2Answer 是一个本地优先的 Python 项目，用于把本地 PDF 题库材料处理成可结构化存储、可进一步做以图搜图的数据资产。

当前项目仍处于 POC 阶段，重点验证从 PDF 页段定位到图片裁剪、元数据持久化、图片向量入库的最小闭环。原始 PDF、生成图片、本地设计文档、向量库、数据库和 conda 环境都不提交到仓库。

## 当前功能

- 读取 YAML 或 JSON 页段配置。
- 检查本地 PDF 基础信息：
  - 文件路径
  - 页数
  - SHA-256 哈希
- 校验配置页码范围。
- 将配置页段渲染为 PNG 页面图片。
- 从渲染页面生成简单的图形裁剪候选图。
- 写入 JSON 处理报告。
- 将本地元数据持久化到 SQLite：
  - 来源文档
  - 裁剪候选图
  - 处理报告摘要
- 可选将裁剪候选图写入本地 ChromaDB：
  - 使用图片 ID 作为向量记录 ID
  - metadata 保留 SQLite 回查字段
  - 默认向量库目录为 `<output-dir>/chroma`
- 支持通过本地查询图片检索相似裁剪候选图：
  - 查询 ChromaDB Top K 相似向量
  - 通过 `image_id` 回查 SQLite 图片记录
  - 输出结构化 JSON
- 提供 CLI 入口。
- 使用生成的样本 PDF 做单元测试和集成测试，不依赖真实本地材料。

## 暂未实现

- OCR
- Web API 图片搜索接口
- 文本搜索接口
- Web UI
- 完整题目解析
- 题干、选项、答案的精细结构化切分
- 面向生产的大规模图片 embedding 模型接入

## 仓库规则

以下路径只用于本地调试，已被 Git 忽略：

```text
data/
.conda/
docs/
```

不要提交：

- 原始 PDF
- 渲染页面图片或裁剪图片
- 本地设计文档
- 本地向量库
- SQLite 或其他数据库文件
- 密钥、Token、账号信息

## 环境准备

在项目根路径下创建 conda 环境：

```powershell
conda env create -p .\.conda -f environment.yml
conda activate .\.conda
```

环境目录固定存放在：

```text
.\.conda
```

## 配置文件

使用页段配置描述需要处理的 PDF 页面范围。

示例：

```yaml
documents:
  sample_doc:
    path: data/raw/sample.pdf
    sections:
      graphic_reasoning:
        page_from: 1
        page_to: 3
```

示例配置文件：

```text
configs/sample.sections.example.yml
```

配置中的页码从 `1` 开始，和人工阅读 PDF 时看到的页码保持一致。

## 运行 PDF 处理 POC

在项目根路径执行：

```powershell
$env:PYTHONPATH = "src"
.\.conda\python.exe -m img2answer.cli --config configs\sample.sections.example.yml --output-dir data\processed --database data\processed\img2answer.sqlite3 --dpi 300
```

输出目录结构：

```text
data/processed/
  pages/
  crops/
  reports/
  img2answer.sqlite3
```

这些输出都不会提交到 Git。

如果省略 `--database`，SQLite 数据库默认写入：

```text
<output-dir>/img2answer.sqlite3
```

## 运行图片向量入库

开启 `--embed-images` 后，CLI 会在完成 PDF 渲染、裁剪和 SQLite 元数据写入后，把裁剪候选图写入 ChromaDB。

```powershell
$env:PYTHONPATH = "src"
.\.conda\python.exe -m img2answer.cli --config configs\sample.sections.example.yml --output-dir data\processed --dpi 300 --embed-images
```

默认向量库目录：

```text
<output-dir>/chroma
```

也可以显式指定：

```powershell
$env:PYTHONPATH = "src"
.\.conda\python.exe -m img2answer.cli --config configs\sample.sections.example.yml --output-dir data\processed --dpi 300 --embed-images --chroma-dir data\processed\chroma --chroma-collection question_images
```

当前 embedding 模型是轻量、确定性的本地 POC 实现，用于验证数据链路；后续可以替换为更适合图形题检索的视觉模型。

## 运行图片相似检索

完成图片向量入库后，可以使用 `search-image` 子命令查询相似图片：

```powershell
$env:PYTHONPATH = "src"
.\.conda\python.exe -m img2answer.cli search-image --image data\query.png --database data\processed\img2answer.sqlite3 --chroma-dir data\processed\chroma --top-k 5
```

输出为 JSON，包含：

- 查询图片路径
- Top K 数量
- 相似图片 ID
- ChromaDB 距离
- Chroma metadata
- SQLite 图片记录

如果 ChromaDB 中存在向量但 SQLite 已无对应图片记录，结果中的 `record` 会是 `null`，用于暴露本地数据一致性问题。

## 运行测试

```powershell
$env:PYTHONPATH = "src"
.\.conda\python.exe -m unittest discover -s tests -v
```

当前测试覆盖：

- 配置加载
- 非法页码范围处理
- PDF 元数据读取
- 页段渲染
- 裁剪候选图生成
- 报告写入
- SQLite 元数据持久化
- 重跑时元数据替换
- 图片 embedding 生成
- 图片向量入库 metadata
- 真实 ChromaDB 持久化
- 图片相似检索
- 检索结果 SQLite 回查
- CLI 向量入库调用链
- CLI 图片检索调用链

## CI 门禁

GitHub Actions 会在提交到 `main` 和面向 `main` 的 Pull Request 中运行测试。

CI 使用 Python 3.11，并执行：

```bash
python -m pip install setuptools wheel
python -m pip install -e ".[vector]" --no-build-isolation
python -c "import img2answer; print(img2answer.__version__)"
python -m unittest discover -s tests -v
```

CI 不依赖 `data/`、`docs/`、`.conda/` 等本地路径。

## 开发流程

项目采用 issue-first 和 SDD 风格流程：

1. 创建或草拟 Issue。
2. 编写并审计本地设计文档。
3. 基于功能分支实现。
4. 实现必须跟随测试。
5. 创建 PR，并使用 `Closes #<issue-number>` 关联 Issue。

分支名保持简短，不要求包含 Issue 编号：

```text
feature/pdf-section-crop-poc
fix/page-range-validation
```
