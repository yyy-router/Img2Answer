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
- 可选使用阿里云在线 OCR 识别指定页段：
  - 后端从 `.env` 读取凭证，只有显式开启才上传页面图片
  - 本地保存文字块、像素坐标和实际返回的置信度
  - 重复运行复用有效缓存，低置信度或无文字结果标记为待核对
- 将本地元数据持久化到 SQLite：
  - 来源文档
  - 裁剪候选图
  - 处理报告摘要
- 可选将裁剪候选图写入本地 ChromaDB：
  - 使用图片 ID 作为向量记录 ID
  - metadata 保留 SQLite 回查字段
  - 默认向量库目录为 `<output-dir>/chroma`
- 支持选择图片 embedding 模型：
  - 默认 `pillow-hash`，用于轻量 POC 和 CI
  - 可选 `openclip`，用于本地真实视觉 embedding
- 支持通过本地查询图片检索相似裁剪候选图：
  - 查询 ChromaDB Top K 相似向量
  - 通过 `image_id` 回查 SQLite 图片记录
  - 输出结构化 JSON
- 提供本地 HTTP API：
  - 上传图片进行相似检索
  - 图片详情回查接口
  - 数据库、向量库和模型使用服务端配置
- 提供 CLI 入口。
- 使用生成的样本 PDF 做单元测试和集成测试，不依赖真实本地材料。

## 暂未实现

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
.env
.env.*
```

`.env.example` 是可提交的通用配置模板。

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

## 阿里云在线 OCR

OCR 是可选的导入能力，使用阿里云 `RecognizeAdvanced` 全文识别高精版。PDF 先在本地渲染为 PNG，再使用官方 SDK 提交图片二进制，不需要公开图床。该接口不直接接收 PDF。[阿里云接口文档](https://help.aliyun.com/zh/ocr/developer-reference/api-ocr-api-2021-07-07-recognizeadvanced)

从项目根目录安装可选依赖，仍使用项目内 conda 环境：

```powershell
.\.conda\python.exe -m pip install -e ".[ocr]" --no-build-isolation
```

在现有 `.env` 中补充以下配置，不要覆盖其他 API 配置。凭证只在本地填写，不提交到 Git，也不提供给前端：

```dotenv
ALIBABA_CLOUD_ACCESS_KEY_ID=
ALIBABA_CLOUD_ACCESS_KEY_SECRET=
IMG2ANSWER_OCR_CONNECT_TIMEOUT_SECONDS=5
IMG2ANSWER_OCR_READ_TIMEOUT_SECONDS=30
IMG2ANSWER_OCR_MIN_CONFIDENCE=80
```

需要先开通对应 OCR 服务，并为 RAM 用户配置所需调用权限。首版固定使用杭州 endpoint `ocr-api.cn-hangzhou.aliyuncs.com`，不支持任意第三方地址。[阿里云 SDK 说明](https://help.aliyun.com/zh/ocr/developer-reference/sdk-overview)

配置含义：连接与读取超时必须为正整数秒；核对阈值是 0 至 100 的有限数值。只读取项目启动目录的 `.env`，进程环境变量优先，配置解析不修改进程环境，不进行变量插值。必须从项目根目录运行。

使用页段配置选出需要识别的页面，然后显式开启：

```powershell
.\.conda\python.exe -m img2answer.cli --config configs\sample.sections.example.yml --output-dir data\processed --dpi 300 --ocr
```

`--ocr` 会将配置范围内的页面图片发送到阿里云，可能产生费用；请确认材料允许外传，并先选择少量页面验证。未指定该开关时，配置了密钥也不会请求云服务，现有图片搜索 API 不调用 OCR。OCR 可与 `--embed-images` 一起使用，但不改变现有裁剪或 embedding 规则。

识别结果保存在：

```text
<output-dir>/ocr/<document-id>/<section>/page-<page-number>.json
```

结果包含识别文字、文字块四点坐标、边界框 `bbox`、可用的置信度、图片哈希、请求 ID 及成功响应的识别数据。坐标以提交图片的左上角为原点，单位是像素。缺失的置信度保存为 `null`，不编造数值。处理报告新增 `ocr_pages`，记录每页结果路径、缓存命中、识别状态和 `needs_review`；SQLite 通过已有报告路径定位这些结果，不新增数据表。

无文字结果标记为 `empty` 并待核对；置信度缺失或低于阈值也会标记待核对。这不等于识别失败，也不是答案正确率保证。目前尚不支持单题切分、选项与答案关联，搜索接口不会新增题干或答案字段。

再次执行相同命令会检查图片内容、请求选项和结果完整性，只补充未命中的页面。修改核对阈值不会重复识别。确实需要重新请求时使用：

```powershell
.\.conda\python.exe -m img2answer.cli --config configs\sample.sections.example.yml --output-dir data\processed --ocr --ocr-refresh
```

`--ocr-refresh` 必须配合 `--ocr`，会跳过有效缓存。服务调用不自动重试，超时并不能证明云端未处理，应核对调用记录后再续跑。权限、额度、限流或异常响应会停止导入并返回非零退出码，不覆盖失败文档的既有成功报告与元数据；此前完成的文档、页面缓存和渲染文件不回滚。

提交图片仅支持静态 PNG/JPEG，最多 10,000,000 字节，单边大于 15 且小于 8192 像素，长宽比小于 50。超限会在调用前拒绝，可调整 `--dpi` 重新渲染；不会自动压缩或改变坐标系，严重模糊的图片不能保证恢复。

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

默认 collection 会按 embedding 模型隔离：

```text
pillow-hash -> question_images
openclip -> question_images_openclip
```

也可以显式指定：

```powershell
$env:PYTHONPATH = "src"
.\.conda\python.exe -m img2answer.cli --config configs\sample.sections.example.yml --output-dir data\processed --dpi 300 --embed-images --chroma-dir data\processed\chroma --chroma-collection question_images
```

当前 embedding 模型是轻量、确定性的本地 POC 实现，用于验证数据链路；后续可以替换为更适合图形题检索的视觉模型。

可以通过 `--embedding-model` 指定模型：

```powershell
$env:PYTHONPATH = "src"
.\.conda\python.exe -m img2answer.cli --config configs\sample.sections.example.yml --output-dir data\processed --dpi 300 --embed-images --embedding-model pillow-hash
```

OpenCLIP 是可选本地模型。使用前需要额外安装可选依赖：

```powershell
.\.conda\python.exe -m pip install -e ".[openclip]"
```

然后用同一个模型完成入库：

```powershell
$env:PYTHONPATH = "src"
.\.conda\python.exe -m img2answer.cli --config configs\sample.sections.example.yml --output-dir data\processed --dpi 300 --embed-images --embedding-model openclip
```

同一个 ChromaDB collection 应只保存同一种 embedding 维度。通常不要把 `pillow-hash` 和 `openclip` 手动写入同一个 collection；如果不传 `--chroma-collection`，CLI 会自动使用各自默认 collection。

## 运行图片相似检索

完成图片向量入库后，可以使用 `search-image` 子命令查询相似图片：

```powershell
$env:PYTHONPATH = "src"
.\.conda\python.exe -m img2answer.cli search-image --image data\query.png --database data\processed\img2answer.sqlite3 --chroma-dir data\processed\chroma --top-k 5
```

如果入库时使用了 OpenCLIP，检索时也应使用同一个模型：

```powershell
$env:PYTHONPATH = "src"
.\.conda\python.exe -m img2answer.cli search-image --image data\query.png --database data\processed\img2answer.sqlite3 --chroma-dir data\processed\chroma --top-k 5 --embedding-model openclip
```

如果入库时没有显式指定 `--chroma-collection`，检索时也可以省略它，CLI 会按 `--embedding-model` 选择对应默认 collection。

输出为 JSON，包含：

- 查询图片路径
- Top K 数量
- 相似图片 ID
- ChromaDB 距离
- Chroma metadata
- SQLite 图片记录
- 面向人工核查的详情字段 `detail`：
  - 来源文档 ID
  - 题型 section
  - 裁剪图路径
  - 原始渲染页路径
  - 裁剪框 bbox
  - 图片宽高
  - 处理报告路径

如果 ChromaDB 中存在向量但 SQLite 已无对应图片记录，结果中的 `record` 和 `detail` 会是 `null`，用于暴露本地数据一致性问题。

## 运行本地 API

API 是可选能力。使用前安装 API 依赖：

```powershell
.\.conda\python.exe -m pip install -e ".[vector,api]"
```

数据库、向量库和模型在项目根目录的 `.env` 中配置，客户端不传这些参数。应用启动时自动读取该文件，通用模板见 `.env.example`：

```dotenv
IMG2ANSWER_DATABASE=data/processed/img2answer.sqlite3
IMG2ANSWER_CHROMA_DIR=data/processed/chroma
IMG2ANSWER_CHROMA_COLLECTION=
IMG2ANSWER_EMBEDDING_MODEL=pillow-hash
IMG2ANSWER_TOP_K=5
IMG2ANSWER_MAX_UPLOAD_BYTES=10485760
```

请将路径改为已完成入库的本地数据路径。从项目根目录启动服务，无需在启动命令中逐项设置配置：

```powershell
.\.conda\python.exe -m uvicorn img2answer.api:app --host 127.0.0.1 --port 8000
```

`.env` 和相对数据路径按启动时的工作目录读取，不会搜索父目录的配置。文件支持 UTF-8、注释和带引号的值，不进行变量插值。进程环境变量优先于 `.env`，未配置的字段使用默认值；文件不存在时也可以启动。修改配置后需要重启服务。实际 `.env` 被 Git 忽略，只提交通用模板。

| 配置项 | 默认值 | 用途 |
| --- | --- | --- |
| `IMG2ANSWER_DATABASE` | `data/processed/img2answer.sqlite3` | SQLite 数据库路径 |
| `IMG2ANSWER_CHROMA_DIR` | `data/processed/chroma` | ChromaDB 持久化目录 |
| `IMG2ANSWER_CHROMA_COLLECTION` | 未设置 | 按模型选择默认 collection |
| `IMG2ANSWER_EMBEDDING_MODEL` | `pillow-hash` | 图片 embedding 模型 |
| `IMG2ANSWER_TOP_K` | `5` | 默认返回数量，必须大于 0 |
| `IMG2ANSWER_MAX_UPLOAD_BYTES` | `10485760` | 图片大小上限，单位字节，默认 10 MiB，必须大于 0 |

如果入库使用 OpenCLIP，先安装模型依赖：

```powershell
.\.conda\python.exe -m pip install -e ".[openclip]"
```

然后在 `.env` 中设置 `IMG2ANSWER_EMBEDDING_MODEL=openclip`。

未指定 collection 时，`pillow-hash` 使用 `question_images`，`openclip` 使用 `question_images_openclip`。如果入库时指定了其他 collection，应通过 `IMG2ANSWER_CHROMA_COLLECTION` 配置相同名称。

配置好 `.env` 后，也可以使用项目脚本启动：

```powershell
.\.conda\Scripts\img2answer-api.exe
```

图片相似检索接口接收 `multipart/form-data`，直接上传查询图片，无需单独调用上传接口：

```http
POST http://127.0.0.1:8000/search/image
```

请求表单字段：

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | 文件 | 是 | 查询图片 |
| `top_k` | 正整数 | 否 | 返回结果数量；省略时使用服务端默认值 |

数据库路径、向量库路径及模型由服务端配置，不属于请求参数。

响应中的 `query_image` 是上传文件名，`results` 为检索结果。查询图片只用于本次检索，系统临时文件在成功或失败后都会清理；上传图片不会作为新题目写入数据库或向量库。结果继续包含用于本地核查的裁剪图、来源页和报告路径，目前没有图片下载接口。

上传图片默认最多 10 MiB，请求体额外允许 64 KiB 的表单开销。服务在 multipart 解析前有界接收并检查实际请求体大小，复制图片时再次检查文件大小，超限返回 `413`；该检查不依赖客户端声明的 `Content-Length`。

每个应用实例在首次有效检索时加载 embedding 模型，后续检索复用同一个模型，并发首次请求也只初始化一次。初始化失败时后续请求可重试；详情查询不加载模型。多个服务进程各自持有模型实例。

图片详情回查接口只需要把检索结果中的 `image_id` 填入 URL：

```http
GET http://127.0.0.1:8000/images/{image_id}
```

接口定义见 `http://127.0.0.1:8000/docs`。

缺少上传文件或 `top_k` 格式错误返回 `422`；空文件、无效图片或 `top_k < 1` 返回 `400`；超限上传返回 `413`；图片 ID 不存在返回 `404`；服务端数据库文件不可用返回 `503`；后端检索或读取失败返回 `500`，详细原因记录在服务端日志中。

API 不负责生成图片向量；需要先使用 CLI 完成 PDF 处理和图片向量入库。

## 运行测试

完整测试包含 OCR SDK 参数校验，需要先安装所有测试使用的可选依赖：

```powershell
.\.conda\python.exe -m pip install -e ".[vector,api,ocr]" --no-build-isolation
```

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
- embedding 模型选择
- 图片向量入库 metadata
- 真实 ChromaDB 持久化
- 图片相似检索
- 检索结果 SQLite 回查
- 检索结果详情回查
- CLI 向量入库调用链
- CLI 图片检索调用链
- 本地 API 图片检索接口
- 本地 API 图片详情回查接口
- API 服务端配置隔离与上传临时文件清理
- API `.env` 配置加载与进程环境变量优先级
- API 上传大小限制、分块请求及拒绝请求的资源清理
- API 模型复用、并发初始化和初始化失败重试
- 图片上传检索与真实 ChromaDB 集成
- OCR 配置、凭证隐藏与 SDK 二进制请求参数
- OCR 响应、坐标、尺寸及置信度校验
- OCR 缓存、强制刷新、失败续跑及导入报告引用

OCR 自动化测试使用模拟云端响应，不上传题库或产生云端调用费用，不代表真实材料的识别准确率。运行全部测试前需要安装 `.[vector,api,ocr]` 可选依赖。

## CI 门禁

GitHub Actions 会在提交到 `main` 和面向 `main` 的 Pull Request 中运行测试。

CI 使用 Python 3.11，并执行：

```bash
python -m pip install setuptools wheel
python -m pip install -e ".[vector,api,ocr]" --no-build-isolation
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
