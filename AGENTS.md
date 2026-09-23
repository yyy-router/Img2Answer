# AGENTS.md

## Project Governance

本项目使用 GitHub 仓库托管：

- Repository: `https://github.com/yyy-router/Img2Answer`
- 默认远程仓库名：`origin`
- 推荐主分支：`main`

所有协作、文档、开发、测试和合并行为都必须遵循本文件约束。

## Runtime Environment

本项目运行环境使用 conda 管理。

约束：

- 必须在当前项目根路径下创建并存储虚拟环境。
- 推荐环境目录：`.conda/`
- 不要使用全局 Python 环境安装项目依赖。
- 不要将虚拟环境目录提交到 Git。
- 依赖安装和运行命令应优先写入项目文档，保证其他协作者可以复现。

示例：

```powershell
conda create -p .\.conda python=3.11
conda activate .\.conda
```

## Git Collaboration Workflow

项目采用 issue-first 的协作流程。

新增功能必须遵循：

1. 新建 GitHub Issue。
2. 在 Issue 中提出需求清单、边界、验收标准和测试要求。
3. 基于 Issue 创建子分支。
4. 在子分支中完成文档、实现和测试。
5. 创建 Pull Request，并在 PR 描述中关联 Issue。
6. PR 必须使用自动关闭关键字关联 Issue，例如 `Closes #123`。
7. 评审通过后再合并代码。

分支命名建议：

```text
feature/<issue-id>-<short-name>
fix/<issue-id>-<short-name>
docs/<issue-id>-<short-name>
test/<issue-id>-<short-name>
```

## SDD Development Mode

本项目采用 SDD（Specification/Design Driven Development）开发模式。

新增功能必须先写文档，再实现代码：

1. 先补充需求说明、设计方案或接口约定。
2. 文档必须经过人工审计。
3. 文档审计通过后，才可以进入实现阶段。
4. 实现必须严格跟随已审计文档。
5. 如果开发过程中发现设计需要调整，必须先更新文档并重新审计。

禁止绕过文档直接实现新增功能。

## Design Principles

项目设计必须遵循第一性原则。

要求：

- 先明确真实问题，再选择技术方案。
- 只实现当前阶段真正需要的能力。
- 不做冗余设计。
- 不做过度设计。
- 不提前引入复杂架构、抽象层或基础设施。
- 优先选择简单、可验证、可替换的实现。
- 每个模块都应有清晰职责，避免为了“未来可能需要”而扩张范围。

判断标准：

- 如果没有明确需求，不添加功能。
- 如果没有重复复杂度，不抽象。
- 如果没有规模压力，不引入重型组件。
- 如果可以用配置解决，不写死逻辑。
- 如果可以先用本地方案验证，不急于服务化。

## Documentation Requirements

所有新增功能必须有对应文档。

文档至少说明：

- 背景与目标
- 非目标
- 输入与输出
- 数据结构或接口约定
- 主要流程
- 错误处理
- 测试方案
- 验收标准

文档可以放在：

```text
docs/
```

重要功能建议按 Issue 建立独立文档：

```text
docs/issues/<issue-id>-<short-name>.md
```

## Testing Requirements

功能实现必须跟随对应测试。

要求：

- 新增业务逻辑必须有单元测试。
- 涉及 PDF 解析、图片裁剪、向量检索等流程时，应补充集成测试或样本验证脚本。
- 修复 bug 时，应先补充能复现问题的测试。
- PR 中必须说明测试范围和测试结果。
- 没有测试的实现不能视为完成。

测试应覆盖：

- 正常输入
- 边界输入
- 失败输入
- 关键数据结构
- 与文档验收标准对应的行为

## Pull Request Requirements

PR 必须包含：

- 关联 Issue，例如 `Closes #123`
- 需求摘要
- 文档变更说明
- 实现变更说明
- 测试说明
- 风险和回滚方式

在满足以下条件前，不应合并：

- 文档已审计
- 代码已实现
- 测试已通过
- 无明显冗余设计或过度设计
- PR 已关联并可自动关闭对应 Issue

## Repository Hygiene

禁止提交：

- `data/` 下的原始 PDF、裁剪图片、向量库和数据库文件
- `.conda/` 或其他本地虚拟环境目录
- 临时文件、缓存文件、日志文件
- 密钥、Token、账号信息

允许提交：

- 源代码
- 测试代码
- 项目文档
- 轻量配置文件
- 小型测试样例或脱敏样例

## Current Project Direction

当前项目方向是从本地 PDF 题库材料中提取目标题型，完成结构化存储，并支持以图搜图。具体材料名称、文件名、页码范围等细节仅用于本地调试，不应写入项目级协作规则。

第一阶段应优先验证：

- PDF 页段定位
- 图形推理题裁剪
- 图片向量生成
- ChromaDB 相似检索
- 题目结构化存储
- 查询结果回查题目详情
