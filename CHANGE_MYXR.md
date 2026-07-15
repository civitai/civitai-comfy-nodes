# HTTP Proxy 中间代理配置

## 修改说明

为 civitai-comfy-nodes 所有 HTTP 请求添加了统一的中间代理支持，代理配置入口集中在 `config.py` 中。

## 修改的文件

### `civitai_comfy_nodes/_debug.py` (新文件)
- 新增 `DEBUG: bool = False` 模块级变量
- 新增 `debug_log()` 函数，`DEBUG=True` 时在控制台打印请求详情

### `civitai_comfy_nodes/config.py`
- 新增 `PROXY_URL: str | None = None` 模块级变量，可直接在文件中设置代理地址
- 新增 `proxy_url()` 函数，解析优先级：`CIVITAI_COMFY_PROXY` 环境变量 > `PROXY_URL`
- `ClientConfig` 数据类新增 `proxy_url: str = ""` 字段
- `resolve_config()` 新增代理解析逻辑：节点输入 > 环境变量 > 模块默认值
- 从 `._debug` 重新导出 `DEBUG` 和 `debug_log`，方便用户统一在 `config.py` 中配置

### `civitai_comfy_nodes/proxy.py`
- `get_proxy()` 扩展代理解析链：`CivitaiProxy` 节点设置 > `config.proxy_url()` > 环境变量

### `civitai_comfy_nodes/client.py`
- `OrchestrationClient.__init__()` 优先使用 `config.proxy_url`，其次回退到 `get_proxy()`
- `_request()`、`download_blob()`、`upload_media()` 新增调试日志

### `civitai_comfy_nodes/catalog.py`
- `search()`、`lookup()`、`components()` 新增调试日志

### `civitai_comfy_nodes/oauth.py`
- `_refresh()`、`interactive_login()` 新增调试日志

### `civitai_comfy_nodes/local_models.py`
- `download_model()` 新增调试日志

### `civitai_comfy_nodes/nodes_manual.py`
- `CivitaiAuth` 节点新增可选 `proxy_url` 输入字段，可在工作流中直接配置代理

## 代理配置方式（按优先级从高到低）

| 优先级 | 方式 | 说明 |
|--------|------|------|
| 1 (最高) | CivitaiAuth 节点 `proxy_url` 字段 | 工作流中通过 Auth 节点设置，仅影响该客户端 |
| 2 | CivitaiProxy 节点 | 工作流中通过 Proxy 节点设置，全局生效 |
| 3 | `CIVITAI_COMFY_PROXY` 环境变量 | 进程级全局代理 |
| 4 (最低) | `config.py` 中 `PROXY_URL` 变量 | 代码级默认值 |

## DEBUG 调试模式

在 `config.py` 中设置 `DEBUG = True` 可在控制台详细打印所有 HTTP 请求信息：

```python
# civitai_comfy_nodes/config.py
DEBUG = True   # 开启调试日志
```

开启后每次请求会打印：
- 请求方法、完整 URL、查询参数
- 使用的代理地址（如有）
- 响应状态码

## 使用示例

### 方式一：在 `config.py` 中设置
```python
# civitai_comfy_nodes/config.py
PROXY_URL = "http://127.0.0.1:7890"
DEBUG = True   # 可选：开启调试日志
```

### 方式二：环境变量
```bash
set CIVITAI_COMFY_PROXY=http://127.0.0.1:7890
```

### 方式三：工作流节点
- 使用 **CivitaiAuth** 节点，在 `proxy_url` 字段填入代理地址
- 或使用 **CivitaiProxy** 节点，启用并填入代理地址

## 网络请求文件分布

所有 HTTP 请求均通过 `requests` 库发送，`proxies` 参数由 `proxy.get_proxy()` 统一提供：

| 文件 | 请求方式 | 用途 |
|------|---------|------|
| `client.py` | `requests.Session` | 编排器 API（提交工作流、查询、下载等） |
| `catalog.py` | `requests.get()` | Civitai 公开 API 搜索/查询模型 |
| `oauth.py` | `requests.post()` | OAuth 令牌刷新/交换 |
| `local_models.py` | `requests.get()` | 流式下载模型文件 |