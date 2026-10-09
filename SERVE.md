# ie-serve：OpenAI 兼容的 HTTP 服务

`ie-serve` 把引擎变成一个 HTTP 服务。聊天界面（例如 Open WebUI）可以直接连接它。

## 名词

- **OpenAI 兼容接口**：路径和 JSON 格式与 OpenAI 的 `/v1/chat/completions` 相同。
- **SSE**（服务器推送事件）：请求里有 `"stream": true` 时，服务每生成一段文字就发送一段。
- **ChatML**：Qwen 的聊天模板。每条消息是 `<|im_start|>角色\n内容<|im_end|>\n`。
- **引擎线程**：唯一使用模型和显卡的线程。它按顺序一次运行一个请求。

## 1. 编译

```
make build/ie-serve build/ie_kernels.hsaco
```

## 2. 启动

```
./build/ie-serve ~/models/q4/Qwen3.8-27B-Q4_K_M.gguf --host 127.0.0.1 --port 8000 --ctx 8192 --draft 3
```

服务先绑定端口，再加载模型。加载完成后，日志里有一行 `ie-serve: http://127.0.0.1:8000/v1`。

注意：如果 `--host` 不是 127.0.0.1，网络上能访问这个地址的设备都能使用模型。这时请设置 `--api-key`。

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--backend gpu\|cpu` | gpu | cpu 只用于小模型测试 |
| `--host ADDR`、`--port N` | 127.0.0.1、8000 | 监听地址和端口 |
| `--api-key KEY` | 无 | 请求必须带 `Authorization: Bearer KEY`。也可以用环境变量 `IE_API_KEY` |
| `--name ID` | 模型文件的 `general.name` | `/v1/models` 返回的模型名 |
| `--ctx N` | 8192 | 提示加输出的最多 token 数。KV 缓存按这个大小分配 |
| `--draft D` | 0 | MTP 草稿数（1 到 7，只在 GPU 上）。聊天时 3 最快（见 `claude/progress.md`） |
| `--draft-pmin P`、`--mtp FILE` | 0.6、无 | 与 `ie-run` 相同 |
| `--chunk N` | 256 | 预填充每块的 token 数（1 到 512） |
| `--temp T`、`--top-k K`、`--top-p P` | 0.7、20、0.8 | 请求没有给出这些参数时使用 |
| `--no-think` | 关 | 提示末尾加一个空的 `<think></think>`，模型不输出推理过程 |
| `--effort E` | xhigh | 思考时的推理强度：xhigh、medium、low（模板里的 reasoning_effort） |
| `--queue N` | 8 | 最多排队的请求数。超过时返回 HTTP 503 |
| `--no-cache` | 关 | 关闭提示缓存，每个请求都从位置 0 计算 |
| `--show-template` | - | 打印模型文件里的聊天模板，然后退出 |

## 3. 接口

| 方法和路径 | 说明 |
|---|---|
| `GET /health` | 返回 `{"status":"ok"}`。不检查密钥 |
| `GET /v1/models` | 返回一个模型 |
| `POST /v1/chat/completions` | 聊天。支持 `messages`、`stream`、`stream_options.include_usage`、`max_tokens`（或 `max_completion_tokens`）、`temperature`、`top_p`、`top_k`、`seed`、`reasoning_effort`、`chat_template_kwargs.enable_thinking`、`tools`、`tool_choice`（只认 `"none"`） |

`messages` 的角色可以是 system、user、assistant、tool（developer 当作 system）。内容只能是文字。

服务按模型文件里的模板（`--show-template`）生成提示，只是不处理工具和图片：

- 思考打开时，提示开头有一条 system 消息，内容是推理强度说明（medium 时没有）。
- 提示以 `<|im_start|>assistant\n<think>\n` 结束。所以服务在回答的开头补上 `<think>\n`，客户端能看到完整的 `<think>…</think>` 块。
- 以前的回答写成 `<think>\n推理\n</think>\n\n回答`。推理来自 `reasoning_content` 字段，或者来自内容里 `</think>` 之前的文字（Open WebUI 的格式）。

模板的代码在 `src/chat.c`。`build/chat_render 请求.json` 打印服务为一个请求生成的提示。

### 工具调用

名词：**工具调用**指模型不直接回答，而是要求客户端运行一个函数，再把结果发回给模型。

1. 请求里有 `tools` 时，服务按模板把工具定义写进开头的 system 消息。每个工具写成一行 JSON，格式与 Python 的 `json.dumps(ensure_ascii=False)` 相同。
2. 模型用自己的格式调用工具：`<tool_call>\n<function=名字>\n<parameter=参数>\n值\n</parameter>\n</function>\n</tool_call>`。
3. 服务只在推理（`</think>`）之后找 `<tool_call>`。找到以后，后面的文字不再作为内容发送，而是在结束时转成 OpenAI 的 `tool_calls`，`finish_reason` 是 `tool_calls`。流式输出时，`tool_calls` 在最后一个分块里一次发送。
4. 参数值按工具的 JSON Schema 转换：`string` 类型保持字符串；`integer`、`number`、`boolean`、`object`、`array` 类型的值如果是合法 JSON，就转成对应的 JSON 值。
5. 历史里 assistant 的 `tool_calls`（`arguments` 是 JSON 字符串）和 `tool` 角色的结果，按模板写回提示。
6. 如果 `<tool_call>` 块不完整（例如达到 `max_tokens`），服务把它当作普通文字返回。

在 Open WebUI 里使用工具：管理员面板 → 设置 → 模型 → 编辑 Qwen3.8 27B → 高级参数 → 函数调用（Function Calling）设为"原生（Native）"。
推测：这个菜单位置适用于当前版本，我没有在你的实例里核实过。

例子：

```
curl -sN http://127.0.0.1:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"用三段话介绍上海的历史"}],"stream":true}'
```

### 提示缓存

名词：**提示缓存**指新请求的提示开头和上一个请求相同时，服务跳过相同的部分，只计算新增的 token。

1. 线性注意力（GDN）的状态不能退回到中间位置，所以服务在两个时刻各保存一份快照（只在显存里复制线性注意力的状态、卷积环和 MTP 的 h 输入；KV 缓存不复制，因为后面的计算只写更高的位置）：
   - 快照 0：提示的最后一个 token 之前。
   - 快照 1：回答结束时（提示加上已经计算过的回答 token）。
2. 新请求的 token 以快照 1 的 token 开头时，从快照 1 继续；否则以快照 0 的 token 开头时，从快照 0 继续；否则从位置 0 开始。
3. 多轮对话时，上一轮的提示和回答（包括推理）都不再计算，只计算新的用户消息。
4. 缓存只有一份，只对应最后一个请求。注意：Open WebUI 在每次聊天后会发一个生成标题的请求，它会替换缓存。如果你想让多轮对话一直命中缓存，可以在 Open WebUI 里关掉标题生成。
5. 使用缓存时，提示的分块和不用缓存时不同，所以数值结果接近但不逐位相同。贪心解码时，检查中的回答相同（见第 6 节）。
6. 使用 MTP 时，草稿里的停止 token 不再被接受，这样模型不会计算到停止 token 之后。输出不变。
7. 日志里的 `cached N` 是跳过的 token 数，`with the last answer` 表示用了快照 1。设置 `IE_SERVE_DEBUG=1` 时，如果快照 1 没有命中，日志会打印两组 token 第一次不同的位置。

## 4. 连接 Open WebUI

推测：下面的命令适用于当前的 Open WebUI 版本。我没有在 amd-gpu-host 上运行过它。

```
docker run -d --network host --restart always --name open-webui \
  -v open-webui:/app/backend/data \
  -e OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1 \
  -e OPENAI_API_KEY=none \
  ghcr.io/open-webui/open-webui:main
```

1. 你用 `--network host` 启动容器。这样容器能访问主机的 127.0.0.1:8000。
2. 你在浏览器里打开 `http://<主机的 Tailscale 名>:8080`，创建第一个账号（管理员）。
3. 你在手机浏览器里打开同一个地址，再"添加到主屏幕"。

如果 ie-serve 设置了 `--api-key`，请把 `OPENAI_API_KEY` 改成同一个密钥。

amd-gpu-host 上的容器另外加了 `-e ENABLE_OLLAMA_API=false -e ENABLE_TAGS_GENERATION=false -e ENABLE_FOLLOW_UP_GENERATION=false -e ENABLE_AUTOCOMPLETE_GENERATION=false`。
原因：ie-serve 一次只运行一个请求，这些后台任务会让聊天排队。标题生成仍然打开。

注意：Open WebUI 第一次启动时会下载一个文本向量模型，下载慢时要等 5 到 10 分钟，8080 端口才会打开。

## 5. 开机自动启动（systemd）

**没有 sudo 时（amd-gpu-host 现在用这种）：用户级服务。**

1. 你把 `tools/ie-serve.service` 的 `[Service]` 和 `[Install]` 部分写到 `~/.config/systemd/user/ie-serve.service`，删掉 `User=` 行，把 `WantedBy` 改成 `default.target`。
2. 你运行 `systemctl --user daemon-reload && systemctl --user enable --now ie-serve`。
3. 你运行 `loginctl enable-linger $USER`。这样没有登录时服务也运行，开机后自动启动。
4. 你用 `journalctl --user -u ie-serve -f` 看日志。

**有 sudo 时：系统级服务。**

1. 你修改 `tools/ie-serve.service` 里的用户和路径。
2. 你运行 `sudo cp tools/ie-serve.service /etc/systemd/system/`。
3. 你运行 `sudo systemctl daemon-reload && sudo systemctl enable --now ie-serve`。
4. 你用 `journalctl -u ie-serve -f` 看日志。

每个请求在日志里有一行：提示 token 数、预填充速度、输出速度、MTP 每步 token 数。

## 6. GPU 上的检查结果（2026-10-08，amd-gpu-host）

1. `ie-run` 的输出不变：917db85 和新代码在三种设置下的 `ids:` 完全相同（不用草稿；7 个草稿；7 个草稿加采样）。
2. 提示与官方模板相同：4 段对话（思考打开、关闭、多轮加 low、medium）用 Jinja 渲染模板，
   再用 llama.cpp b11222 的 `llama-tokenize` 分词。结果与 ie-serve 的提示 ID 逐个相同。
3. 生成：中文聊天（3 个草稿）每秒 51 个 token，写代码每秒 78 个 token。客户端断开后，生成会停止。
4. 工具调用的提示：`tools/chat_check.py` 用 Jinja（与 transformers 的设置相同）渲染 5 段对话（带工具、带调用和结果、关闭思考、无工具），
   与 `build/chat_render` 的输出逐字相同。用 `llama-tokenize --no-escape` 分词后，与 ie-serve 的提示 ID 也逐个相同。
   注意：`llama-tokenize` 默认把文字里的 `\n` 转成换行，所以对比时必须加 `--no-escape`。
5. 工具调用的生成：模型一次调用了两个工具（上海、北京，`days` 是整数 2）；流式输出时返回了 `tool_calls`；
   把工具结果发回后，模型给出了正确的最终回答。

6. 提示缓存：`build/cache_check` 先运行提示 A，再运行"A 加回答加 7 个 token"（用快照 1），再运行同一个提示（用快照 0），最后不用缓存运行同一个提示。
   - CPU（逐个 token 计算）：Qwen3.5-0.8B（有线性注意力）和小模型的采样回答完全相同。`make test` 包含小模型的这项检查。
   - GPU（27B）：贪心解码时，有无缓存、有无 MTP 的回答都相同。温度 1 的全词表采样时回答不同，原因是提示分块不同。
   - 真实对话：5,385 个 token 的文档加一个问题，第二轮跳过 5,469 个 token，预填充从 6.7 秒降到 0.13 秒。
   - `ie-run` 的输出不变（不用草稿、7 个草稿、7 个草稿加采样）。

重复第 4 项检查：

```
./build/ie-serve MODEL --show-template > /tmp/tpl.txt
make build/chat_render && python3 tools/chat_check.py /tmp/tpl.txt build/chat_render
```

以后修改模板代码时，可以设置 `IE_SERVE_DEBUG=1`，服务会打印每个请求的提示 ID，用来重复第 2 项检查。

## 7. 分词器

`src/tok.c` 实现 GPT2 字节级 BPE 和 Qwen 的预分词正则（`qwen2` 和 `qwen35` 两种）。
Unicode 类别表 `src/tok_unicode.h` 由 `tools/gen_unicode.py` 生成。

已确认：`make tokcheck LLAMA_CPP=../llama.cpp` 用 llama.cpp 的测试词表检查分词结果。
qwen2 的 46 个测试和 qwen35 的 50 个测试全部相同。

## 8. 在没有显卡的机器上测试

```
pip install gguf numpy
python3 tools/make_tiny_chat_gguf.py ../llama.cpp/models/ggml-vocab-qwen35.gguf build/tiny-chat.gguf
./build/ie-serve build/tiny-chat.gguf --backend cpu --port 18000 --ctx 4096
```

小模型的权重是随机的，所以输出的文字没有意义。它只用来检查 HTTP、模板、分词和流式输出。
测试用的环境变量：`IE_EXTRA_STOP=ID` 增加一个停止 token；`IE_SERVE_DEBUG=1` 打印提示 ID。

## 9. 限制

- 一次只运行一个请求。其他请求排队。
- 提示缓存只有一份，只对应最后一个请求。
- 不支持图片、`stop` 字符串、`logprobs`、`n > 1`。`tool_choice` 只认 `"none"`；强制调用某个工具的写法不支持。
- 只支持 GPT2 字节级词表和 ChatML 模板（Qwen2、Qwen3.5、Qwen3.8）。
- 只监听 IPv4 地址。
