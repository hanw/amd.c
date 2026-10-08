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
| `--show-template` | - | 打印模型文件里的聊天模板，然后退出 |

## 3. 接口

| 方法和路径 | 说明 |
|---|---|
| `GET /health` | 返回 `{"status":"ok"}`。不检查密钥 |
| `GET /v1/models` | 返回一个模型 |
| `POST /v1/chat/completions` | 聊天。支持 `messages`、`stream`、`stream_options.include_usage`、`max_tokens`（或 `max_completion_tokens`）、`temperature`、`top_p`、`top_k`、`seed`、`reasoning_effort`、`chat_template_kwargs.enable_thinking` |

`messages` 的角色只能是 system、user、assistant（developer 当作 system）。内容只能是文字。

服务按模型文件里的模板（`--show-template`）生成提示，只是不处理工具和图片：

- 思考打开时，提示开头有一条 system 消息，内容是推理强度说明（medium 时没有）。
- 提示以 `<|im_start|>assistant\n<think>\n` 结束。所以服务在回答的开头补上 `<think>\n`，客户端能看到完整的 `<think>…</think>` 块。
- 以前的回答写成 `<think>\n推理\n</think>\n\n回答`。推理来自 `reasoning_content` 字段，或者来自内容里 `</think>` 之前的文字（Open WebUI 的格式）。

例子：

```
curl -sN http://127.0.0.1:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"用三段话介绍上海的历史"}],"stream":true}'
```

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

## 5. 开机自动启动（systemd）

1. 你修改 `tools/ie-serve.service` 里的用户和路径。
2. 你运行 `sudo cp tools/ie-serve.service /etc/systemd/system/`。
3. 你运行 `sudo systemctl daemon-reload && sudo systemctl enable --now ie-serve`。
4. 你用 `journalctl -u ie-serve -f` 看日志。每个请求有一行：提示 token 数、预填充速度、输出速度、MTP 每步 token 数。

## 6. GPU 上的检查结果（2026-10-08，amd-gpu-host）

1. `ie-run` 的输出不变：917db85 和新代码在三种设置下的 `ids:` 完全相同（不用草稿；7 个草稿；7 个草稿加采样）。
2. 提示与官方模板相同：4 段对话（思考打开、关闭、多轮加 low、medium）用 Jinja 渲染模板，
   再用 llama.cpp b11222 的 `llama-tokenize` 分词。结果与 ie-serve 的提示 ID 逐个相同。
3. 生成：中文聊天（3 个草稿）每秒 51 个 token，写代码每秒 78 个 token。客户端断开后，生成会停止。

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
- 每个请求从位置 0 重新计算整个提示。多轮对话时，前面的对话每次都重新预填充。
- 不支持图片、工具调用（tools）、`stop` 字符串、`logprobs`、`n > 1`。
- 只支持 GPT2 字节级词表和 ChatML 模板（Qwen2、Qwen3.5、Qwen3.8）。
- 只监听 IPv4 地址。
