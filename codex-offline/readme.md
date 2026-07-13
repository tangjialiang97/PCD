1. 解压并按照
mkdir -p ~/.local/bin

tar -xzf codex-x86_64-unknown-linux-musl.tar.gz

mv codex-x86_64-unknown-linux-musl ~/.local/bin/codex

chmod +x ~/.local/bin/codex

2. 添加到path
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc || \
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

source ~/.bashrc
hash -r

which codex

3. 配置api

创建 ~/.codex/config.toml 文件，并添加如下配置：
model = "gpt-5.5"
model_provider = "our"
preferred_auth_method = "apikey"


[model_providers.our]
name = "Any Router"
base_url = "https://ai.xem8k5.top/v1"
wire_api = "responses"
创建 ~/.codex/auth.json 文件，并添加如下配置：

{
  "OPENAI_API_KEY":"sk-uVGf0vcnjThZ87isMHYuU7SgQlNpdc2SUB9csQ1HUWmmo1Sc"
}

4. 临时使用
   
OPENAI_API_KEY='sk-uVGf0vcnjThZ87isMHYuU7SgQlNpdc2SUB9csQ1HUWmmo1Sc' codex \
  -m gpt-5.5 \
  -c 'model_provider="our"' \
  -c 'model_providers.our.name="test"' \
  -c 'model_providers.our.base_url="https://ai.xem8k5.top/v1"' \
  -c 'model_providers.our.env_key="OPENAI_API_KEY"' \
  -c 'model_providers.our.wire_api="responses"'
