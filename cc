-- CC Switch SQLite 导出
-- 生成时间: 2026-04-10 02:49:15
-- user_version: 6
PRAGMA foreign_keys=OFF;
PRAGMA user_version=6;
BEGIN TRANSACTION;
CREATE TABLE mcp_servers (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, server_config TEXT NOT NULL,
            description TEXT, homepage TEXT, docs TEXT, tags TEXT NOT NULL DEFAULT '[]',
            enabled_claude BOOLEAN NOT NULL DEFAULT 0, enabled_codex BOOLEAN NOT NULL DEFAULT 0,
            enabled_gemini BOOLEAN NOT NULL DEFAULT 0, enabled_opencode BOOLEAN NOT NULL DEFAULT 0
        );
CREATE TABLE model_pricing (
            model_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
            input_cost_per_million TEXT NOT NULL, output_cost_per_million TEXT NOT NULL,
            cache_read_cost_per_million TEXT NOT NULL DEFAULT '0',
            cache_creation_cost_per_million TEXT NOT NULL DEFAULT '0'
        );
CREATE TABLE prompts (
            id TEXT NOT NULL, app_type TEXT NOT NULL, name TEXT NOT NULL, content TEXT NOT NULL,
            description TEXT, enabled BOOLEAN NOT NULL DEFAULT 1, created_at INTEGER, updated_at INTEGER,
            PRIMARY KEY (id, app_type)
        );
CREATE TABLE provider_endpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider_id TEXT NOT NULL,
                app_type TEXT NOT NULL,
                url TEXT NOT NULL,
                added_at INTEGER,
                FOREIGN KEY (provider_id, app_type) REFERENCES providers(id, app_type) ON DELETE CASCADE
            );
CREATE TABLE provider_health (
            provider_id TEXT NOT NULL, app_type TEXT NOT NULL, is_healthy INTEGER NOT NULL DEFAULT 1,
            consecutive_failures INTEGER NOT NULL DEFAULT 0, last_success_at TEXT, last_failure_at TEXT,
            last_error TEXT, updated_at TEXT NOT NULL,
            PRIMARY KEY (provider_id, app_type),
            FOREIGN KEY (provider_id, app_type) REFERENCES providers(id, app_type) ON DELETE CASCADE
        );
CREATE TABLE providers (
                id TEXT NOT NULL,
                app_type TEXT NOT NULL,
                name TEXT NOT NULL,
                settings_config TEXT NOT NULL,
                website_url TEXT,
                category TEXT,
                created_at INTEGER,
                sort_index INTEGER,
                notes TEXT,
                icon TEXT,
                icon_color TEXT,
                meta TEXT NOT NULL DEFAULT '{}',
                is_current BOOLEAN NOT NULL DEFAULT 0,
                in_failover_queue BOOLEAN NOT NULL DEFAULT 0, "cost_multiplier" TEXT NOT NULL DEFAULT '1.0', "limit_daily_usd" TEXT, "limit_monthly_usd" TEXT, "provider_type" TEXT,
                PRIMARY KEY (id, app_type)
            );
CREATE TABLE proxy_config (
            app_type TEXT PRIMARY KEY CHECK (app_type IN ('claude','codex','gemini')),
            proxy_enabled INTEGER NOT NULL DEFAULT 0, listen_address TEXT NOT NULL DEFAULT '127.0.0.1',
            listen_port INTEGER NOT NULL DEFAULT 15721, enable_logging INTEGER NOT NULL DEFAULT 1,
            enabled INTEGER NOT NULL DEFAULT 0, auto_failover_enabled INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 3, streaming_first_byte_timeout INTEGER NOT NULL DEFAULT 60,
            streaming_idle_timeout INTEGER NOT NULL DEFAULT 120, non_streaming_timeout INTEGER NOT NULL DEFAULT 600,
            circuit_failure_threshold INTEGER NOT NULL DEFAULT 4, circuit_success_threshold INTEGER NOT NULL DEFAULT 2,
            circuit_timeout_seconds INTEGER NOT NULL DEFAULT 60, circuit_error_rate_threshold REAL NOT NULL DEFAULT 0.6,
            circuit_min_requests INTEGER NOT NULL DEFAULT 10,
            default_cost_multiplier TEXT NOT NULL DEFAULT '1',
            pricing_model_source TEXT NOT NULL DEFAULT 'response',
            created_at TEXT NOT NULL DEFAULT (datetime('now')), updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        , live_takeover_active INTEGER NOT NULL DEFAULT 0);
CREATE TABLE proxy_live_backup (
            app_type TEXT PRIMARY KEY, original_config TEXT NOT NULL, backed_up_at TEXT NOT NULL
        );
CREATE TABLE proxy_request_logs (
            request_id TEXT PRIMARY KEY, provider_id TEXT NOT NULL, app_type TEXT NOT NULL, model TEXT NOT NULL,
            request_model TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0, cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
            input_cost_usd TEXT NOT NULL DEFAULT '0', output_cost_usd TEXT NOT NULL DEFAULT '0',
            cache_read_cost_usd TEXT NOT NULL DEFAULT '0', cache_creation_cost_usd TEXT NOT NULL DEFAULT '0',
            total_cost_usd TEXT NOT NULL DEFAULT '0', latency_ms INTEGER NOT NULL, first_token_ms INTEGER,
            duration_ms INTEGER, status_code INTEGER NOT NULL, error_message TEXT, session_id TEXT,
            provider_type TEXT, is_streaming INTEGER NOT NULL DEFAULT 0,
            cost_multiplier TEXT NOT NULL DEFAULT '1.0', created_at INTEGER NOT NULL
        );
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE skill_repos (
            owner TEXT NOT NULL, name TEXT NOT NULL, branch TEXT NOT NULL DEFAULT 'main',
            enabled BOOLEAN NOT NULL DEFAULT 1, PRIMARY KEY (owner, name)
        );
CREATE TABLE skills (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            directory TEXT NOT NULL,
            repo_owner TEXT,
            repo_name TEXT,
            repo_branch TEXT DEFAULT 'main',
            readme_url TEXT,
            enabled_claude BOOLEAN NOT NULL DEFAULT 0,
            enabled_codex BOOLEAN NOT NULL DEFAULT 0,
            enabled_gemini BOOLEAN NOT NULL DEFAULT 0,
            enabled_opencode BOOLEAN NOT NULL DEFAULT 0,
            installed_at INTEGER NOT NULL DEFAULT 0
        );
CREATE TABLE stream_check_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, provider_id TEXT NOT NULL, provider_name TEXT NOT NULL,
            app_type TEXT NOT NULL, status TEXT NOT NULL, success INTEGER NOT NULL, message TEXT NOT NULL,
            response_time_ms INTEGER, http_status INTEGER, model_used TEXT,
            retry_count INTEGER DEFAULT 0, tested_at INTEGER NOT NULL
        );
CREATE TABLE usage_daily_rollups (
                date TEXT NOT NULL,
                app_type TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                model TEXT NOT NULL,
                request_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
                total_cost_usd TEXT NOT NULL DEFAULT '0',
                avg_latency_ms INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (date, app_type, provider_id, model)
            );
CREATE INDEX idx_providers_failover
             ON providers(app_type, in_failover_queue, sort_index);
CREATE INDEX idx_request_logs_created_at ON proxy_request_logs(created_at);
CREATE INDEX idx_request_logs_model ON proxy_request_logs(model);
CREATE INDEX idx_request_logs_provider ON proxy_request_logs(provider_id, app_type);
CREATE INDEX idx_request_logs_session ON proxy_request_logs(session_id);
CREATE INDEX idx_request_logs_status ON proxy_request_logs(status_code);
CREATE INDEX idx_stream_check_logs_provider
             ON stream_check_logs(app_type, provider_id, tested_at DESC);
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('claude-opus-4-5-20251101', 'Claude Opus 4.5', '5', '25', '0.50', '6.25');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('claude-sonnet-4-5-20250929', 'Claude Sonnet 4.5', '3', '15', '0.30', '3.75');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('claude-haiku-4-5-20251001', 'Claude Haiku 4.5', '1', '5', '0.10', '1.25');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('claude-opus-4-20250514', 'Claude Opus 4', '15', '75', '1.50', '18.75');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('claude-opus-4-1-20250805', 'Claude Opus 4.1', '15', '75', '1.50', '18.75');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('claude-sonnet-4-20250514', 'Claude Sonnet 4', '3', '15', '0.30', '3.75');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('claude-3-5-haiku-20241022', 'Claude 3.5 Haiku', '0.80', '4', '0.08', '1');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('claude-3-5-sonnet-20241022', 'Claude 3.5 Sonnet', '3', '15', '0.30', '3.75');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.2', 'GPT-5.2', '1.75', '14', '0.175', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.2-low', 'GPT-5.2', '1.75', '14', '0.175', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.2-medium', 'GPT-5.2', '1.75', '14', '0.175', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.2-high', 'GPT-5.2', '1.75', '14', '0.175', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.2-xhigh', 'GPT-5.2', '1.75', '14', '0.175', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.2-codex', 'GPT-5.2 Codex', '1.75', '14', '0.175', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.2-codex-low', 'GPT-5.2 Codex', '1.75', '14', '0.175', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.2-codex-medium', 'GPT-5.2 Codex', '1.75', '14', '0.175', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.2-codex-high', 'GPT-5.2 Codex', '1.75', '14', '0.175', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.2-codex-xhigh', 'GPT-5.2 Codex', '1.75', '14', '0.175', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.1', 'GPT-5.1', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.1-low', 'GPT-5.1', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.1-medium', 'GPT-5.1', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.1-high', 'GPT-5.1', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.1-minimal', 'GPT-5.1', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.1-codex', 'GPT-5.1 Codex', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.1-codex-mini', 'GPT-5.1 Codex', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.1-codex-max', 'GPT-5.1 Codex', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.1-codex-max-high', 'GPT-5.1 Codex', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5.1-codex-max-xhigh', 'GPT-5.1 Codex', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5', 'GPT-5', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5-low', 'GPT-5', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5-medium', 'GPT-5', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5-high', 'GPT-5', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5-minimal', 'GPT-5', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5-codex', 'GPT-5 Codex', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5-codex-low', 'GPT-5 Codex', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5-codex-medium', 'GPT-5 Codex', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5-codex-high', 'GPT-5 Codex', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5-codex-mini', 'GPT-5 Codex', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5-codex-mini-medium', 'GPT-5 Codex', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gpt-5-codex-mini-high', 'GPT-5 Codex', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gemini-3-pro-preview', 'Gemini 3 Pro Preview', '2', '12', '0.2', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gemini-3-flash-preview', 'Gemini 3 Flash Preview', '0.5', '3', '0.05', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gemini-2.5-pro', 'Gemini 2.5 Pro', '1.25', '10', '0.125', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('gemini-2.5-flash', 'Gemini 2.5 Flash', '0.3', '2.5', '0.03', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('doubao-seed-code', 'Doubao Seed Code', '1.20', '8.00', '0.24', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('deepseek-v3.2', 'DeepSeek V3.2', '2.00', '3.00', '0.40', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('deepseek-v3.1', 'DeepSeek V3.1', '4.00', '12.00', '0.80', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('deepseek-v3', 'DeepSeek V3', '2.00', '8.00', '0.40', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('kimi-k2-thinking', 'Kimi K2 Thinking', '4.00', '16.00', '1.00', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('kimi-k2-0905', 'Kimi K2', '4.00', '16.00', '1.00', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('kimi-k2-turbo', 'Kimi K2 Turbo', '8.00', '58.00', '1.00', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('minimax-m2.1', 'MiniMax M2.1', '2.10', '8.40', '0.21', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('minimax-m2.1-lightning', 'MiniMax M2.1 Lightning', '2.10', '16.80', '0.21', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('minimax-m2', 'MiniMax M2', '2.10', '8.40', '0.21', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('glm-4.7', 'GLM-4.7', '2.00', '8.00', '0.40', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('glm-4.6', 'GLM-4.6', '2.00', '8.00', '0.40', '0');
INSERT INTO "model_pricing" ("model_id", "display_name", "input_cost_per_million", "output_cost_per_million", "cache_read_cost_per_million", "cache_creation_cost_per_million") VALUES ('mimo-v2-flash', 'Mimo V2 Flash', '0', '0', '0', '0');
INSERT INTO "providers" ("id", "app_type", "name", "settings_config", "website_url", "category", "created_at", "sort_index", "notes", "icon", "icon_color", "meta", "is_current", "in_failover_queue", "cost_multiplier", "limit_daily_usd", "limit_monthly_usd", "provider_type") VALUES ('openai', 'claude', 'OpenAI', '{"env":{"ANTHROPIC_AUTH_TOKEN":"sk-c2c080cee3a03919ab05488ed464d6563c1c701278c1ab35522439c014d07f62","ANTHROPIC_BASE_URL":"https://ai.dooo.ng","ANTHROPIC_DEFAULT_HAIKU_MODEL":"GPT-5.4","ANTHROPIC_DEFAULT_OPUS_MODEL":"GPT-5.4","ANTHROPIC_DEFAULT_SONNET_MODEL":"GPT-5.4","ANTHROPIC_MODEL":"GPT-5.4"}}', 'https://ai.dooo.ng', NULL, 1775551494, NULL, NULL, NULL, NULL, '{}', 1, 0, '1.0', NULL, NULL, NULL);
INSERT INTO "providers" ("id", "app_type", "name", "settings_config", "website_url", "category", "created_at", "sort_index", "notes", "icon", "icon_color", "meta", "is_current", "in_failover_queue", "cost_multiplier", "limit_daily_usd", "limit_monthly_usd", "provider_type") VALUES ('openai', 'codex', 'openai', '{"auth":{"OPENAI_API_KEY":"sk-c2c080cee3a03919ab05488ed464d6563c1c701278c1ab35522439c014d07f62","auth_mode":"apikey"},"config":"model_provider = \"custom\"\nmodel = \"gpt-5.4\"\nmodel_reasoning_effort = \"xhigh\"\nservice_tier = \"fast\"\n\n[model_providers.custom]\nname = \"custom\"\nbase_url = \"https://ai.dooo.ng/v1\"\nwire_api = \"responses\"\nrequires_openai_auth = true\n\n[projects.\"/opt/data/private/tjl/code/Time-Series-Library-fm\"]\ntrust_level = \"trusted\"\n\n[projects.\"/opt/data/private/RAG/think_like_graph\"]\ntrust_level = \"trusted\"\n"}', 'https://ai.dooo.ng', NULL, 1775551848, NULL, NULL, NULL, NULL, '{}', 0, 0, '1.0', NULL, NULL, NULL);
INSERT INTO "providers" ("id", "app_type", "name", "settings_config", "website_url", "category", "created_at", "sort_index", "notes", "icon", "icon_color", "meta", "is_current", "in_failover_queue", "cost_multiplier", "limit_daily_usd", "limit_monthly_usd", "provider_type") VALUES ('by2', 'codex', 'by2', '{"auth":{"OPENAI_API_KEY":"sk-WA5q5xN2eYUHFv9rmzT2CFo6rXoPan5mWVw7hNqNcPs1Mxfb"},"config":"model_provider = \"custom\"\nmodel = \"gpt-5.4\"\n\n[model_providers.custom]\nname = \"custom\"\nbase_url = \"https://pro.ominiai.cn/v1\"\nwire_api = \"responses\"\nrequires_openai_auth = true\n\n[projects.\"/opt/data/private/tjl/code/Time-Series-Library-fm\"]\ntrust_level = \"trusted\"\n"}', NULL, NULL, 1775623835, NULL, NULL, NULL, NULL, '{}', 0, 0, '1.0', NULL, NULL, NULL);
INSERT INTO "providers" ("id", "app_type", "name", "settings_config", "website_url", "category", "created_at", "sort_index", "notes", "icon", "icon_color", "meta", "is_current", "in_failover_queue", "cost_multiplier", "limit_daily_usd", "limit_monthly_usd", "provider_type") VALUES ('by3', 'codex', 'by3', '{"auth":{"OPENAI_API_KEY":"sk-WzyF7fbZOhtcplkyxDWUyj7etQT7HtRljEu1dIvplvoY6r7Q","auth_mode":"apikey"},"config":"model_provider = \"custom\"\nmodel = \"gpt-5.4\"\nservice_tier = \"fast\"\n\n[model_providers.custom]\nname = \"custom\"\nbase_url = \"http://new.xem8k5.top:3000/v1\"\nwire_api = \"responses\"\nrequires_openai_auth = true\n\n[projects.\"/opt/data/private/tjl/code/Time-Series-Library-fm\"]\ntrust_level = \"trusted\"\n\n[projects.\"/opt/data/private/RAG/think_like_graph\"]\ntrust_level = \"trusted\"\n"}', NULL, NULL, 1775628963, NULL, NULL, NULL, NULL, '{}', 1, 0, '1.0', NULL, NULL, NULL);
INSERT INTO "providers" ("id", "app_type", "name", "settings_config", "website_url", "category", "created_at", "sort_index", "notes", "icon", "icon_color", "meta", "is_current", "in_failover_queue", "cost_multiplier", "limit_daily_usd", "limit_monthly_usd", "provider_type") VALUES ('by4', 'codex', 'by4', '{"auth":{"OPENAI_API_KEY":"sk-80d79a1b5c58edf11754050a4b85240bd3dad15767acc52e1162a0c48f158429"},"config":"model_provider = \"custom\"\nmodel = \"gpt-5.4\"\n\n[model_providers.custom]\nname = \"custom\"\nbase_url = \"https://api.99dun.cc/v1\"\nwire_api = \"responses\"\nrequires_openai_auth = true\n\n[projects.\"/opt/data/private/tjl/code/Time-Series-Library-fm\"]\ntrust_level = \"trusted\"\n"}', NULL, NULL, 1775728646, NULL, NULL, NULL, NULL, '{}', 0, 0, '1.0', NULL, NULL, NULL);
INSERT INTO "providers" ("id", "app_type", "name", "settings_config", "website_url", "category", "created_at", "sort_index", "notes", "icon", "icon_color", "meta", "is_current", "in_failover_queue", "cost_multiplier", "limit_daily_usd", "limit_monthly_usd", "provider_type") VALUES ('by5', 'codex', 'by5', '{"auth":{"OPENAI_API_KEY":"sk-EJpj1v3na5TI2EMO9YjWX98iKMjBoOv5cuFgJdbEOQp3w4Uw"},"config":"model_provider = \"custom\"\nmodel = \"gpt-5.4\"\n\n[model_providers.custom]\nname = \"custom\"\nbase_url = \"https://api.apiplus.cloud/v1\"\nwire_api = \"responses\"\nrequires_openai_auth = true\n\n[projects.\"/opt/data/private/tjl/code/Time-Series-Library-fm\"]\ntrust_level = \"trusted\"\n"}', NULL, NULL, 1775735021, NULL, NULL, NULL, NULL, '{}', 0, 0, '1.0', NULL, NULL, NULL);
INSERT INTO "proxy_config" ("app_type", "proxy_enabled", "listen_address", "listen_port", "enable_logging", "enabled", "auto_failover_enabled", "max_retries", "streaming_first_byte_timeout", "streaming_idle_timeout", "non_streaming_timeout", "circuit_failure_threshold", "circuit_success_threshold", "circuit_timeout_seconds", "circuit_error_rate_threshold", "circuit_min_requests", "default_cost_multiplier", "pricing_model_source", "created_at", "updated_at", "live_takeover_active") VALUES ('claude', 0, '127.0.0.1', 15721, 1, 0, 0, 6, 90, 180, 600, 8, 3, 90, 0.7, 15, '1', 'response', '2026-04-07 08:40:38', '2026-04-07 08:40:38', 0);
INSERT INTO "proxy_config" ("app_type", "proxy_enabled", "listen_address", "listen_port", "enable_logging", "enabled", "auto_failover_enabled", "max_retries", "streaming_first_byte_timeout", "streaming_idle_timeout", "non_streaming_timeout", "circuit_failure_threshold", "circuit_success_threshold", "circuit_timeout_seconds", "circuit_error_rate_threshold", "circuit_min_requests", "default_cost_multiplier", "pricing_model_source", "created_at", "updated_at", "live_takeover_active") VALUES ('codex', 0, '127.0.0.1', 15721, 1, 0, 0, 3, 60, 120, 600, 4, 2, 60, 0.6, 10, '1', 'response', '2026-04-07 08:40:38', '2026-04-07 08:40:38', 0);
INSERT INTO "proxy_config" ("app_type", "proxy_enabled", "listen_address", "listen_port", "enable_logging", "enabled", "auto_failover_enabled", "max_retries", "streaming_first_byte_timeout", "streaming_idle_timeout", "non_streaming_timeout", "circuit_failure_threshold", "circuit_success_threshold", "circuit_timeout_seconds", "circuit_error_rate_threshold", "circuit_min_requests", "default_cost_multiplier", "pricing_model_source", "created_at", "updated_at", "live_takeover_active") VALUES ('gemini', 0, '127.0.0.1', 15721, 1, 0, 0, 5, 60, 120, 600, 4, 2, 60, 0.6, 10, '1', 'response', '2026-04-07 08:40:38', '2026-04-07 08:40:38', 0);
INSERT INTO "settings" ("key", "value") VALUES ('common_config_codex', 'model_reasoning_effort = "high"
disable_response_storage = true');
INSERT INTO "skill_repos" ("owner", "name", "branch", "enabled") VALUES ('anthropics', 'skills', 'main', 1);
INSERT INTO "skill_repos" ("owner", "name", "branch", "enabled") VALUES ('ComposioHQ', 'awesome-claude-skills', 'master', 1);
INSERT INTO "skill_repos" ("owner", "name", "branch", "enabled") VALUES ('cexll', 'myclaude', 'master', 1);
INSERT INTO "skill_repos" ("owner", "name", "branch", "enabled") VALUES ('JimLiu', 'baoyu-skills', 'main', 1);
INSERT INTO "stream_check_logs" ("id", "provider_id", "provider_name", "app_type", "status", "success", "message", "response_time_ms", "http_status", "model_used", "retry_count", "tested_at") VALUES (1, 'openai', 'OpenAI', 'claude', 'operational', 1, 'Check succeeded', 937, 200, 'GPT-5.4', 0, 1775551562);
COMMIT;
PRAGMA foreign_keys=ON;
