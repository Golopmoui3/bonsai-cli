# Contributing to Bonsai CLI

Thanks for wanting to help improve **Bonsai CLI**! Whether you are fixing an edge case in `MarkdownStreamer`, hardening tool guardrails, improving Task-Aware Context Compression (TAC), or adding new `llama-server` tuning flags, this guide explains how to work safely in the codebase.

---

## Development Setup (Windows x64)

1. **Install Rust (2021 edition)** via `rustup` (MSVC or MinGW-w64 toolchain).
2. Clone the repository and verify tests pass with locked dependencies:

```powershell
git clone https://github.com/Golopmoui3/bonsai-cli.git
cd bonsai-cli
cargo test --locked
```

3. Build the release executable:

```powershell
cargo build --release --locked
.\target\release\bonsai.exe --help
```

---

## Core Design Principles & Invariants

Before opening a pull request, make sure your changes preserve these non-negotiable guarantees:

1. **Zero Secret Leaks (`redact_proxy` / `redact_secrets`)**:
   - Proxy credentials (`user:pass@host`) and VPN subscription tokens must **never** appear unmasked in terminal output, REPL history (`history.txt`), `edit_file`/`write_file` disk writes, Markdown exports, or model tool results.
   - Keep `check_write_guardrails` active so agent tools cannot overwrite Bonsai's internal PowerShell proxy shim (`proxybin`).
2. **UTF-8 & Windows Line Endings**:
   - Respect UTF-8 codepage (`65001`), multibyte char boundaries (use `.chars().count()` / byte-boundary checks when slicing strings with Cyrillic/Unicode text), and preserve CRLF + UTF-8 BOM when editing `.ps1` files (`tool_edit_file`).
3. **Conservative TAC Compaction**:
   - Context compaction (`compress_task_aware`) must never increase total estimated tokens across any mode (`Light`, `Medium`, `Heavy`), must preserve role alternation, and must keep the protected recent window intact.
4. **Accurate Tool Schemas (`tools_spec`)**:
   - Any change to `tool_*` arguments in `App` must stay synchronized with the JSON schema returned by `tools_spec()`.
5. **Avoid Extra Crates Unless Necessary**:
   - `Cargo.toml` intentionally keeps the dependency footprint lean with offline/rustls builds in mind. Prefer stdlib or existing crates where reasonable.

---

## Testing Expectations

Every bug fix or feature must ship with unit tests inside `mod security_audit_tests` in [`src/main.rs`](src/main.rs):

- Run the full suite before committing:
  ```powershell
  cargo test --locked --all-targets
  ```
- Ensure zero compiler warnings under default `cargo check --all-targets`.
- For REPL/CLI flag changes, smoke-test `--help` and `--no-server` piped commands to verify backwards compatibility.

---

## Commit Convention

Follow Conventional Commits so changelog and history remain clean:

- `feat(<scope>): ...` — new CLI flags, REPL commands, agent tools, or engine capabilities
- `fix(<scope>): ...` — bug fixes in streaming, TAC, proxy, or guardrails
- `docs: ...` — `README.md`, `CONTRIBUTING.md`, or inline documentation updates
- `ci: ...` — GitHub Actions workflow adjustments
- `test: ...` — additional unit or regression coverage

---

## Pull Request Checklist

- [ ] `cargo test --locked` passes with 0 failures and 0 warnings
- [ ] New CLI flags/commands are documented in `print_help()` and `README.md`
- [ ] `BonsaiHelper` Tab completion includes any newly added slash commands
- [ ] No real tokens, IPs, credentials, logs, or model binaries are committed
