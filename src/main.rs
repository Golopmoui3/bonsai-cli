//! bonsai — lightweight streaming CLI for Ternary-Bonsai (llama.cpp Prism ML fork).

use anyhow::{bail, Context, Result};
use base64::engine::general_purpose::{STANDARD, URL_SAFE};
use base64::Engine as _;
use clap::{Parser, ValueEnum};
use futures_util::StreamExt;
use rustyline::completion::{Completer, FilenameCompleter, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::{Hinter, HistoryHinter};
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{CompletionType, Config, Context as RlContext, Editor, Helper};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::borrow::Cow::{self, Owned};
use std::collections::BTreeSet;
use std::ffi::{OsStr, OsString};
use std::io::{IsTerminal, Read, Seek, SeekFrom, Write};
use tokio::io::AsyncReadExt;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const CREATE_NO_WINDOW: u32 = 0x0800_0000;
const DEFAULT_MODEL: &str =
    r"C:\Bonsai\models\Ternary-Bonsai-2-27B-Uncensored-Heretic-PTQ1_0.gguf";
const DEFAULT_SERVER: &str = r"C:\Bonsai\llama\llama-server.exe";
const DEFAULT_MMPROJ: &str = r"C:\Bonsai\vision\mmproj-Q8_0.gguf";
const LOG_PATH: &str = r"C:\Bonsai\llama\server.log";

#[cfg(windows)]
fn enable_ansi_support() {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetStdHandle(nStdHandle: u32) -> *mut std::ffi::c_void;
        fn GetConsoleMode(hConsoleHandle: *mut std::ffi::c_void, lpMode: *mut u32) -> i32;
        fn SetConsoleMode(hConsoleHandle: *mut std::ffi::c_void, dwMode: u32) -> i32;
        fn SetConsoleOutputCP(wCodePageID: u32) -> i32;
        fn SetConsoleCP(wCodePageID: u32) -> i32;
    }

    const STD_OUTPUT_HANDLE: u32 = 0xFFFFFFF5;
    const STD_ERROR_HANDLE: u32 = 0xFFFFFFF4;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;

    unsafe {
        let _ = SetConsoleOutputCP(65001);
        let _ = SetConsoleCP(65001);

        for handle_id in [STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
            let handle = GetStdHandle(handle_id);
            if !handle.is_null() && handle as isize != -1 {
                let mut mode = 0u32;
                if GetConsoleMode(handle, &mut mode) != 0 {
                    let _ = SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
                }
            }
        }
    }
}

#[cfg(windows)]
fn get_console_width() -> Option<usize> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetStdHandle(nStdHandle: u32) -> *mut std::ffi::c_void;
        fn GetConsoleScreenBufferInfo(
            hConsoleOutput: *mut std::ffi::c_void,
            lpConsoleScreenBufferInfo: *mut CONSOLE_SCREEN_BUFFER_INFO,
        ) -> i32;
    }
    const STD_OUTPUT_HANDLE: u32 = 0xFFFFFFF5;
    #[repr(C)]
    struct COORD {
        x: i16,
        y: i16,
    }
    #[repr(C)]
    struct SMALL_RECT {
        left: i16,
        top: i16,
        right: i16,
        bottom: i16,
    }
    #[repr(C)]
    #[allow(non_snake_case)]
    struct CONSOLE_SCREEN_BUFFER_INFO {
        dwSize: COORD,
        dwCursorPosition: COORD,
        wAttributes: u16,
        srWindow: SMALL_RECT,
        dwMaximumWindowSize: COORD,
    }

    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut info: CONSOLE_SCREEN_BUFFER_INFO = std::mem::zeroed();
        if GetConsoleScreenBufferInfo(handle, &mut info) != 0 {
            let width = info.srWindow.right - info.srWindow.left + 1;
            if width > 20 {
                return Some(width as usize);
            }
        }
    }
    None
}

fn terminal_width() -> usize {
    #[cfg(windows)]
    {
        get_console_width().unwrap_or(80)
    }
    #[cfg(not(windows))]
    {
        80
    }
}

static LAST_BADGE_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn update_top_right_session_badge(session_tokens: usize, budget: Option<usize>) {
    if !std::io::stderr().is_terminal() {
        return;
    }
    let badge = if let Some(b) = budget {
        format!("[Session: {session_tokens}/{b} tok]")
    } else {
        format!("[Session: {session_tokens} tok]")
    };
    #[cfg(windows)]
    if let Some(width) = get_console_width() {
        let badge_chars = badge.chars().count();
        let prev_chars = LAST_BADGE_LEN.swap(badge_chars, Ordering::SeqCst);
        let max_len = prev_chars.max(badge_chars);
        let start_col = width.saturating_sub(max_len).max(1);
        let left_erase = " ".repeat(prev_chars.saturating_sub(badge_chars));
        eprint!(
            "\x1b[s\x1b[1;{start_col}H{left_erase}\x1b[1;36m{badge}\x1b[0m\x1b[u\x1b]0;Bonsai | {badge}\x07"
        );
        let _ = std::io::stderr().flush();
    }
}

#[cfg(not(windows))]
fn enable_ansi_support() {}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Effort {
    #[value(name = "none")]
    None,
    #[value(name = "low")]
    Low,
    #[value(name = "medium")]
    Medium,
    #[value(name = "high")]
    High,
    #[value(name = "xhigh")]
    XHigh,
}

impl Effort {
    fn as_str(self) -> &'static str {
        match self {
            Effort::None => "none",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::XHigh => "xhigh",
        }
    }
    fn thinking(self) -> bool {
        !matches!(self, Effort::None)
    }
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" => Some(Effort::None),
            "low" => Some(Effort::Low),
            "medium" => Some(Effort::Medium),
            "high" => Some(Effort::High),
            "xhigh" => Some(Effort::XHigh),
            _ => None,
        }
    }
}

fn parse_ctx(s: &str) -> Result<u32, String> {
    let lower = s.trim().to_ascii_lowercase();
    let val: u32 = if let Some(k) = lower.strip_suffix('k') {
        let n: u32 = k.trim().parse().map_err(|e| format!("invalid context number '{s}': {e}"))?;
        n.checked_mul(1024).ok_or_else(|| format!("context size '{s}' overflows u32"))?
    } else {
        lower.parse().map_err(|e| format!("invalid context number '{s}': {e}"))?
    };
    if val < 512 {
        return Err(format!("context size '{s}' is too small (min 512)"));
    }
    Ok(val)
}

fn parse_token_limit(s: &str) -> Result<usize, String> {
    let lower = s.trim().to_ascii_lowercase();
    let val: usize = if let Some(k) = lower.strip_suffix('k') {
        let n: usize = k.trim().parse().map_err(|e| format!("invalid token count '{s}': {e}"))?;
        n.checked_mul(1024).ok_or_else(|| format!("token count '{s}' overflows"))?
    } else if let Some(m) = lower.strip_suffix('m') {
        let n: usize = m.trim().parse().map_err(|e| format!("invalid token count '{s}': {e}"))?;
        n.checked_mul(1024 * 1024).ok_or_else(|| format!("token count '{s}' overflows"))?
    } else {
        lower.parse().map_err(|e| format!("invalid token count '{s}': {e}"))?
    };
    if val == 0 {
        return Err(format!("token limit '{s}' must be >= 1"));
    }
    Ok(val)
}

fn parse_tps(s: &str) -> Result<f64, String> {
    let r: f64 = s.trim().parse().map_err(|e| format!("invalid tok/s '{s}': {e}"))?;
    if !r.is_finite() || r < 0.1 || r > 5000.0 {
        return Err(format!("tokens/sec '{s}' must be between 0.1 and 5000"));
    }
    Ok(r)
}

fn parse_kv_type(s: &str) -> Result<String, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "f16" | "fp16" | "default" => Ok("f16".to_string()),
        "bf16" => Ok("bf16".to_string()),
        "q8" | "q8_0" => Ok("q8_0".to_string()),
        "q4" | "q4_0" => Ok("q4_0".to_string()),
        "q5" | "q5_0" => Ok("q5_0".to_string()),
        "iq4_nl" => Ok("iq4_nl".to_string()),
        other => Err(format!("unsupported KV cache type '{other}' (use f16, bf16, q8_0, q4_0, q5_0, iq4_nl)")),
    }
}

fn parse_spec_type(s: &str) -> Result<String, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "none" | "off" => Ok("none".to_string()),
        "on" | "ngram" | "ngram-cache" | "cache" => Ok("ngram-cache".to_string()),
        "ngram-simple" | "simple" => Ok("ngram-simple".to_string()),
        "ngram-mod" | "mod" => Ok("ngram-mod".to_string()),
        "ngram-map-k" => Ok("ngram-map-k".to_string()),
        other => Err(format!("unsupported spec type '{other}' (use off, ngram-cache, ngram-simple, ngram-mod)")),
    }
}

#[derive(Parser, Debug)]
#[command(name = "bonsai", version, about = "Chat with Ternary-Bonsai through llama-server")]
struct Cli {
    /// Path to the GGUF model
    #[arg(short = 'm', long, default_value = DEFAULT_MODEL)]
    model: String,

    /// Context size (e.g. 8192, 16k, 32k, 32768)
    #[arg(long, default_value = "8192", value_parser = parse_ctx)]
    ctx: u32,

    /// GPU layers
    #[arg(long, default_value_t = 99)]
    ngl: u32,

    /// CPU threads (llama default when omitted)
    #[arg(short = 't', long)]
    threads: Option<u32>,

    /// llama-server port
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Reasoning effort
    #[arg(long, visible_alias = "reasoning", value_enum, default_value_t = Effort::XHigh)]
    effort: Effort,

    /// System prompt
    #[arg(short = 's', long)]
    system: Option<String>,

    /// Sampling temperature (overrides preset)
    #[arg(long)]
    temp: Option<f32>,

    /// top_p (overrides preset)
    #[arg(long = "top-p")]
    top_p: Option<f32>,

    /// top_k (overrides preset)
    #[arg(long = "top-k")]
    top_k: Option<i32>,

    /// min_p (overrides preset)
    #[arg(long = "min-p")]
    min_p: Option<f32>,

    /// presence_penalty (overrides preset)
    #[arg(long)]
    presence: Option<f32>,

    /// Attach an image to the prompt
    #[arg(long)]
    image: Option<PathBuf>,

    /// Single prompt ("-" reads stdin); prints answer and exits
    #[arg(short = 'p', long)]
    prompt: Option<String>,

    /// Hide reasoning output
    #[arg(long)]
    hide_reasoning: bool,

    /// Do not print the stats line
    #[arg(long)]
    no_stats: bool,

    /// Connect to an already running server; never spawn one
    #[arg(long)]
    no_server: bool,

    /// Do not stop the server on exit
    #[arg(long)]
    keep_alive: bool,

    /// Kill the server recorded in the pid file and exit
    #[arg(long)]
    stop: bool,

    /// Start the interactive agent CLI
    #[arg(long)]
    start: bool,

    /// Enable agent tools in single-shot mode
    #[arg(long, conflicts_with = "no_agent")]
    agent: bool,

    /// Disable agent tools (plain chat)
    #[arg(long = "no-agent")]
    no_agent: bool,

    /// Auto-approve every tool call (dangerous)
    #[arg(long)]
    bypass: bool,

    /// Route agent command traffic through this proxy (http/https/socks5 URL)
    #[arg(long, value_name = "URL")]
    proxy: Option<String>,

    /// Session token budget limit (e.g. 50k, 100000; default inactive)
    #[arg(long, value_parser = parse_token_limit, value_name = "TOKENS")]
    budget: Option<usize>,

    /// Reasoning / thinking token budget per response (e.g. 2k, 4096; default unlimited)
    #[arg(long = "think-budget", value_parser = parse_token_limit, value_name = "TOKENS")]
    think_budget: Option<usize>,

    /// Generation speed limit in tokens per second (e.g. 20; default unlimited)
    #[arg(long, value_parser = parse_tps, value_name = "TPS")]
    tokens: Option<f64>,

    /// Hardware / GPU load limit percentage (1..99%, e.g. 50; default unlimited)
    #[arg(long, visible_alias = "throttle", value_parser = clap::value_parser!(u8).range(1..=99), value_name = "PERCENT")]
    load: Option<u8>,

    /// KV cache quantization type for K/V (f16, bf16, q8_0, q4_0)
    #[arg(long = "kv", value_parser = parse_kv_type, value_name = "TYPE")]
    kv: Option<String>,

    /// Speculative decoding without draft model (ngram-cache, ngram-simple, none)
    #[arg(long = "spec", value_parser = parse_spec_type, value_name = "TYPE")]
    spec: Option<String>,

    /// Disable markdown formatting in terminal output (print raw markdown)
    #[arg(long = "no-markdown")]
    no_markdown: bool,

    /// llama-server executable
    #[arg(long, default_value = DEFAULT_SERVER, hide = true)]
    server_bin: String,
}

// ---------------------------------------------------------------------------
// Paths / pid file
// ---------------------------------------------------------------------------

fn home() -> PathBuf {
    PathBuf::from(std::env::var("USERPROFILE").unwrap_or_else(|_| ".".into()))
}

fn bonsai_dir() -> PathBuf {
    home().join(".bonsai")
}

fn pid_path_for(port: u16) -> PathBuf {
    if port == 8080 {
        bonsai_dir().join("server.pid")
    } else {
        bonsai_dir().join(format!("server-{port}.pid"))
    }
}

fn history_path() -> PathBuf {
    bonsai_dir().join("history.txt")
}

fn ensure_dirs() -> Result<()> {
    std::fs::create_dir_all(bonsai_dir()).context("create ~/.bonsai")?;
    let clip = bonsai_dir().join("clipboard.ps1");
    if !clip.exists() {
        let _ = std::fs::write(&clip, CLIPBOARD_PS1);
    }
    let wrap = ps_wrapper_path();
    let wrapper = format!("{PS_PROXY_PRELUDE}\n{PS_FILE_WRAPPER_BODY}");
    if !wrap.exists() || std::fs::read_to_string(&wrap).map(|s| s != wrapper).unwrap_or(true) {
        let _ = std::fs::write(&wrap, wrapper);
    }
    Ok(())
}

fn proxy_shim_dir() -> PathBuf {
    bonsai_dir().join("proxybin")
}

fn ps_wrapper_path() -> PathBuf {
    bonsai_dir().join("ps_proxy_file.ps1")
}

/// A private powershell.exe shim is placed first in PATH for agent commands.
/// It injects the proxy prelude into nested PowerShell processes, including
/// `powershell -File script.ps1`, where the parent process settings cannot
/// be inherited. The shim is a hard link/copy of this executable.
fn ensure_ps_shim(force: bool) -> Result<()> {
    let dir = proxy_shim_dir();
    std::fs::create_dir_all(&dir).context("create proxy shim directory")?;
    let target = dir.join("powershell.exe");
    let exe = std::env::current_exe().context("locate bonsai executable")?;
    if !force {
        if let (Ok(t), Ok(e)) = (std::fs::metadata(&target), std::fs::metadata(&exe)) {
            if t.len() == e.len() && t.len() > 0 {
                return Ok(());
            }
        }
    }
    let _ = std::fs::remove_file(&target);
    match std::fs::hard_link(&exe, &target) {
        Ok(()) => {}
        Err(_) => {
            std::fs::copy(&exe, &target)
                .with_context(|| format!("copy proxy shim to {}", target.display()))?;
        }
    }
    Ok(())
}

fn real_powershell_exe() -> PathBuf {
    let candidate = std::env::var_os("SystemRoot")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"))
        .join("System32")
        .join("WindowsPowerShell")
        .join("v1.0")
        .join("powershell.exe");
    if candidate.is_file() {
        candidate
    } else {
        PathBuf::from(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
    }
}

fn invoked_as_ps_shim() -> bool {
    let mut args = std::env::args_os();
    let argv0_is_shim = args
        .next()
        .as_deref()
        .and_then(|p| Path::new(p).file_name())
        .map(|n| n.eq_ignore_ascii_case("powershell") || n.eq_ignore_ascii_case("powershell.exe"))
        .unwrap_or(false);
    let exe_is_shim = std::env::current_exe()
        .ok()
        .and_then(|p| {
            p.file_name()
                .map(|n| n.eq_ignore_ascii_case("powershell.exe"))
        })
        .unwrap_or(false);
    if argv0_is_shim || exe_is_shim {
        return true;
    }
    args.next()
        .map(|a| a == OsStr::new("--ps-shim"))
        .unwrap_or(false)
}

/// Entry point when this binary is invoked through the `powershell.exe` shim.
/// It preserves the original PowerShell command line and injects the proxy
/// prelude into `-File`, `-Command` and `-EncodedCommand` invocations.
fn ps_shim_main() -> Result<()> {
    let _ = ensure_dirs();
    let mut raw: Vec<OsString> = std::env::args_os().skip(1).collect();
    if raw
        .first()
        .map(|a| a == OsStr::new("--ps-shim"))
        .unwrap_or(false)
    {
        raw.remove(0);
    }

    let mut extra_env: Vec<(String, OsString)> = Vec::new();
    let args = if std::env::var("BONSAI_PROXY")
        .ok()
        .filter(|p| !p.trim().is_empty())
        .is_some()
    {
        rewrite_ps_args(&raw, &mut extra_env)?
    } else {
        raw
    };

    let mut cmd = std::process::Command::new(real_powershell_exe());
    cmd.args(&args);
    for (key, value) in extra_env {
        cmd.env(key, value);
    }
    let status = cmd.status().context("start PowerShell")?;
    std::process::exit(status.code().unwrap_or(1));
}

fn flag_lower(s: &OsStr) -> String {
    s.to_string_lossy().to_ascii_lowercase()
}

fn rewrite_ps_args(
    raw: &[OsString],
    extra_env: &mut Vec<(String, OsString)>,
) -> Result<Vec<OsString>> {
    let mut out: Vec<OsString> = Vec::new();
    let mut i = 0usize;
    while i < raw.len() {
        let token = raw[i].to_string_lossy();
        let lower = token.to_ascii_lowercase();

        let is_file = matches!(lower.as_str(), "-file" | "-f" | "/file" | "/f")
            || lower.starts_with("-file:")
            || lower.starts_with("-f:")
            || lower.starts_with("/file:")
            || lower.starts_with("/f:");
        let is_command = matches!(lower.as_str(), "-command" | "-c" | "/command" | "/c");
        let is_encoded = matches!(
            lower.as_str(),
            "-encodedcommand" | "-encoded" | "-ec" | "-e" | "/encodedcommand" | "/encoded" | "/ec" | "/e"
        );

        if is_file {
            let script = if lower.starts_with("-file:") || lower.starts_with("/file:") {
                OsString::from(&token[6..])
            } else if lower.starts_with("-f:") || lower.starts_with("/f:") {
                OsString::from(&token[3..])
            } else {
                i += 1;
                if i >= raw.len() {
                    break;
                }
                raw[i].clone()
            };
            let script_args = raw.get(i + 1..).unwrap_or(&[]).to_vec();

            if !out.iter().any(|a| flag_lower(a) == "-noprofile") {
                out.push(OsString::from("-NoProfile"));
            }
            if !out.iter().any(|a| flag_lower(a) == "-executionpolicy") {
                out.push(OsString::from("-ExecutionPolicy"));
                out.push(OsString::from("Bypass"));
            }

            let args_json = serde_json::to_vec(
                &script_args
                    .iter()
                    .map(|a| a.to_string_lossy().into_owned())
                    .collect::<Vec<String>>(),
            )?;
            extra_env.push(("BONSAI_PS_SCRIPT".into(), script));
            extra_env.push((
                "BONSAI_PS_ARGS_B64".into(),
                OsString::from(STANDARD.encode(args_json)),
            ));
            out.push(OsString::from("-File"));
            out.push(ps_wrapper_path().into_os_string());
            return Ok(out);
        }

        if is_command {
            i += 1;
            if i >= raw.len() {
                break;
            }
            let command_parts: Vec<String> = raw[i..]
                .iter()
                .map(|a| a.to_string_lossy().into_owned())
                .collect();
            let command = command_parts.join(" ");
            out.push(OsString::from("-Command"));
            out.push(OsString::from(format!("{PS_PROXY_PRELUDE}{command}")));
            return Ok(out);
        }

        if is_encoded {
            i += 1;
            if i >= raw.len() {
                break;
            }
            let encoded = raw[i].to_string_lossy();
            if let Ok(bytes) = STANDARD.decode(encoded.as_bytes()) {
                let utf16: Vec<u16> = bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let decoded = String::from_utf16_lossy(&utf16);
                out.push(OsString::from("-Command"));
                out.push(OsString::from(format!("{PS_PROXY_PRELUDE}{decoded}")));
                out.extend(raw.get(i + 1..).unwrap_or(&[]).iter().cloned());
                return Ok(out);
            }
        }

        let takes_arg = matches!(
            lower.as_str(),
            "-executionpolicy" | "-ep" | "-ex" | "-exec" | "/executionpolicy" | "/ep" | "/ex" | "/exec"
                | "-configurationname" | "-cn" | "/configurationname" | "/cn"
                | "-psconsolefile" | "/psconsolefile"
                | "-version" | "-v" | "/version" | "/v"
                | "-inputformat" | "-if" | "-inp" | "/inputformat" | "/if" | "/inp"
                | "-outputformat" | "-of" | "-out" | "/outputformat" | "/of" | "/out"
                | "-windowstyle" | "-w" | "-win" | "/windowstyle" | "/w" | "/win"
        );
        if takes_arg {
            out.push(raw[i].clone());
            i += 1;
            if i < raw.len() {
                out.push(raw[i].clone());
                i += 1;
            }
            continue;
        }

        if token.starts_with('-') || token.starts_with('/') {
            out.push(raw[i].clone());
            i += 1;
            continue;
        }

        // Positional argument: either a script or command
        let clean = lower.trim_matches(|c| c == '\'' || c == '"');
        if clean.ends_with(".ps1") {
            let script = raw[i].clone();
            let script_args = raw.get(i + 1..).unwrap_or(&[]).to_vec();

            if !out.iter().any(|a| flag_lower(a) == "-noprofile") {
                out.push(OsString::from("-NoProfile"));
            }
            if !out.iter().any(|a| flag_lower(a) == "-executionpolicy") {
                out.push(OsString::from("-ExecutionPolicy"));
                out.push(OsString::from("Bypass"));
            }

            let args_json = serde_json::to_vec(
                &script_args
                    .iter()
                    .map(|a| a.to_string_lossy().into_owned())
                    .collect::<Vec<String>>(),
            )?;
            extra_env.push(("BONSAI_PS_SCRIPT".into(), script));
            extra_env.push((
                "BONSAI_PS_ARGS_B64".into(),
                OsString::from(STANDARD.encode(args_json)),
            ));
            out.push(OsString::from("-File"));
            out.push(ps_wrapper_path().into_os_string());
            return Ok(out);
        } else {
            let command_parts: Vec<String> = raw[i..]
                .iter()
                .map(|a| a.to_string_lossy().into_owned())
                .collect();
            let command = command_parts.join(" ");
            out.push(OsString::from("-Command"));
            out.push(OsString::from(format!("{PS_PROXY_PRELUDE}{command}")));
            return Ok(out);
        }
    }
    Ok(out)
}

const CLIPBOARD_PS1: &str = r#"param([Parameter(Mandatory=$true)][string]$Path)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$img = $null
for ($i = 0; $i -lt 5; $i++) {
    try {
        $img = [System.Windows.Forms.Clipboard]::GetImage()
        break
    } catch {
        Start-Sleep -Milliseconds 100
    }
}
if ($null -eq $img) { exit 2 }
try {
    $dir = Split-Path -Parent $Path
    if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $img.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    $img.Dispose()
    exit 0
} catch {
    if ($img) { $img.Dispose() }
    exit 3
}
"#;

#[link(name = "user32")]
unsafe extern "system" {
    fn IsClipboardFormatAvailable(format: u32) -> i32;
    fn GetClipboardSequenceNumber() -> u32;
    fn OpenClipboard(hWndNewOwner: *mut std::ffi::c_void) -> i32;
    fn EmptyClipboard() -> i32;
    fn SetClipboardData(uFormat: u32, hMem: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    fn CloseClipboard() -> i32;
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GlobalAlloc(uFlags: u32, dwBytes: usize) -> *mut std::ffi::c_void;
    fn GlobalLock(hMem: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    fn GlobalUnlock(hMem: *mut std::ffi::c_void) -> i32;
    fn GlobalFree(hMem: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
}

const CF_DIB: u32 = 8;
const CF_UNICODETEXT: u32 = 13;
const CF_DIBV5: u32 = 17;

#[cfg(windows)]
fn copy_text_to_clipboard(text: &str) -> Result<()> {
    const GMEM_MOVEABLE: u32 = 0x0002;
    let utf16: Vec<u16> = text.encode_utf16().chain(std::iter::once(0)).collect();
    let byte_len = utf16.len() * std::mem::size_of::<u16>();

    unsafe {
        let hmem = GlobalAlloc(GMEM_MOVEABLE, byte_len);
        if hmem.is_null() {
            bail!("GlobalAlloc failed for clipboard");
        }
        let ptr = GlobalLock(hmem) as *mut u16;
        if ptr.is_null() {
            let _ = GlobalFree(hmem);
            bail!("GlobalLock failed for clipboard");
        }
        std::ptr::copy_nonoverlapping(utf16.as_ptr(), ptr, utf16.len());
        let _ = GlobalUnlock(hmem);

        let mut opened = false;
        for _ in 0..5 {
            if OpenClipboard(std::ptr::null_mut()) != 0 {
                opened = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(30));
        }
        if !opened {
            let _ = GlobalFree(hmem);
            bail!("could not open Windows clipboard (locked by another app)");
        }
        let _ = EmptyClipboard();
        let res = SetClipboardData(CF_UNICODETEXT, hmem);
        let _ = CloseClipboard();
        if res.is_null() {
            let _ = GlobalFree(hmem);
            bail!("SetClipboardData(CF_UNICODETEXT) failed");
        }
    }
    Ok(())
}

#[cfg(not(windows))]
fn copy_text_to_clipboard(_text: &str) -> Result<()> {
    bail!("clipboard copy is currently implemented for Windows only");
}

fn percent_decode_simple(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let h1 = (bytes[i + 1] as char).to_digit(16);
            let h2 = (bytes[i + 2] as char).to_digit(16);
            if let (Some(d1), Some(d2)) = (h1, h2) {
                out.push(((d1 << 4) | d2) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubNode {
    pub id: usize,
    pub proto: String,
    pub tag: String,
    pub raw: String,
}

fn is_subscription_url(url: &str) -> bool {
    let lower = url.trim().to_ascii_lowercase();
    (lower.starts_with("http://") || lower.starts_with("https://"))
        && (lower.contains("/sub/")
            || lower.contains("/subscribe")
            || lower.contains("sub=1")
            || lower.contains("token=")
            || lower.contains("alius-sub.com"))
}

fn decode_subscription_base64(raw: &str) -> Result<String> {
    if raw.contains("://") && raw.contains('\n') {
        return Ok(raw.to_string());
    }
    let clean: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
    let mut padded = clean.clone();
    while padded.len() % 4 != 0 {
        padded.push('=');
    }
    if let Ok(bytes) = STANDARD.decode(&padded) {
        if let Ok(s) = String::from_utf8(bytes) {
            return Ok(s);
        }
    }
    if let Ok(bytes) = URL_SAFE.decode(&padded) {
        if let Ok(s) = String::from_utf8(bytes) {
            return Ok(s);
        }
    }
    if raw.contains("://") {
        return Ok(raw.to_string());
    }
    bail!("failed to decode subscription content as base64 or plain text");
}

fn parse_subscription_feed(content: &str) -> Vec<SubNode> {
    let decoded = match decode_subscription_base64(content) {
        Ok(s) => s,
        Err(_) => content.to_string(),
    };
    let mut nodes = Vec::new();
    let mut id = 1;
    for line in decoded.lines() {
        let line = line.trim().trim_start_matches('\u{feff}');
        if line.is_empty() {
            continue;
        }
        if let Some((proto, tag)) = parse_node_summary(line) {
            nodes.push(SubNode {
                id,
                proto,
                tag,
                raw: line.to_string(),
            });
            id += 1;
        }
    }
    nodes
}

fn parse_node_summary(line: &str) -> Option<(String, String)> {
    let (proto, rest) = line.split_once("://")?;
    let proto_clean = proto.trim().trim_start_matches('\u{feff}').to_ascii_lowercase();
    let tag = match rest.split_once('#') {
        Some((_, t)) => percent_decode_simple(t.trim()),
        None => format!("{proto_clean}-node"),
    };
    Some((proto_clean, tag))
}

fn is_supported_node(n: &SubNode) -> bool {
    n.proto == "vless" || n.proto == "trojan"
}

fn find_default_node(nodes: &[SubNode]) -> Option<usize> {
    if nodes.is_empty() {
        return None;
    }
    let cities = ["франкфурт", "frankfurt", "вена", "vienna", "амстердам", "amsterdam", "варшава", "warsaw"];
    for city in &cities {
        if let Some((idx, _)) = nodes.iter().enumerate().find(|(_, n)| {
            is_supported_node(n) && n.tag.to_lowercase().contains(city)
        }) {
            return Some(idx);
        }
    }
    nodes.iter().enumerate().find(|(_, n)| is_supported_node(n)).map(|(i, _)| i)
}

fn find_node_by_query<'a>(nodes: &'a [SubNode], query: &str) -> Option<(usize, &'a SubNode)> {
    let q = query.trim();
    if let Ok(num) = q.parse::<usize>() {
        if num >= 1 && num <= nodes.len() {
            return Some((num - 1, &nodes[num - 1]));
        }
    }
    let q_lower = q.to_lowercase();
    nodes.iter().enumerate().find(|(_, n)| {
        n.tag.to_lowercase().contains(&q_lower)
    }).map(|(i, n)| (i, n))
}

fn find_xray_exe() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from(r"C:\Bonsai\tools\xray\xray.exe"),
        bonsai_dir().join("xray").join("xray.exe"),
    ];
    for c in &candidates {
        if c.exists() {
            return Some(c.clone());
        }
    }
    if let Ok(cur) = std::env::current_exe() {
        if let Some(p) = cur.parent() {
            let in_tools = p.join("tools").join("xray").join("xray.exe");
            if in_tools.exists() {
                return Some(in_tools);
            }
        }
    }
    None
}

fn kill_lingering_xray() {
    let pid_path = bonsai_dir().join("xray.pid");
    if let Ok(pid_str) = std::fs::read_to_string(&pid_path) {
        if let Ok(pid) = pid_str.trim().parse::<u32>() {
            #[cfg(windows)]
            {
                let _ = std::process::Command::new("taskkill")
                    .args(["/FI", "IMAGENAME eq xray.exe", "/PID", &pid.to_string(), "/F"])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .output();
            }
        }
        let _ = std::fs::remove_file(&pid_path);
    }
    let cfg_path = bonsai_dir().join("xray_active.json");
    let _ = std::fs::remove_file(&cfg_path);
}

fn build_xray_outbound(node: &SubNode) -> Result<Value> {
    let raw = &node.raw;
    if raw.starts_with("vless://") {
        let uri = &raw[8..];
        let main_part = match uri.split_once('#') {
            Some((m, _)) => m,
            None => uri,
        };
        let (uuid, host_query) = main_part.split_once('@')
            .context("missing @ in vless uri")?;
        let (host_port, query) = match host_query.split_once('?') {
            Some((hp, q)) => (hp, q),
            None => (host_query, ""),
        };
        let (host, port_str) = host_port.rsplit_once(':')
            .context("missing port in vless uri")?;
        let host = host.trim_matches('[').trim_matches(']');
        let port: u16 = port_str.trim_end_matches('/').parse().context("invalid port")?;

        let mut flow = String::new();
        let mut security = String::new();
        let mut sni = host.to_string();
        let mut fp = "chrome".to_string();
        let mut pbk = String::new();
        let mut sid = String::new();
        let mut net_type = "tcp".to_string();
        let mut ws_path = String::new();
        let mut host_hdr = String::new();
        let mut service_name = String::new();
        let mut xhttp_mode = String::new();
        let mut alpn = String::new();
        let mut allow_insecure = false;

        for param in query.split('&') {
            if let Some((k, v)) = param.split_once('=') {
                let v = percent_decode_simple(v);
                match k {
                    "flow" => flow = v,
                    "security" => security = v,
                    "sni" => sni = v,
                    "fp" => fp = v,
                    "pbk" => pbk = v,
                    "sid" => sid = v,
                    "type" => net_type = v.to_ascii_lowercase(),
                    "path" => ws_path = v,
                    "host" => host_hdr = v,
                    "serviceName" | "servicename" => service_name = v,
                    "mode" => xhttp_mode = v,
                    "alpn" => alpn = v,
                    "allowInsecure" | "insecure" => {
                        allow_insecure = matches!(v.as_str(), "1" | "true" | "yes");
                    }
                    _ => {}
                }
            }
        }

        let mut stream_settings = json!({
            "network": net_type,
        });

        match net_type.as_str() {
            "ws" => {
                let mut ws = json!({
                    "path": if ws_path.is_empty() { "/" } else { &ws_path }
                });
                let hdr_host = if !host_hdr.is_empty() { &host_hdr } else { &sni };
                if !hdr_host.is_empty() {
                    ws["headers"] = json!({ "Host": hdr_host });
                }
                stream_settings["wsSettings"] = ws;
            }
            "grpc" => {
                stream_settings["grpcSettings"] = json!({
                    "serviceName": service_name
                });
            }
            "xhttp" | "splithttp" => {
                let mut xh = json!({
                    "path": if ws_path.is_empty() { "/" } else { &ws_path }
                });
                if !host_hdr.is_empty() {
                    xh["host"] = json!(host_hdr);
                }
                if !xhttp_mode.is_empty() {
                    xh["mode"] = json!(xhttp_mode);
                }
                stream_settings["xhttpSettings"] = xh;
            }
            "httpupgrade" => {
                let mut hu = json!({
                    "path": if ws_path.is_empty() { "/" } else { &ws_path }
                });
                if !host_hdr.is_empty() {
                    hu["host"] = json!(host_hdr);
                }
                stream_settings["httpupgradeSettings"] = hu;
            }
            _ => {}
        }

        if security == "reality" {
            stream_settings["security"] = json!("reality");
            stream_settings["realitySettings"] = json!({
                "fingerprint": if fp.is_empty() { "chrome" } else { &fp },
                "serverName": sni,
                "publicKey": pbk,
                "shortId": sid,
            });
        } else if security == "tls" {
            stream_settings["security"] = json!("tls");
            let mut tls_obj = json!({
                "serverName": sni,
                "allowInsecure": allow_insecure,
            });
            if !fp.is_empty() {
                tls_obj["fingerprint"] = json!(fp);
            }
            if !alpn.is_empty() {
                let list: Vec<&str> = alpn.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
                if !list.is_empty() {
                    tls_obj["alpn"] = json!(list);
                }
            }
            stream_settings["tlsSettings"] = tls_obj;
        }

        let mut user_obj = json!({
            "id": uuid,
            "encryption": "none",
        });
        if !flow.is_empty() {
            user_obj["flow"] = json!(flow);
        } else if security == "reality" && net_type == "tcp" {
            user_obj["flow"] = json!("xtls-rprx-vision");
        }

        Ok(json!({
            "protocol": "vless",
            "settings": {
                "vnext": [{
                    "address": host,
                    "port": port,
                    "users": [user_obj]
                }]
            },
            "streamSettings": stream_settings,
        }))
    } else if raw.starts_with("trojan://") {
        let uri = &raw[9..];
        let main_part = match uri.split_once('#') {
            Some((m, _)) => m,
            None => uri,
        };
        let (password_raw, host_query) = main_part.split_once('@')
            .context("missing @ in trojan uri")?;
        let password = percent_decode_simple(password_raw);
        let (host_port, query) = match host_query.split_once('?') {
            Some((hp, q)) => (hp, q),
            None => (host_query, ""),
        };
        let (host, port_str) = host_port.rsplit_once(':')
            .context("missing port in trojan uri")?;
        let host = host.trim_matches('[').trim_matches(']');
        let port: u16 = port_str.trim_end_matches('/').parse().context("invalid port")?;
        let mut sni = host.to_string();
        let mut net_type = "tcp".to_string();
        let mut ws_path = String::new();
        let mut host_hdr = String::new();
        let mut service_name = String::new();
        let mut alpn = String::new();
        let mut allow_insecure = false;

        for param in query.split('&') {
            if let Some((k, v)) = param.split_once('=') {
                let v = percent_decode_simple(v);
                match k {
                    "sni" | "peer" => sni = v,
                    "type" => net_type = v.to_ascii_lowercase(),
                    "path" => ws_path = v,
                    "host" => host_hdr = v,
                    "serviceName" | "servicename" => service_name = v,
                    "alpn" => alpn = v,
                    "allowInsecure" | "insecure" => {
                        allow_insecure = matches!(v.as_str(), "1" | "true" | "yes");
                    }
                    _ => {}
                }
            }
        }

        let mut tls_obj = json!({
            "serverName": sni,
            "allowInsecure": allow_insecure,
        });
        if !alpn.is_empty() {
            let list: Vec<&str> = alpn.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
            if !list.is_empty() {
                tls_obj["alpn"] = json!(list);
            }
        }

        let mut stream_settings = json!({
            "network": net_type,
            "security": "tls",
            "tlsSettings": tls_obj,
        });
        if net_type == "ws" {
            let mut ws = json!({
                "path": if ws_path.is_empty() { "/" } else { &ws_path }
            });
            let hdr_host = if !host_hdr.is_empty() { &host_hdr } else { &sni };
            if !hdr_host.is_empty() {
                ws["headers"] = json!({ "Host": hdr_host });
            }
            stream_settings["wsSettings"] = ws;
        } else if net_type == "grpc" {
            stream_settings["grpcSettings"] = json!({
                "serviceName": service_name
            });
        }

        Ok(json!({
            "protocol": "trojan",
            "settings": {
                "servers": [{
                    "address": host,
                    "port": port,
                    "password": password,
                }]
            },
            "streamSettings": stream_settings
        }))
    } else {
        bail!("Protocol '{}' is not supported by the Xray client bridge yet", node.proto);
    }
}

fn normalize_proxy(input: &str) -> Result<String> {
    let raw = input.trim();
    if raw.is_empty() {
        bail!("proxy URL is empty");
    }
    let with_scheme = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    };
    let u = reqwest::Url::parse(&with_scheme)
        .with_context(|| format!("invalid proxy URL '{raw}'"))?;
    match u.scheme() {
        "http" | "https" | "socks5" | "socks5h" => {}
        other => bail!("unsupported proxy scheme '{other}' (use http, https, socks5 or socks5h)"),
    }
    if u.host_str().is_none() {
        bail!("proxy URL has no host");
    }
    Ok(with_scheme)
}

fn mask_proxy(url: &str) -> String {
    let trimmed = url.trim();
    if is_subscription_url(trimmed) {
        if let Ok(mut u) = reqwest::Url::parse(trimmed) {
            let path = u.path().to_string();
            if let Some(pos) = path.find("/sub/") {
                let prefix = &path[..pos + 5];
                u.set_path(&format!("{prefix}***"));
            } else {
                u.set_path("/***");
            }
            u.set_query(None);
            return u.to_string();
        }
    }
    match reqwest::Url::parse(trimmed) {
        Ok(mut u) => {
            if u.password().is_some() {
                let _ = u.set_password(Some("***"));
            }
            u.to_string()
        }
        Err(_) => trimmed.to_string(),
    }
}

/// Environment inherited by every command the agent spawns.
fn apply_proxy_env(cmd: &mut tokio::process::Command, proxy: &str) {
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "BONSAI_PROXY",
    ] {
        cmd.env(key, proxy);
    }
    cmd.env("NO_PROXY", "localhost,127.0.0.1,::1");
    cmd.env("no_proxy", "localhost,127.0.0.1,::1");

    // Nested `powershell -File` does not inherit the parent process's .NET
    // proxy. Put our shim first in PATH so every child PowerShell is wrapped.
    let shim = proxy_shim_dir();
    if shim.is_dir() {
        let mut path = OsString::from(shim.as_os_str());
        path.push(";");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        cmd.env("PATH", path);
    }
}

const PS_PROXY_PRELUDE: &str = r#"$bp=$env:BONSAI_PROXY; if($bp){ try { $u=[Uri]$bp; $wp=New-Object System.Net.WebProxy($u); if($u.UserInfo){ $c=$u.UserInfo.Split(':',2); $pu=[Uri]::UnescapeDataString($c[0]); $pp=[Uri]::UnescapeDataString($c[1]); $wp.Credentials=New-Object System.Net.NetworkCredential($pu,$pp); $sec=ConvertTo-SecureString $pp -AsPlainText -Force; $pc=New-Object System.Management.Automation.PSCredential($pu,$sec); $PSDefaultParameterValues['Invoke-WebRequest:ProxyCredential']=$pc; $PSDefaultParameterValues['Invoke-RestMethod:ProxyCredential']=$pc }; [System.Net.WebRequest]::DefaultWebProxy=$wp; $PSDefaultParameterValues['Invoke-WebRequest:Proxy']=$bp; $PSDefaultParameterValues['Invoke-RestMethod:Proxy']=$bp } catch {} }; "#;

const PS_FILE_WRAPPER_BODY: &str = r#"
$__s=$env:BONSAI_PS_SCRIPT
if($__s){
  $__s=$__s.Trim('"',"'")
  if(-not [System.IO.Path]::IsPathRooted($__s)){
    $__s=Join-Path (Get-Location) $__s
  }
}
$__a=@()
if($env:BONSAI_PS_ARGS_B64){
  try {
    $__json=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:BONSAI_PS_ARGS_B64))
    $__v=$__json | ConvertFrom-Json
    if($null -ne $__v){ $__a=@($__v) }
  } catch {}
}
Remove-Item Env:BONSAI_PS_SCRIPT -ErrorAction SilentlyContinue
Remove-Item Env:BONSAI_PS_ARGS_B64 -ErrorAction SilentlyContinue
$OutputEncoding=[Console]::OutputEncoding=[Text.Encoding]::UTF8
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force -ErrorAction SilentlyContinue
$__err=$false
try {
  & $__s @__a
  if(-not $?){ $__err=$true }
} catch {
  [Console]::Error.WriteLine("PowerShell script error: " + $_.Exception.Message)
  [Console]::Error.WriteLine($_.ScriptStackTrace)
  $__err=$true
}
if($__err){
  if($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0){ exit $LASTEXITCODE }
  else { exit 1 }
}
elseif($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0){ exit $LASTEXITCODE }
else { exit 0 }
"#;

fn clipboard_seq() -> u32 {
    unsafe { GetClipboardSequenceNumber() }
}

fn clipboard_has_image() -> bool {
    unsafe { IsClipboardFormatAvailable(CF_DIB) != 0 || IsClipboardFormatAvailable(CF_DIBV5) != 0 }
}

/// Save the clipboard image as PNG through a tiny PowerShell STA helper.
fn save_clipboard_image(out: &Path) -> bool {
    let script = bonsai_dir().join("clipboard.ps1");
    let status = std::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-File"])
        .arg(&script)
        .arg("-Path")
        .arg(out)
        .creation_flags(CREATE_NO_WINDOW)
        .status();
    matches!(status, Ok(s) if s.success()) && out.is_file()
}

fn write_pid(port: u16, pid: u32) -> Result<()> {
    std::fs::write(pid_path_for(port), pid.to_string()).context("write server pid file")?;
    Ok(())
}

fn read_pid(port: u16) -> Option<u32> {
    std::fs::read_to_string(pid_path_for(port))
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

fn clear_pid_if(port: u16, pid: u32) {
    if read_pid(port) == Some(pid) {
        let _ = std::fs::remove_file(pid_path_for(port));
    }
}

fn rotate_server_log_if_large() {
    if let Ok(meta) = std::fs::metadata(LOG_PATH) {
        if meta.len() > 5 * 1024 * 1024 {
            let old = format!("{LOG_PATH}.old");
            let _ = std::fs::remove_file(&old);
            let _ = std::fs::rename(LOG_PATH, old);
        }
    }
}

// ---------------------------------------------------------------------------
// Sampling presets (Prism ML)
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
struct Sampling {
    temp: f32,
    top_p: f32,
    top_k: i32,
    min_p: f32,
    presence: f32,
}

fn preset(effort: Effort) -> Sampling {
    match effort {
        Effort::None => Sampling {
            temp: 0.7,
            top_p: 0.80,
            top_k: 20,
            min_p: 0.0,
            presence: 1.5,
        },
        Effort::Low | Effort::Medium | Effort::High | Effort::XHigh => Sampling {
            temp: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.05,
            presence: 0.0,
        },
    }
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

struct ServerHandle {
    child: Option<tokio::process::Child>,
    pid: u32,
    port: u16,
    owned: bool,
    keep_alive: bool,
    mmproj: bool,
    model: String,
    ctx: u32,
    ngl: u32,
    kv_type: Option<String>,
    spec_type: Option<String>,
}

impl ServerHandle {
    fn drop_guard(&mut self) {
        if self.owned && !self.keep_alive {
            if let Some(c) = self.child.as_mut() {
                let _ = c.start_kill();
            }
            clear_pid_if(self.port, self.pid);
        }
    }

    async fn shutdown(&mut self) {
        if self.owned && !self.keep_alive {
            if let Some(mut c) = self.child.take() {
                let _ = c.kill().await;
                let _ = c.wait().await;
            }
            clear_pid_if(self.port, self.pid);
        } else {
            // Keep the process running; just drop our handle.
            self.child.take();
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.drop_guard();
    }
}

async fn health(client: &reqwest::Client, base: &str) -> bool {
    match client
        .get(format!("{base}/health"))
        .timeout(Duration::from_secs(2))
        .send()
        .await
    {
        Ok(r) => r.status().is_success(),
        Err(_) => false,
    }
}

async fn fetch_server_props(client: &reqwest::Client, base: &str) -> Option<Value> {
    let r = client
        .get(format!("{base}/props"))
        .timeout(Duration::from_secs(3))
        .send()
        .await
        .ok()?;
    if !r.status().is_success() {
        return None;
    }
    r.json::<Value>().await.ok()
}

async fn fetch_server_slots(client: &reqwest::Client, base: &str) -> Option<Value> {
    let r = client
        .get(format!("{base}/slots"))
        .timeout(Duration::from_secs(3))
        .send()
        .await
        .ok()?;
    if !r.status().is_success() {
        return None;
    }
    r.json::<Value>().await.ok()
}

fn tail(path: &str, max_bytes: u64) -> String {
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return String::new(),
    };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let start = len.saturating_sub(max_bytes);
    let _ = f.seek(SeekFrom::Start(start));
    let mut buf = Vec::new();
    let _ = f.read_to_end(&mut buf);
    String::from_utf8_lossy(&buf).into_owned()
}

async fn stop_by_pidfile(port: u16) -> Result<()> {
    let ppath = pid_path_for(port);
    let pid = match read_pid(port) {
        Some(p) => p,
        None => {
            println!("no server pid file at {}", ppath.display());
            return Ok(());
        }
    };
    let out = std::process::Command::new("taskkill")
        .args([
            "/FI",
            "IMAGENAME eq llama-server.exe",
            "/PID",
            &pid.to_string(),
            "/T",
            "/F",
        ])
        .output()
        .context("run taskkill")?;
    let _ = std::fs::remove_file(&ppath);
    if out.status.success() {
        println!("stopped llama-server (pid {pid}, port {port})");
    } else {
        println!(
            "pid {pid} not running ({}), pid file removed",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

struct App {
    cli: Cli,
    sampling: Sampling,
    effort: Effort,
    client: reqwest::Client,
    base: String,
    server: Option<ServerHandle>,
    messages: Vec<Value>,
    system: Option<String>,
    image: Option<PathBuf>,
    show_reasoning: bool,
    stats: bool,
    agent: bool,
    bypass: bool,
    proxy: Option<String>,
    interactive: bool,
    cwd: PathBuf,
    last_clip_seq: u32,
    tool_seq: u64,
    sub_url: Option<String>,
    sub_nodes: Vec<SubNode>,
    active_sub_idx: Option<usize>,
    xray_child: Option<std::process::Child>,
    session_tokens: usize,
    budget: Option<usize>,
    think_budget: Option<usize>,
    tokens_per_sec: Option<f64>,
    load_limit: Option<u8>,
    kv_type: Option<String>,
    spec_type: Option<String>,
    server_config_dirty: bool,
    markdown: bool,
}

impl App {
    fn new(cli: Cli, interactive: bool) -> Result<Self> {
        ensure_dirs()?;
        kill_lingering_xray();
        let mut sampling = preset(cli.effort);
        if let Some(v) = cli.temp {
            sampling.temp = v;
        }
        if let Some(v) = cli.top_p {
            sampling.top_p = v;
        }
        if let Some(v) = cli.top_k {
            sampling.top_k = v;
        }
        if let Some(v) = cli.min_p {
            sampling.min_p = v;
        }
        if let Some(v) = cli.presence {
            sampling.presence = v;
        }
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .no_proxy()
            .build()?;
        let base = format!("http://127.0.0.1:{}", cli.port);
        let agent = !cli.no_agent && (interactive || cli.agent);
        let proxy = match cli.proxy.as_deref() {
            Some(p) if is_subscription_url(p) => None,
            Some(p) => Some(normalize_proxy(p)?),
            None => None,
        };
        if proxy.is_some() {
            ensure_ps_shim(true)?;
        }
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from(r"C:\Bonsai"));
        let last_clip_seq = clipboard_seq();
        let _ = std::fs::remove_dir_all(bonsai_dir().join("clipboard"));
        Ok(App {
            system: cli.system.clone(),
            image: cli.image.clone(),
            show_reasoning: !cli.hide_reasoning,
            stats: !cli.no_stats,
            effort: cli.effort,
            sampling,
            client,
            base,
            server: None,
            messages: Vec::new(),
            agent,
            bypass: cli.bypass,
            proxy,
            interactive,
            cwd,
            last_clip_seq,
            tool_seq: 0,
            sub_url: None,
            sub_nodes: Vec::new(),
            active_sub_idx: None,
            xray_child: None,
            session_tokens: 0,
            budget: cli.budget,
            think_budget: cli.think_budget,
            tokens_per_sec: cli.tokens,
            load_limit: cli.load,
            kv_type: cli.kv.clone(),
            spec_type: cli.spec.clone().filter(|s| s != "none"),
            server_config_dirty: false,
            markdown: !cli.no_markdown,
            cli,
        })
    }

    async fn restart_tracked_or_pid_server(&mut self) {
        if let Some(mut h) = self.server.take() {
            h.keep_alive = false; // Force restart when user explicitly changed config
            h.shutdown().await;
        } else if read_pid(self.cli.port).is_some() && !self.cli.no_server {
            let _ = stop_by_pidfile(self.cli.port).await;
        }
    }

    async fn ensure_server(&mut self, want_mmproj: bool) -> Result<()> {
        if health(&self.client, &self.base).await {
            let mut needs_restart = self.server_config_dirty;
            if let Some(h) = &self.server {
                if h.model != self.cli.model
                    || (want_mmproj && !h.mmproj)
                    || h.ctx != self.cli.ctx
                    || h.ngl != self.cli.ngl
                    || h.kv_type != self.kv_type
                    || h.spec_type != self.spec_type
                {
                    needs_restart = true;
                }
            } else if let Some(props) = fetch_server_props(&self.client, &self.base).await {
                let remote_ctx = props
                    .pointer("/default_generation_settings/params/n_ctx")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32);
                let remote_vision = props
                    .pointer("/modalities/vision")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let remote_model = props
                    .get("model_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let model_differs = !remote_model.is_empty()
                    && Path::new(remote_model)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| {
                            !Path::new(&self.cli.model)
                                .file_name()
                                .and_then(|m| m.to_str())
                                .map(|m| m.eq_ignore_ascii_case(n))
                                .unwrap_or(false)
                        })
                        .unwrap_or(false);
                let ctx_differs = remote_ctx.map(|c| c != self.cli.ctx).unwrap_or(false);
                let vision_missing = want_mmproj && !remote_vision;

                if model_differs || ctx_differs || vision_missing {
                    if read_pid(self.cli.port).is_some() && !self.cli.no_server {
                        needs_restart = true;
                    } else {
                        if ctx_differs {
                            eprintln!(
                                "  \x1b[33m!\x1b[0m \x1b[90mExternal engine has n_ctx={} (requested {})\x1b[0m",
                                remote_ctx.unwrap_or(0),
                                self.cli.ctx
                            );
                        }
                        if vision_missing {
                            eprintln!("  \x1b[33m!\x1b[0m \x1b[90mExternal engine does not have vision (--mmproj) loaded\x1b[0m");
                        }
                    }
                }
            }

            if !needs_restart {
                if self.server.is_none() {
                    eprintln!("  \x1b[32m●\x1b[0m \x1b[90mConnected to engine at {}\x1b[0m", self.base);
                }
                return Ok(());
            }
            if !self.cli.no_server {
                eprintln!("server: restarting for updated model/context/vision settings...");
                self.restart_tracked_or_pid_server().await;
            } else {
                eprintln!("  \x1b[33m!\x1b[0m \x1b[90mEngine settings changed but --no-server is active; using external server.\x1b[0m");
                self.server_config_dirty = false;
                return Ok(());
            }
        } else if let Some(mut h) = self.server.take() {
            // Our child died (or an external server went away) — clean up.
            h.shutdown().await;
        }

        if self.cli.no_server {
            bail!(
                "server is not responding at {} and --no-server was given",
                self.base
            );
        }

        self.server_config_dirty = false;
        self.spawn_server(want_mmproj).await
    }

    async fn spawn_server(&mut self, want_mmproj: bool) -> Result<()> {
        let bin = PathBuf::from(&self.cli.server_bin);
        if !bin.exists() {
            bail!("llama-server not found at {}", bin.display());
        }
        let model = PathBuf::from(&self.cli.model);
        if !model.exists() {
            bail!("model not found at {}", model.display());
        }

        rotate_server_log_if_large();
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(LOG_PATH)
            .with_context(|| format!("open log {LOG_PATH}"))?;
        let log_err = log.try_clone()?;

        let mut cmd = tokio::process::Command::new(&bin);
        cmd.current_dir(bin.parent().unwrap_or(Path::new("C:\\Bonsai\\llama")))
            .arg("-m")
            .arg(&self.cli.model)
            .arg("-ngl")
            .arg(self.cli.ngl.to_string())
            .arg("-c")
            .arg(self.cli.ctx.to_string())
            .arg("-fa")
            .arg("on")
            .arg("--jinja")
            .arg("--reasoning")
            .arg("on")
            .arg("--reasoning-format")
            .arg("deepseek");
        // Server-side default stays "medium"; per-request chat_template_kwargs
        // (enable_thinking / reasoning_effort) select the real behaviour.
        cmd.arg("--reasoning-effort")
            .arg("medium")
            .arg("--reasoning-effort-allow")
            .arg("low,medium,high,xhigh")
            .arg("--reasoning-effort-fallback")
            .arg("medium")
            .arg("--temp")
            .arg(format!("{}", self.sampling.temp))
            .arg("--top-p")
            .arg(format!("{}", self.sampling.top_p))
            .arg("--top-k")
            .arg(self.sampling.top_k.to_string())
            .arg("--min-p")
            .arg(format!("{}", self.sampling.min_p))
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(self.cli.port.to_string())
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::from(log))
            .stderr(std::process::Stdio::from(log_err))
            .creation_flags(CREATE_NO_WINDOW);

        if let Some(t) = self.cli.threads {
            cmd.arg("-t").arg(t.to_string());
        }
        if let Some(kv) = &self.kv_type {
            cmd.arg("-ctk").arg(kv).arg("-ctv").arg(kv);
        }
        if let Some(spec) = &self.spec_type {
            if spec != "none" {
                cmd.arg("--spec-type").arg(spec);
            }
        }
        if want_mmproj {
            if Path::new(DEFAULT_MMPROJ).exists() {
                cmd.arg("--mmproj").arg(DEFAULT_MMPROJ);
            } else {
                eprintln!("warning: --image given but {DEFAULT_MMPROJ} is missing");
            }
        }

        eprintln!("  \x1b[90m● Initializing local model engine (27B-Q1)...\x1b[0m");
        let child = cmd.spawn().context("spawn llama-server")?;
        let pid = child.id().unwrap_or(0);
        write_pid(self.cli.port, pid)?;

        let mut handle = ServerHandle {
            child: Some(child),
            pid,
            port: self.cli.port,
            owned: true,
            keep_alive: self.cli.keep_alive,
            mmproj: want_mmproj,
            model: self.cli.model.clone(),
            ctx: self.cli.ctx,
            ngl: self.cli.ngl,
            kv_type: self.kv_type.clone(),
            spec_type: self.spec_type.clone(),
        };

        let started = Instant::now();
        let mut announced = Instant::now();
        let mut dots = false;
        loop {
            if CANCEL.load(Ordering::SeqCst) {
                if let Some(mut c) = handle.child.take() {
                    let _ = c.kill().await;
                    let _ = c.wait().await;
                }
                clear_pid_if(self.cli.port, pid);
                bail!("server startup cancelled by user");
            }
            if health(&self.client, &self.base).await {
                if dots {
                    eprintln!();
                }
                eprintln!("  \x1b[32m●\x1b[0m \x1b[90mModel engine ready ({:.1}s)\x1b[0m\n", started.elapsed().as_secs_f32());
                self.server = Some(handle);
                return Ok(());
            }
            if let Some(c) = handle.child.as_mut() {
                if let Some(status) = c.try_wait()? {
                    clear_pid_if(self.cli.port, pid);
                    let t = tail(LOG_PATH, 8192);
                    bail!(
                        "llama-server exited with {status}\n--- {LOG_PATH} (tail) ---\n{}",
                        t.trim_end()
                    );
                }
            }
            if started.elapsed() > Duration::from_secs(600) {
                if let Some(mut c) = handle.child.take() {
                    let _ = c.kill().await;
                }
                clear_pid_if(self.cli.port, pid);
                let t = tail(LOG_PATH, 8192);
                bail!(
                    "llama-server did not become healthy in 600s\n--- {LOG_PATH} (tail) ---\n{}",
                    t.trim_end()
                );
            }
            if announced.elapsed() >= Duration::from_secs(5) {
                eprint!(".");
                let _ = std::io::stderr().flush();
                dots = true;
                announced = Instant::now();
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
}

// ---------------------------------------------------------------------------
// Chat request / SSE streaming
// ---------------------------------------------------------------------------

#[derive(Default, Debug)]
struct Stats {
    prompt_n: u64,
    prompt_per_second: Option<f64>,
    predicted_n: u64,
    predicted_per_second: Option<f64>,
    elapsed: f64,
}

fn mime_for(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        _ => "image/png",
    }
}

fn image_data_url(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read image {}", path.display()))?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(format!("data:{};base64,{}", mime_for(path), b64))
}

fn user_message(text: &str, image: Option<&Path>) -> Result<Value> {
    match image {
        None => Ok(json!({"role": "user", "content": text})),
        Some(p) => {
            let url = image_data_url(p)?;
            Ok(json!({
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": url}}
                ]
            }))
        }
    }
}

fn chat_body(app: &App, messages: &[Value]) -> Value {
    let mut body = json!({
        "model": "bonsai",
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
        "temperature": app.sampling.temp,
        "top_p": app.sampling.top_p,
        "top_k": app.sampling.top_k,
        "min_p": app.sampling.min_p,
        "presence_penalty": app.sampling.presence,
        "reasoning_effort": if app.effort == Effort::None { "none" } else { app.effort.as_str() },
        "chat_template_kwargs": {
            "enable_thinking": app.effort.thinking(),
            "reasoning_effort": if app.effort == Effort::None { "medium" } else { app.effort.as_str() }
        }
    });
    if let Some(b) = app.budget {
        let rem = b.saturating_sub(app.session_tokens).max(1);
        body["max_tokens"] = json!(rem);
    }
    if app.effort.thinking() {
        if let Some(tb) = app.think_budget {
            body["reasoning_budget_tokens"] = json!(tb as i64);
        }
    }
    if app.agent {
        body["tools"] = tools_spec();
        body["tool_choice"] = json!("auto");
    }
    body
}

#[derive(Clone, Debug, Default)]
struct ToolCall {
    id: String,
    name: String,
    arguments: String,
}

fn merge_tool_calls(calls: &mut Vec<ToolCall>, arr: &[Value]) {
    for (pos, tc) in arr.iter().enumerate() {
        let idx = tc
            .get("index")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(pos);
        while calls.len() <= idx {
            calls.push(ToolCall::default());
        }
        let slot = &mut calls[idx];
        if let Some(id) = tc.get("id").and_then(|x| x.as_str()) {
            if !id.is_empty() {
                slot.id = id.to_string();
            }
        }
        if let Some(f) = tc.get("function") {
            if let Some(name) = f.get("name").and_then(|x| x.as_str()) {
                if !name.is_empty() && slot.name != name {
                    if name.starts_with(&slot.name) {
                        slot.name = name.to_string();
                    } else {
                        slot.name.push_str(name);
                    }
                }
            }
            if let Some(args) = f.get("arguments") {
                match args {
                    Value::String(s) => slot.arguments.push_str(s),
                    Value::Null => {}
                    other => slot.arguments.push_str(&other.to_string()),
                }
            }
        }
    }
}

static CANCEL: AtomicBool = AtomicBool::new(false);

fn redact_secrets(text: &str, secrets: &[String]) -> String {
    let mut out = text.to_string();
    for s in secrets {
        if !s.is_empty() {
            out = out.replace(s, "[REDACTED]");
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Markdown Stream Renderer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum LinkState {
    None,
    Text(String),
    ExpectParen(String),
    Url(String, String),
}

#[derive(Debug, Clone)]
pub struct MarkdownStreamer {
    pub enabled: bool,
    pub color: bool,
    in_bold: bool,
    in_italic: bool,
    in_strike: bool,
    in_inline_code: bool,
    in_code_block: bool,
    code_fence_len: usize,
    code_at_line_start: bool,
    pending_code_indent: bool,
    in_header: bool,
    in_table_row: bool,
    header_color: Option<&'static str>,
    at_line_start: bool,
    star_at_line_start: bool,
    last_was_newline: bool,
    escaped: bool,
    code_block_lang: bool,
    prev_char: Option<char>,
    pending: String,
    link_state: LinkState,
}

impl MarkdownStreamer {
    pub fn new(enabled: bool, color: bool) -> Self {
        Self {
            enabled,
            color,
            in_bold: false,
            in_italic: false,
            in_strike: false,
            in_inline_code: false,
            in_code_block: false,
            code_fence_len: 3,
            code_at_line_start: false,
            pending_code_indent: false,
            in_header: false,
            in_table_row: false,
            header_color: None,
            at_line_start: true,
            star_at_line_start: true,
            last_was_newline: false,
            escaped: false,
            code_block_lang: false,
            prev_char: None,
            pending: String::new(),
            link_state: LinkState::None,
        }
    }

    pub fn write<W: std::io::Write>(&mut self, text: &str, out: &mut W) -> std::io::Result<()> {
        if !self.enabled || !self.color {
            out.write_all(text.as_bytes())?;
            return Ok(());
        }

        for ch in text.chars() {
            if ch == '\r' {
                continue;
            }
            self.process_char(ch, out)?;
        }
        Ok(())
    }

    fn write_char<W: std::io::Write>(out: &mut W, ch: char) -> std::io::Result<()> {
        let mut b = [0u8; 4];
        out.write_all(ch.encode_utf8(&mut b).as_bytes())
    }

    fn restore_styles<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        if !self.color {
            return Ok(());
        }
        write!(out, "\x1b[0m")?;
        if self.in_header {
            if let Some(hc) = self.header_color {
                write!(out, "{hc}")?;
            }
        }
        if self.in_bold {
            write!(out, "\x1b[1;37m")?;
        }
        if self.in_italic {
            write!(out, "\x1b[3m")?;
        }
        if self.in_strike {
            write!(out, "\x1b[9m")?;
        }
        Ok(())
    }

    fn flush_link_abort<W: std::io::Write>(&mut self, out: &mut W) -> std::io::Result<()> {
        let state = std::mem::replace(&mut self.link_state, LinkState::None);
        match state {
            LinkState::None => {}
            LinkState::Text(t) => write!(out, "[{t}")?,
            LinkState::ExpectParen(t) => write!(out, "[{t}]")?,
            LinkState::Url(t, u) => write!(out, "[{t}]({u}")?,
        }
        Ok(())
    }

    fn handle_link_char<W: std::io::Write>(&mut self, ch: char, out: &mut W) -> std::io::Result<bool> {
        let state = std::mem::replace(&mut self.link_state, LinkState::None);
        match state {
            LinkState::None => Ok(false),
            LinkState::Text(mut t) => {
                if ch == ']' {
                    self.link_state = LinkState::ExpectParen(t);
                    Ok(true)
                } else if ch == '\n' || ch == '[' || t.chars().count() > 120 {
                    write!(out, "[{t}")?;
                    Ok(false)
                } else {
                    t.push(ch);
                    self.link_state = LinkState::Text(t);
                    Ok(true)
                }
            }
            LinkState::ExpectParen(t) => {
                if ch == '(' {
                    self.link_state = LinkState::Url(t, String::new());
                    Ok(true)
                } else {
                    write!(out, "[{t}]")?;
                    Ok(false)
                }
            }
            LinkState::Url(t, mut u) => {
                if ch == ')' {
                    write!(out, "\x1b[4;36m{t}\x1b[0m \x1b[90m({u})\x1b[0m")?;
                    if self.in_header || self.in_bold || self.in_italic || self.in_strike {
                        self.restore_styles(out)?;
                    }
                    self.prev_char = Some(')');
                    Ok(true)
                } else if ch == '\n' || ch.is_whitespace() || u.chars().count() > 300 {
                    write!(out, "[{t}]({u}")?;
                    Ok(false)
                } else {
                    u.push(ch);
                    self.link_state = LinkState::Url(t, u);
                    Ok(true)
                }
            }
        }
    }

    fn process_char<W: std::io::Write>(&mut self, ch: char, out: &mut W) -> std::io::Result<()> {
        // Inside fenced code block
        if self.in_code_block {
            if self.code_block_lang {
                if ch == '\n' {
                    self.code_block_lang = false;
                    self.code_at_line_start = true;
                    self.pending_code_indent = true;
                    write!(out, "\x1b[0m\n")?;
                } else {
                    Self::write_char(out, ch)?;
                }
                self.prev_char = Some(ch);
                return Ok(());
            }

            if self.code_at_line_start && (ch == '`' || !self.pending.is_empty()) {
                if ch == '`' {
                    self.pending.push('`');
                    return Ok(());
                }
                if self.pending.len() >= self.code_fence_len {
                    if ch == '\n' {
                        self.in_code_block = false;
                        self.pending.clear();
                        self.pending_code_indent = false;
                        write!(out, "\x1b[0m\x1b[90m└──\x1b[0m\n")?;
                        self.at_line_start = true;
                        self.star_at_line_start = true;
                        self.last_was_newline = true;
                        self.prev_char = Some('\n');
                        return Ok(());
                    } else if ch == ' ' || ch == '\t' {
                        return Ok(()); // Ignore trailing spaces on closing fence line
                    }
                }
                // Not a closing fence: flush indent + pending backticks
                if self.pending_code_indent {
                    write!(out, "  \x1b[38;5;252m")?;
                    self.pending_code_indent = false;
                }
                if !self.pending.is_empty() {
                    out.write_all(self.pending.as_bytes())?;
                    self.pending.clear();
                }
                self.code_at_line_start = false;
            }

            if ch == '\n' {
                out.write_all(b"\n")?;
                self.code_at_line_start = true;
                self.pending_code_indent = true;
            } else {
                if self.pending_code_indent {
                    write!(out, "  \x1b[38;5;252m")?;
                    self.pending_code_indent = false;
                }
                if ch != ' ' && ch != '\t' {
                    self.code_at_line_start = false;
                }
                Self::write_char(out, ch)?;
            }
            self.prev_char = Some(ch);
            return Ok(());
        }

        // Inside inline code: only single ` closes it
        if self.in_inline_code {
            if ch == '`' {
                self.in_inline_code = false;
                self.restore_styles(out)?;
                self.prev_char = Some('`');
                return Ok(());
            }
            if ch == '\n' && self.last_was_newline {
                // Paragraph boundary resets unclosed inline code
                self.in_inline_code = false;
                self.restore_styles(out)?;
            }
            self.last_was_newline = ch == '\n';
            Self::write_char(out, ch)?;
            self.prev_char = Some(ch);
            return Ok(());
        }

        // Handle active link parsing
        if !matches!(self.link_state, LinkState::None) {
            if self.handle_link_char(ch, out)? {
                return Ok(());
            }
        }

        // Handle backslash escape
        if self.escaped {
            self.escaped = false;
            self.last_was_newline = false;
            if !self.pending.is_empty() {
                let _ = self.resolve_pending(out, Some('\\'))?;
            }
            if matches!(ch, '*' | '_' | '`' | '#' | '-' | '>' | '~' | '\\' | '[' | ']' | '|') {
                self.at_line_start = false;
                Self::write_char(out, ch)?;
                self.prev_char = Some(ch);
                return Ok(());
            } else {
                self.at_line_start = false;
                out.write_all(b"\\")?;
                // fall through to process `ch` normally
            }
        }

        if ch == '\\' {
            if !self.pending.is_empty() {
                let _ = self.resolve_pending(out, Some('\\'))?;
                if self.in_inline_code || self.in_code_block {
                    return self.process_char('\\', out);
                }
            }
            self.escaped = true;
            return Ok(());
        }

        // Newline resets header, table row, and on blank line resets unclosed inline styles
        if ch == '\n' {
            let _ = self.resolve_pending(out, Some('\n'))?;
            let is_blank = self.last_was_newline;
            if self.in_header {
                self.in_header = false;
                self.header_color = None;
                self.in_bold = false;
                self.in_italic = false;
                self.in_strike = false;
                write!(out, "\x1b[0m")?;
            } else if is_blank && (self.in_bold || self.in_italic || self.in_strike) {
                self.in_bold = false;
                self.in_italic = false;
                self.in_strike = false;
                write!(out, "\x1b[0m")?;
            }
            self.in_table_row = false;
            out.write_all(b"\n")?;
            self.at_line_start = true;
            self.star_at_line_start = true;
            self.last_was_newline = true;
            self.prev_char = Some('\n');
            return Ok(());
        }

        self.last_was_newline = false;

        // Start link buffering on '['
        if ch == '[' && !self.in_header {
            if !self.pending.is_empty() {
                let _ = self.resolve_pending(out, Some('['))?;
                if self.in_inline_code || self.in_code_block {
                    return self.process_char('[', out);
                }
            }
            self.at_line_start = false;
            self.link_state = LinkState::Text(String::new());
            return Ok(());
        }

        // Table rows: lines starting with '|'
        if ch == '|' {
            if !self.pending.is_empty() {
                let _ = self.resolve_pending(out, Some('|'))?;
                if self.in_inline_code || self.in_code_block {
                    return self.process_char('|', out);
                }
            }
            if self.at_line_start {
                self.in_table_row = true;
                self.at_line_start = false;
            }
            if self.in_table_row {
                write!(out, "\x1b[90m│\x1b[0m")?;
                if self.in_bold || self.in_italic || self.in_strike {
                    self.restore_styles(out)?;
                }
            } else {
                out.write_all(b"|")?;
            }
            self.prev_char = Some('|');
            return Ok(());
        }

        // Resolve incompatible pending delimiters before handling the new char
        if !self.pending.is_empty() {
            let continues = match ch {
                '*' | '_' | '~' | '`' => self.pending.chars().all(|c| c == ch),
                '#' if self.at_line_start => self.pending.chars().all(|c| c == '#'),
                '-' if self.at_line_start => self.pending.chars().all(|c| c == '-'),
                '0'..='9' if self.at_line_start => self.pending.chars().all(|c| c.is_ascii_digit()),
                '.' if self.at_line_start => self.pending.chars().all(|c| c.is_ascii_digit()),
                _ => false,
            };
            if !continues {
                let space_consumed = self.resolve_pending(out, Some(ch))?;
                if space_consumed && ch == ' ' {
                    self.prev_char = Some(' ');
                    return Ok(());
                }
                if self.in_inline_code || self.in_code_block {
                    return self.process_char(ch, out);
                }
            }
        }

        // Accumulate delimiters
        match ch {
            '*' => {
                if self.pending.is_empty() {
                    self.star_at_line_start = self.at_line_start;
                }
                self.pending.push('*');
                return Ok(());
            }
            '_' | '~' | '`' => {
                self.pending.push(ch);
                return Ok(());
            }
            '#' if self.at_line_start => {
                self.pending.push('#');
                return Ok(());
            }
            '-' if self.at_line_start => {
                self.pending.push('-');
                return Ok(());
            }
            '>' if self.at_line_start && self.pending.is_empty() => {
                self.pending.push('>');
                return Ok(());
            }
            '0'..='9' if self.at_line_start => {
                self.pending.push(ch);
                return Ok(());
            }
            '.' if self.at_line_start && !self.pending.is_empty() && self.pending.chars().all(|c| c.is_ascii_digit()) => {
                self.pending.push('.');
                return Ok(());
            }
            _ => {}
        }

        if ch != ' ' && ch != '\t' {
            self.at_line_start = false;
        }

        Self::write_char(out, ch)?;
        self.prev_char = Some(ch);
        Ok(())
    }

    fn resolve_pending<W: std::io::Write>(&mut self, out: &mut W, next_char: Option<char>) -> std::io::Result<bool> {
        if self.pending.is_empty() {
            return Ok(false);
        }

        let p = std::mem::take(&mut self.pending);
        let at_line_end = matches!(next_char, None | Some('\n'));

        // Horizontal rule: ---, ***, ___ at line start
        if self.at_line_start && p.len() >= 3 && (p.chars().all(|c| c == '-') || p.chars().all(|c| c == '*') || p.chars().all(|c| c == '_')) && at_line_end {
            self.at_line_start = false;
            write!(out, "\x1b[90m────────────────────────────────────\x1b[0m")?;
            return Ok(false);
        }

        // Backticks: `, ```, ````...
        if p.chars().all(|c| c == '`') {
            if p.len() >= 3 && self.at_line_start {
                self.in_code_block = true;
                self.code_fence_len = p.len();
                self.code_block_lang = next_char != Some('\n');
                self.code_at_line_start = true;
                self.pending_code_indent = next_char == Some('\n');
                self.at_line_start = false;
                write!(out, "\x1b[90m┌──\x1b[0m \x1b[33m")?;
                return Ok(false);
            } else if p.len() == 1 {
                if self.in_inline_code {
                    self.in_inline_code = false;
                    self.restore_styles(out)?;
                } else {
                    let can_open = next_char.map(|c| !c.is_whitespace() && c != '`').unwrap_or(false);
                    if can_open {
                        self.in_inline_code = true;
                        write!(out, "\x1b[36m")?;
                    } else {
                        out.write_all(b"`")?;
                    }
                }
                self.at_line_start = false;
                return Ok(false);
            } else {
                self.at_line_start = false;
                out.write_all(p.as_bytes())?;
                return Ok(false);
            }
        }

        // Strikethrough: ~~
        if p.chars().all(|c| c == '~') {
            if p.len() == 2 {
                if self.in_strike {
                    self.in_strike = false;
                    self.restore_styles(out)?;
                } else if next_char.map(|c| !c.is_whitespace()).unwrap_or(false) {
                    self.in_strike = true;
                    write!(out, "\x1b[9m")?;
                } else {
                    out.write_all(b"~~")?;
                }
            } else {
                out.write_all(p.as_bytes())?;
            }
            self.at_line_start = false;
            return Ok(false);
        }

        let is_intraword = self.prev_char.map(|c| c.is_alphanumeric()).unwrap_or(false)
            && next_char.map(|c| c.is_alphanumeric()).unwrap_or(false);
        let is_ws = next_char.map(|c| c.is_whitespace()).unwrap_or(true);

        // Asterisks: **, *, ***
        if p.chars().all(|c| c == '*') {
            match p.len() {
                2 => {
                    if self.in_bold {
                        self.in_bold = false;
                        self.restore_styles(out)?;
                    } else if !is_ws {
                        self.in_bold = true;
                        write!(out, "\x1b[1;37m")?;
                    } else {
                        out.write_all(b"**")?;
                    }
                    self.at_line_start = false;
                    return Ok(false);
                }
                1 => {
                    if self.star_at_line_start && next_char == Some(' ') {
                        self.at_line_start = false;
                        self.star_at_line_start = false;
                        write!(out, "  \x1b[32m•\x1b[0m ")?;
                        if self.in_bold || self.in_italic || self.in_strike {
                            self.restore_styles(out)?;
                        }
                        return Ok(true);
                    }
                    self.star_at_line_start = false;
                    if self.in_italic {
                        self.in_italic = false;
                        self.restore_styles(out)?;
                    } else if !is_intraword && !is_ws {
                        self.in_italic = true;
                        write!(out, "\x1b[3m")?;
                    } else {
                        out.write_all(b"*")?;
                    }
                    self.at_line_start = false;
                    return Ok(false);
                }
                3 => {
                    if self.in_bold && self.in_italic {
                        self.in_bold = false;
                        self.in_italic = false;
                        self.restore_styles(out)?;
                    } else if !is_ws {
                        self.in_bold = true;
                        self.in_italic = true;
                        write!(out, "\x1b[1;3;37m")?;
                    } else {
                        out.write_all(b"***")?;
                    }
                    self.at_line_start = false;
                    return Ok(false);
                }
                _ => {
                    self.at_line_start = false;
                    out.write_all(p.as_bytes())?;
                    return Ok(false);
                }
            }
        }

        // Underscores: __, _
        if p.chars().all(|c| c == '_') {
            match p.len() {
                2 => {
                    if self.in_bold {
                        self.in_bold = false;
                        self.restore_styles(out)?;
                    } else if !is_intraword && !is_ws {
                        self.in_bold = true;
                        write!(out, "\x1b[1;37m")?;
                    } else {
                        out.write_all(b"__")?;
                    }
                    self.at_line_start = false;
                    return Ok(false);
                }
                1 => {
                    if self.in_italic {
                        self.in_italic = false;
                        self.restore_styles(out)?;
                    } else if !is_intraword && !is_ws {
                        self.in_italic = true;
                        write!(out, "\x1b[3m")?;
                    } else {
                        out.write_all(b"_")?;
                    }
                    self.at_line_start = false;
                    return Ok(false);
                }
                _ => {
                    self.at_line_start = false;
                    out.write_all(p.as_bytes())?;
                    return Ok(false);
                }
            }
        }

        // Headers: #, ##, ###...
        if p.chars().all(|c| c == '#') {
            if next_char == Some(' ') {
                self.in_header = true;
                self.at_line_start = false;
                let color_code = match p.len() {
                    1 => "\x1b[1;34m",
                    2 => "\x1b[1;32m",
                    _ => "\x1b[1;36m",
                };
                self.header_color = Some(color_code);
                write!(out, "{color_code}{p} ")?;
                return Ok(true);
            } else {
                self.at_line_start = false;
                out.write_all(p.as_bytes())?;
                return Ok(false);
            }
        }

        // Dash list: -
        if p == "-" {
            if next_char == Some(' ') {
                self.at_line_start = false;
                write!(out, "  \x1b[32m•\x1b[0m ")?;
                return Ok(true);
            } else {
                self.at_line_start = false;
                out.write_all(b"-")?;
                return Ok(false);
            }
        }

        // Quote: >
        if p == ">" {
            if next_char == Some(' ') {
                self.at_line_start = false;
                write!(out, "\x1b[90m│\x1b[0m ")?;
                return Ok(true);
            } else {
                self.at_line_start = false;
                out.write_all(b">")?;
                return Ok(false);
            }
        }

        // Numbered list: 1. or 10.
        if p.ends_with('.') && p.trim_end_matches('.').chars().all(|c| c.is_ascii_digit()) {
            if next_char == Some(' ') {
                self.at_line_start = false;
                write!(out, "  \x1b[33m{p}\x1b[0m ")?;
                return Ok(true);
            } else {
                self.at_line_start = false;
                out.write_all(p.as_bytes())?;
                return Ok(false);
            }
        }

        self.at_line_start = false;
        out.write_all(p.as_bytes())?;
        Ok(false)
    }

    pub fn finish<W: std::io::Write>(&mut self, out: &mut W) -> std::io::Result<()> {
        if self.escaped {
            out.write_all(b"\\")?;
            self.escaped = false;
        }
        self.flush_link_abort(out)?;
        if self.in_code_block {
            if self.code_at_line_start && self.pending.len() >= self.code_fence_len {
                self.pending.clear();
                self.pending_code_indent = false;
                write!(out, "\x1b[0m\x1b[90m└──\x1b[0m\n")?;
            } else {
                if !self.pending.is_empty() {
                    if self.pending_code_indent {
                        write!(out, "  \x1b[38;5;252m")?;
                        self.pending_code_indent = false;
                    }
                    out.write_all(self.pending.as_bytes())?;
                    self.pending.clear();
                }
                write!(out, "\x1b[0m\n\x1b[90m└──\x1b[0m\n")?;
            }
            self.in_code_block = false;
        } else if !self.pending.is_empty() {
            let _ = self.resolve_pending(out, None);
        }
        if self.color && (self.in_bold || self.in_italic || self.in_strike || self.in_inline_code || self.in_header) {
            write!(out, "\x1b[0m")?;
        }
        self.in_bold = false;
        self.in_italic = false;
        self.in_strike = false;
        self.in_inline_code = false;
        self.in_code_block = false;
        self.in_header = false;
        self.in_table_row = false;
        self.header_color = None;
        self.at_line_start = true;
        self.star_at_line_start = true;
        self.last_was_newline = false;
        self.prev_char = None;
        out.flush()?;
        Ok(())
    }
}

async fn stream_chat(
    client: &reqwest::Client,
    base: &str,
    body: Value,
    show_reasoning: bool,
    cancel: &'static AtomicBool,
    secrets: &[String],
    tokens_per_sec: Option<f64>,
    load_limit: Option<u8>,
    markdown: bool,
    remaining_budget: Option<usize>,
) -> Result<(Stats, String, String, Vec<ToolCall>)> {
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .context("POST /v1/chat/completions")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("server returned {status}: {text}");
    }

    let color = std::io::stdout().is_terminal();
    let mut md_streamer = MarkdownStreamer::new(markdown, color);

    let mut out = std::io::stdout();
    let mut buf: Vec<u8> = Vec::new();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut reasoning_open = false;
    let mut printed_any = false;
    let mut stats = Stats::default();
    let mut usage_prompt: Option<u64> = None;
    let mut usage_completion: Option<u64> = None;
    let mut streamed_tokens: u64 = 0;
    let started = Instant::now();
    let mut last_token = Instant::now();
    let mut is_first_token = true;

    let mut stream = resp.bytes_stream();
    'outer: while let Some(chunk) = stream.next().await {
        if cancel.load(Ordering::SeqCst) {
            tool_calls.clear();
            break;
        }
        let chunk = chunk.context("read SSE stream")?;
        buf.extend_from_slice(&chunk);

        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\r', '\n']);
            let data = match line.strip_prefix("data:") {
                Some(d) => d.trim_start(),
                None => continue,
            };
            if data == "[DONE]" {
                break 'outer;
            }
            let v: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                usage_prompt = u.get("prompt_tokens").and_then(|x| x.as_u64());
                usage_completion = u.get("completion_tokens").and_then(|x| x.as_u64());
            }
            if let Some(t) = v.get("timings").filter(|t| !t.is_null()) {
                stats.prompt_n = t.get("prompt_n").and_then(|x| x.as_u64()).unwrap_or(0);
                stats.predicted_n = t.get("predicted_n").and_then(|x| x.as_u64()).unwrap_or(0);
                stats.prompt_per_second = t.get("prompt_per_second").and_then(|x| x.as_f64());
                stats.predicted_per_second =
                    t.get("predicted_per_second").and_then(|x| x.as_f64());
            }

            let delta = match v
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
            {
                Some(d) => d,
                None => continue,
            };

            let has_text = delta.get("reasoning_content").and_then(|x| x.as_str()).map(|s| !s.is_empty()).unwrap_or(false)
                || delta.get("content").and_then(|x| x.as_str()).map(|s| !s.is_empty()).unwrap_or(false);
            if has_text {
                streamed_tokens += 1;
                if is_first_token {
                    is_first_token = false;
                    last_token = Instant::now();
                } else {
                    let compute_elapsed = last_token.elapsed();
                    let mut sleep_dur = Duration::ZERO;

                    if let Some(load_pct) = load_limit {
                        if load_pct > 0 && load_pct < 100 {
                            let factor = (100.0 - load_pct as f64) / (load_pct as f64);
                            if factor.is_finite() && factor >= 0.0 {
                                let sleep_secs = compute_elapsed.as_secs_f64() * factor;
                                if sleep_secs.is_finite() && sleep_secs >= 0.0 && sleep_secs <= 60.0 {
                                    let duty_sleep = Duration::from_secs_f64(sleep_secs);
                                    sleep_dur = sleep_dur.max(duty_sleep);
                                }
                            }
                        }
                    }

                    if let Some(tps) = tokens_per_sec {
                        if tps >= 0.1 && tps <= 5000.0 {
                            let target_dur = Duration::from_secs_f64(1.0 / tps);
                            if compute_elapsed < target_dur {
                                sleep_dur = sleep_dur.max(target_dur - compute_elapsed);
                            }
                        }
                    }

                    // Clamp single sleep to 500ms max to prevent UI freezes
                    sleep_dur = sleep_dur.min(Duration::from_millis(500));

                    if !sleep_dur.is_zero() {
                        tokio::time::sleep(sleep_dur).await;
                        if cancel.load(Ordering::SeqCst) {
                            tool_calls.clear();
                            break 'outer;
                        }
                    }
                    last_token = Instant::now();
                }
            }

            if let Some(r) = delta.get("reasoning_content").and_then(|x| x.as_str()) {
                reasoning.push_str(r);
                if show_reasoning {
                    if !reasoning_open {
                        if color {
                            write!(out, "\x1b[90m╭─ Thought ──────────────────────────────────────────\n")?;
                        } else {
                            write!(out, "╭─ Thought ──────────────────────────────────────────\n")?;
                        }
                        reasoning_open = true;
                    }
                    let safe = redact_secrets(r, secrets);
                    out.write_all(safe.as_bytes())?;
                    out.flush()?;
                    printed_any = true;
                }
            }
            if let Some(c) = delta.get("content").and_then(|x| x.as_str()) {
                if reasoning_open {
                    if color {
                        write!(out, "\n╰────────────────────────────────────────────────────\x1b[0m\n\n")?;
                    } else {
                        write!(out, "\n╰────────────────────────────────────────────────────\n\n")?;
                    }
                    reasoning_open = false;
                }
                content.push_str(c);
                let safe = redact_secrets(c, secrets);
                md_streamer.write(&safe, &mut out)?;
                out.flush()?;
                printed_any = true;
            }
            if let Some(arr) = delta.get("tool_calls").and_then(|x| x.as_array()) {
                merge_tool_calls(&mut tool_calls, arr);
            } else if let Some(arr) = v
                .pointer("/choices/0/message/tool_calls")
                .and_then(|x| x.as_array())
            {
                merge_tool_calls(&mut tool_calls, arr);
            }

            if let Some(rem) = remaining_budget {
                if streamed_tokens as usize >= rem {
                    tool_calls.clear();
                    break 'outer;
                }
            }
        }
    }

    if reasoning_open {
        if color {
            write!(out, "\n╰────────────────────────────────────────────────────\x1b[0m\n\n")?;
        } else {
            write!(out, "\n╰────────────────────────────────────────────────────\n\n")?;
        }
    }
    md_streamer.finish(&mut out)?;
    if printed_any {
        println!();
    }
    out.flush()?;

    stats.elapsed = started.elapsed().as_secs_f64();
    if stats.predicted_n == 0 {
        stats.predicted_n = usage_completion.unwrap_or_else(|| {
            if streamed_tokens > 0 {
                streamed_tokens
            } else if !content.is_empty() || !reasoning.is_empty() {
                (estimate_tokens(&content) + estimate_tokens(&reasoning)) as u64
            } else {
                0
            }
        });
    }
    if stats.prompt_n == 0 {
        stats.prompt_n = usage_prompt.unwrap_or(0);
    }
    if stats.predicted_per_second.is_none() && stats.predicted_n > 0 && stats.elapsed > 0.0 {
        stats.predicted_per_second = Some(stats.predicted_n as f64 / stats.elapsed);
    }

    let content = redact_secrets(&content, secrets);
    let reasoning = redact_secrets(&reasoning, secrets);
    Ok((stats, content, reasoning, tool_calls))
}

impl App {
    fn print_stats(&mut self, s: &Stats) {
        self.session_tokens += s.predicted_n as usize;
        update_top_right_session_badge(self.session_tokens, self.budget);
        if !self.stats {
            return;
        }
        let tg = match s.predicted_per_second {
            Some(v) => format!("{v:.1} tok/s"),
            None => "-".to_string(),
        };
        let session_str = if let Some(b) = self.budget {
            format!("[Session: {}/{} tok]", self.session_tokens, b)
        } else {
            format!("[Session: {} tok]", self.session_tokens)
        };
        let term_w = terminal_width();
        let left = format!("  ↳ {} tokens · {} · {:.1}s", s.predicted_n, tg, s.elapsed);
        let pad = term_w.saturating_sub(left.chars().count() + session_str.chars().count()).max(2);
        eprintln!(
            "\x1b[90m{}{:pad$}\x1b[36m{}\x1b[0m\n",
            left, "", session_str, pad = pad
        );
    }

    fn agent_prompt(&self) -> String {
        let mut prompt = format!(
            "You are Bonsai, an autonomous local CLI agent on Windows.\n\
             Working directory: {}\n\
             Use the provided tools whenever the task requires interacting with this computer.\n\
             - run_command runs PowerShell (default) or cmd. Note: in PowerShell `curl` is an alias for Invoke-WebRequest; use `web_fetch` for web pages or `curl.exe` for curl.\n\
             - read_file reads text files (use `start_line`/`end_line` for large files), edit_file performs targeted find-and-replace edits, write_file creates/overwrites files, list_directory lists folders.\n\
             - search_files searches text patterns recursively across files without spawning a shell.\n\
             - web_fetch fetches HTTP/HTTPS URLs cleanly using the active proxy.\n\
             - Relative paths resolve against the working directory. Every run_command starts there; cd does not persist between calls.\n\
             - The user must approve each tool call unless bypass mode is enabled. If a call is denied, explain and propose an alternative.\n\
             - Never claim that an action succeeded without checking the tool result. Keep the final answer concise.",
            self.cwd.display()
        );
        if self.proxy.is_some() {
            prompt.push_str(
                "\nPROXY IS ENABLED: all network access is mandatory and must go through the configured proxy.\n\
                 - Every command you run, and every script, batch file, program or file you create that performs network access, must use the proxy.\n\
                 - HTTP_PROXY, HTTPS_PROXY, ALL_PROXY, http_proxy, https_proxy, all_proxy and BONSAI_PROXY are already set in every process you launch.\n\
                 - Direct PowerShell commands: the tool runner injects the .NET proxy and Invoke-WebRequest/Invoke-RestMethod defaults automatically. Plain Invoke-RestMethod or Invoke-WebRequest calls are sufficient; do not build proxy objects or re-set proxy variables.\n\
                 - PowerShell scripts run as `powershell -File script.ps1`: a proxy shim is first in PATH and injects the same settings into the nested process automatically. Run them with `powershell` or `powershell.exe`, never with the absolute path C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe, or the shim is bypassed. Do not add proxy code to the script.\n\
                 - cmd and external programs (curl.exe, git, pip, npm, cargo, etc.) receive the proxy through the standard environment variables. If a program ignores them, configure it to read the proxy from the environment instead of a hardcoded value.\n\
                 - For scripts in other languages, read the proxy from the environment at runtime (for example os.environ['HTTPS_PROXY']). Never hardcode credentials or a proxy URL.\n\
                 - Never unset or override these variables, never use --noproxy, -NoProxy, direct-connection flags or hardcoded external IPs, and never contact external hosts directly.\n\
                 - Never inspect, echo, enumerate or write proxy environment variables; their values are secrets. Do not run commands such as echo $env:HTTPS_PROXY or Get-ChildItem Env:.\n\
                 - Never print, log or write proxy credentials into files or chat messages.\n\
                 - Only localhost, 127.0.0.1 and ::1 may bypass the proxy. If a command cannot honor the proxy, stop and tell the user."
            );
        }
        prompt
    }

    fn effective_system(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(s) = self.system.as_ref().filter(|s| !s.trim().is_empty()) {
            parts.push(s.trim().to_string());
        }
        if self.agent {
            parts.push(self.agent_prompt());
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }

    fn request_messages(&self) -> Vec<Value> {
        let mut out = Vec::with_capacity(self.messages.len() + 1);
        if let Some(sys) = self.effective_system() {
            out.push(json!({"role": "system", "content": sys}));
        }
        for m in &self.messages {
            if m.get("role").and_then(|r| r.as_str()) == Some("system") {
                continue;
            }
            out.push(m.clone());
        }
        out
    }

    pub fn estimate_current_tokens(&self) -> usize {
        estimate_total_tokens(&self.messages, self.effective_system().as_deref())
    }

    pub fn auto_compact_if_needed(&mut self) -> bool {
        let current = self.estimate_current_tokens();
        let limit = self.cli.ctx as usize;
        let threshold = (limit * 3) / 4; // 75%
        if current > threshold {
            let mode = if current > (limit * 9 / 10) {
                CompressionMode::Heavy
            } else {
                CompressionMode::Medium
            };
            if let Some(res) = compress_task_aware(&mut self.messages, mode, &self.cwd) {
                let pct = if res.before_tokens > 0 {
                    (res.before_tokens.saturating_sub(res.after_tokens) * 100) / res.before_tokens
                } else {
                    0
                };
                eprintln!(
                    "  \x1b[33m● Context auto-compacted ({}: {}% reduced: {} → {} tokens)\x1b[0m",
                    res.mode.name(),
                    pct,
                    res.before_tokens,
                    res.after_tokens
                );
                return true;
            }
        }
        false
    }

    fn resolve_path(&self, p: &str) -> PathBuf {
        let path = PathBuf::from(p);
        if path.is_absolute() {
            path
        } else {
            self.cwd.join(path)
        }
    }

    fn clipboard_dir() -> PathBuf {
        bonsai_dir().join("clipboard")
    }

    /// Grab the current clipboard image, if any, and return the saved PNG.
    fn attach_clipboard(&mut self) -> Option<PathBuf> {
        let seq = clipboard_seq();
        self.last_clip_seq = seq;
        if !clipboard_has_image() {
            return None;
        }
        let dir = Self::clipboard_dir();
        std::fs::create_dir_all(&dir).ok()?;
        let out = dir.join(format!("clip-{seq}.png"));
        if save_clipboard_image(&out) {
            Some(out)
        } else {
            None
        }
    }

    /// Attach an image copied since the previous message (Ctrl+C then Ctrl+V flow).
    fn try_auto_clipboard(&mut self) -> Option<PathBuf> {
        if !self.interactive {
            return None;
        }
        let seq = clipboard_seq();
        if seq == self.last_clip_seq {
            return None;
        }
        if !clipboard_has_image() {
            self.last_clip_seq = seq;
            return None;
        }
        self.attach_clipboard()
    }

// (uses top-level percent_decode_simple)

    fn proxy_passwords(&self) -> Vec<String> {
        let mut pws = Vec::new();
        if let Some(proxy) = &self.proxy {
            if let Ok(u) = reqwest::Url::parse(proxy) {
                if let Some(pw) = u.password() {
                    if pw.len() >= 4 {
                        pws.push(pw.to_string());
                    }
                    if !pw.is_empty() {
                        let decoded = percent_decode_simple(pw);
                        if decoded != pw && decoded.len() >= 4 {
                            pws.push(decoded);
                        }
                    }
                }
            }
        }
        if let Some(sub_url) = &self.sub_url {
            if let Ok(u) = reqwest::Url::parse(sub_url) {
                let path = u.path();
                if let Some(pos) = path.find("/sub/") {
                    let token = path[pos + 5..].trim_matches('/');
                    if !token.is_empty() {
                        pws.push(token.to_string());
                    }
                }
                for (k, v) in u.query_pairs() {
                    let k_lower = k.to_ascii_lowercase();
                    if (k_lower.contains("token") || k_lower.contains("pass") || k_lower == "key") && v.len() >= 4 {
                        pws.push(v.to_string());
                    }
                }
            }
        }
        for node in &self.sub_nodes {
            if let Some((_, rest)) = node.raw.split_once("://") {
                let cred_host = match rest.split_once('?') {
                    Some((ch, _)) => ch,
                    None => match rest.split_once('#') {
                        Some((ch, _)) => ch,
                        None => rest,
                    },
                };
                if let Some((cred, _)) = cred_host.split_once('@') {
                    if cred.len() >= 8 {
                        pws.push(cred.to_string());
                    }
                }
            }
        }
        pws
    }

    #[allow(dead_code)]
    fn proxy_password(&self) -> Option<String> {
        self.proxy_passwords().into_iter().next()
    }

    /// Never let proxy credentials reach the model, the terminal or tool output.
    fn redact_proxy(&self, text: &str) -> String {
        let mut out = text.to_string();
        if let Some(proxy) = &self.proxy {
            out = out.replace(proxy, &mask_proxy(proxy));
        }
        if let Some(sub_url) = &self.sub_url {
            out = out.replace(sub_url, &mask_proxy(sub_url));
        }
        let passwords = self.proxy_passwords();
        redact_secrets(&out, &passwords)
    }

    fn approve(&mut self, title: &str, details: &str) -> bool {
        let details = self.redact_proxy(details);
        if self.bypass {
            eprintln!("[bypass] {title}");
            for line in details.lines() {
                eprintln!("  {line}");
            }
            return true;
        }
        if !self.interactive && !std::io::stdin().is_terminal() {
            eprintln!("[tool] {title}: denied (non-interactive input; use --bypass)");
            return false;
        }
        eprintln!("\n[tool] {title}");
        for line in details.lines() {
            eprintln!("  {line}");
        }
        eprint!("Allow? [y]es / [n]o / [a]lways: ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) => {
                eprintln!("(no input) denied");
                false
            }
            Ok(_) => match line.trim().to_ascii_lowercase().as_str() {
                "y" | "yes" => true,
                "a" | "always" => {
                    self.bypass = true;
                    eprintln!("bypass = on: further tool calls run without asking");
                    true
                }
                _ => {
                    eprintln!("denied");
                    false
                }
            },
            Err(e) => {
                eprintln!("input error: {e}; denied");
                false
            }
        }
    }

    fn split_command_segments(input: &str) -> Vec<String> {
        let mut segs = Vec::new();
        let mut cur = String::new();
        let mut in_single = false;
        let mut in_double = false;
        let mut chars = input.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '"' if !in_single => {
                    in_double = !in_double;
                    cur.push(c);
                }
                '\'' if !in_double => {
                    in_single = !in_single;
                    cur.push(c);
                }
                ';' | '|' | '\n' if !in_single && !in_double => {
                    if c == '|' && chars.peek() == Some(&'|') {
                        chars.next();
                    }
                    if !cur.trim().is_empty() {
                        segs.push(std::mem::take(&mut cur));
                    } else {
                        cur.clear();
                    }
                }
                '&' if !in_single && !in_double => {
                    if chars.peek() == Some(&'&') {
                        chars.next();
                    }
                    if !cur.trim().is_empty() {
                        segs.push(std::mem::take(&mut cur));
                    } else {
                        cur.clear();
                    }
                }
                _ => cur.push(c),
            }
        }
        if !cur.trim().is_empty() {
            segs.push(cur);
        }
        segs
    }

    fn has_unquoted_shell_ops(s: &str) -> bool {
        let mut in_single = false;
        let mut in_double = false;
        let mut prev = '\0';
        for ch in s.chars() {
            match ch {
                '"' if !in_single && prev != '\\' => in_double = !in_double,
                '\'' if !in_double => in_single = !in_single,
                ';' | '|' | '&' if !in_single && !in_double => return true,
                _ => {}
            }
            prev = ch;
        }
        false
    }

    fn is_protected_root_target(arg: &str) -> bool {
        let mut s = arg
            .trim_matches(['"', '\'', ' '])
            .to_ascii_lowercase()
            .replace('/', "\\");
        if let Some(stripped) = s.strip_prefix(r"\\?\") {
            s = stripped.to_string();
        }
        while s.ends_with('\\') && s.len() > 3 {
            s.pop();
        }
        let bytes = s.as_bytes();
        if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
            let rest = &s[2..];
            if matches!(rest, "" | "\\" | "*" | "\\*") {
                return true;
            }
        }
        matches!(
            s.as_str(),
            "\\" | "*"
                | "\\*"
                | "%systemdrive%"
                | "%systemdrive%\\"
                | "%systemdrive%\\*"
                | "$env:systemdrive"
                | "$env:systemdrive\\"
                | "$env:systemdrive\\*"
                | "%windir%"
                | "%systemroot%"
                | "c:\\windows"
                | "c:\\windows\\*"
                | "c:\\users"
                | "c:\\users\\*"
                | "c:\\program files"
                | "c:\\program files\\*"
                | "c:\\program files (x86)"
                | "c:\\program files (x86)\\*"
                | "c:\\programdata"
                | "c:\\programdata\\*"
        )
    }

    fn is_destructive_command(cmd: &str) -> Option<&'static str> {
        for seg in Self::split_command_segments(cmd) {
            let tokens = Self::split_shell_args(&seg);
            let Some(first) = tokens.first() else {
                continue;
            };
            let cmd_base = Path::new(first)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(first.as_str())
                .to_ascii_lowercase();
            let rest_tokens: Vec<String> = tokens[1..].iter().map(|t| t.to_ascii_lowercase()).collect();

            if cmd_base == "format"
                && rest_tokens.iter().any(|t| {
                    t.starts_with("/fs")
                        || t == "/q"
                        || (t.len() == 2 && t.ends_with(':'))
                })
            {
                return Some("format disk command is blocked");
            }

            if cmd_base == "vssadmin"
                && rest_tokens.iter().any(|t| t == "delete")
                && rest_tokens.iter().any(|t| t == "shadows" || t.starts_with("shadow"))
            {
                return Some("vssadmin shadow copy deletion is blocked");
            }

            if cmd_base == "bcdedit" && rest_tokens.iter().any(|t| t == "/delete" || t == "-delete") {
                return Some("bcdedit deletion is blocked");
            }

            if matches!(
                cmd_base.as_str(),
                "remove-item" | "rmdir" | "rd" | "rm" | "del" | "erase" | "ri"
            ) {
                let has_recurse = rest_tokens.iter().any(|t| {
                    t == "-recurse"
                        || t == "-r"
                        || t == "/s"
                        || (t.starts_with('/') && t.contains('s') && t.len() <= 6)
                        || (t.starts_with('-') && !t.starts_with("--") && t.contains('r') && t.len() <= 5)
                });
                let targets_root = tokens[1..].iter().any(|t| Self::is_protected_root_target(t));
                if has_recurse && targets_root {
                    return Some("recursive deletion of root drive or system directory is blocked");
                }
            }
        }
        None
    }

    fn split_shell_args(input: &str) -> Vec<String> {
        let mut args = Vec::new();
        let mut cur = String::new();
        let mut in_single = false;
        let mut in_double = false;
        let mut chars = input.chars().peekable();

        while let Some(c) = chars.next() {
            match c {
                '\\' if in_double => {
                    if let Some(&next) = chars.peek() {
                        if next == '"' || next == '\\' {
                            cur.push(next);
                            chars.next();
                            continue;
                        }
                    }
                    cur.push('\\');
                }
                '"' if !in_single => {
                    in_double = !in_double;
                }
                '\'' if !in_double => {
                    in_single = !in_single;
                }
                c if c.is_whitespace() && !in_single && !in_double => {
                    if !cur.is_empty() {
                        args.push(std::mem::take(&mut cur));
                    }
                }
                _ => cur.push(c),
            }
        }
        if !cur.is_empty() {
            args.push(cur);
        }
        args
    }

    fn is_powershell_file(path: &Path) -> bool {
        path.extension().map_or(false, |ext| {
            ext.eq_ignore_ascii_case("ps1")
                || ext.eq_ignore_ascii_case("psm1")
                || ext.eq_ignore_ascii_case("psd1")
        })
    }

    fn check_write_guardrails(&self, path: &Path, content: &str) -> Option<&'static str> {
        let path_lower = path.to_string_lossy().to_ascii_lowercase().replace('/', "\\");
        let shim_dir_lower = proxy_shim_dir().to_string_lossy().to_ascii_lowercase().replace('/', "\\");
        let wrapper_lower = ps_wrapper_path().to_string_lossy().to_ascii_lowercase().replace('/', "\\");
        if path_lower.starts_with(&shim_dir_lower) || path_lower == wrapper_lower {
            return Some("error: modifying Bonsai internal proxy shim files is forbidden.");
        }
        let leaks_proxy = self
            .proxy
            .as_ref()
            .map(|p| content.contains(p.as_str()))
            .unwrap_or(false)
            || self
                .proxy_passwords()
                .iter()
                .any(|pw| content.contains(pw.as_str()));
        if leaks_proxy {
            return Some("error: refusing to write proxy credentials into a file. Read the proxy from the environment variables inside the script instead.");
        }
        None
    }

    fn decode_text_bytes(buf: &[u8]) -> Option<String> {
        if buf.starts_with(&[0xFF, 0xFE]) {
            let u16s: Vec<u16> = buf[2..]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            return Some(String::from_utf16_lossy(&u16s));
        }
        if buf.starts_with(&[0xFE, 0xFF]) {
            let u16s: Vec<u16> = buf[2..]
                .chunks_exact(2)
                .map(|c| u16::from_be_bytes([c[0], c[1]]))
                .collect();
            return Some(String::from_utf16_lossy(&u16s));
        }
        if buf.contains(&0) {
            return None;
        }
        let valid_slice = match std::str::from_utf8(buf) {
            Ok(s) => return Some(s.to_string()),
            Err(e) => {
                if e.error_len().is_none() && buf.len().saturating_sub(e.valid_up_to()) < 4 {
                    &buf[..e.valid_up_to()]
                } else {
                    buf
                }
            }
        };
        Some(String::from_utf8_lossy(valid_slice).into_owned())
    }

    fn strip_html_to_text(html: &str) -> String {
        let mut out = String::with_capacity(html.len().min(32768));
        let lower = html.to_ascii_lowercase();
        let bytes = html.as_bytes();
        let lower_bytes = lower.as_bytes();
        let mut i = 0;
        let len = bytes.len();
        let mut in_tag = false;
        let mut tag_name = String::new();
        let mut capturing_tag_name = false;

        while i < len {
            if !in_tag {
                if bytes[i] == b'<' {
                    // Check script or style
                    if lower_bytes[i..].starts_with(b"<script") {
                        if let Some(end) = lower[i..].find("</script>") {
                            i += end + 9;
                            continue;
                        }
                    } else if lower_bytes[i..].starts_with(b"<style") {
                        if let Some(end) = lower[i..].find("</style>") {
                            i += end + 8;
                            continue;
                        }
                    }
                    in_tag = true;
                    tag_name.clear();
                    capturing_tag_name = true;
                    i += 1;
                    continue;
                }
                let ch = html[i..].chars().next().unwrap_or(' ');
                out.push(ch);
                i += ch.len_utf8();
            } else {
                let ch = html[i..].chars().next().unwrap_or(' ');
                if ch == '>' {
                    in_tag = false;
                    let t = tag_name.to_ascii_lowercase();
                    if matches!(
                        t.as_str(),
                        "br" | "/p" | "/div" | "/li" | "/tr" | "/h1" | "/h2" | "/h3" | "/h4" | "/h5" | "/h6" | "li"
                    ) {
                        if !out.ends_with('\n') {
                            out.push('\n');
                        }
                    }
                } else if capturing_tag_name {
                    if ch.is_whitespace() {
                        capturing_tag_name = false;
                    } else {
                        tag_name.push(ch);
                    }
                }
                i += ch.len_utf8();
            }
        }
        let decoded = out
            .replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'");
        let mut clean = String::new();
        let mut empty_run = 0;
        for line in decoded.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                empty_run += 1;
                if empty_run <= 1 && !clean.is_empty() {
                    clean.push('\n');
                }
            } else {
                empty_run = 0;
                clean.push_str(trimmed);
                clean.push('\n');
            }
        }
        clean
    }

    async fn tool_run_command(&mut self, args: &Value) -> String {
        let command = match args.get("command").and_then(|v| v.as_str()) {
            Some(c) if !c.trim().is_empty() => c.to_string(),
            _ => return "error: missing 'command'".to_string(),
        };
        if let Some(reason) = Self::is_destructive_command(&command) {
            return format!("error: command blocked by safety guardrail: {reason}");
        }
        if self.proxy.is_some() {
            let lower_cmd = command.to_ascii_lowercase().replace('/', "\\");
            if lower_cmd.contains(r"\windowspowershell") || lower_cmd.contains(r"\powershell\7") || lower_cmd.contains("pwsh.exe") {
                return "error: running PowerShell via absolute system path bypasses the mandatory proxy shim. Run 'powershell' or 'powershell.exe' instead.".to_string();
            }
        }
        let shell = match args
            .get("shell")
            .and_then(|v| v.as_str())
            .unwrap_or("powershell")
        {
            "powershell" | "pwsh" | "ps" => "powershell",
            "cmd" | "cmd.exe" | "batch" => "cmd",
            other => return format!("error: unsupported shell '{other}' (use powershell or cmd)"),
        };
        let cwd = args
            .get("cwd")
            .and_then(|v| v.as_str())
            .map(|s| self.resolve_path(s))
            .unwrap_or_else(|| self.cwd.clone());
        let timeout = args
            .get("timeout_sec")
            .and_then(|v| v.as_u64())
            .unwrap_or(120)
            .clamp(1, 600);
        let details = format!(
            "shell: {shell}\ncwd: {}\ncommand:\n{command}",
            cwd.display()
        );
        if !self.approve(&format!("run_command ({shell})"), &details) {
            return "error: the user denied permission to run this command. Do not retry it; ask how to proceed.".to_string();
        }
        if !cwd.is_dir() {
            return format!("error: working directory does not exist: {}", cwd.display());
        }

        let mut cmd = if shell == "cmd" {
            let mut c = tokio::process::Command::new("cmd.exe");
            c.arg("/D")
                .arg("/C")
                .arg(format!("chcp 65001>nul & {command}"));
            c
        } else {
            let trimmed = command.trim();
            let lower = trimmed.to_ascii_lowercase();
            let ps_bin = if self.proxy.is_some() {
                proxy_shim_dir().join("powershell.exe")
            } else {
                real_powershell_exe()
            };
            let mut c = tokio::process::Command::new(ps_bin);
            let prelude = if self.proxy.is_some() {
                PS_PROXY_PRELUDE
            } else {
                ""
            };
            if (lower.starts_with("powershell ") || lower.starts_with("powershell.exe "))
                && !Self::has_unquoted_shell_ops(trimmed)
            {
                let rest_str = if lower.starts_with("powershell.exe ") {
                    trimmed[15..].trim()
                } else {
                    trimmed[11..].trim()
                };
                let parsed_args = Self::split_shell_args(rest_str);
                c.args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                ]);
                c.args(parsed_args);
            } else {
                c.args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                ])
                .arg(format!(
                    "{prelude}$OutputEncoding=[Console]::OutputEncoding=[Text.Encoding]::UTF8; & {{ {command} }}; if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }} elseif (-not $?) {{ exit 1 }} else {{ exit 0 }}"
                ));
            }
            c
        };
        if let Some(proxy) = &self.proxy {
            if let Err(e) = ensure_ps_shim(false) {
                return format!("error: cannot prepare the PowerShell proxy shim: {e:#}");
            }
            apply_proxy_env(&mut cmd, proxy);
        }
        cmd.current_dir(&cwd)
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true)
            .creation_flags(CREATE_NO_WINDOW);

        match tokio::time::timeout(Duration::from_secs(timeout), cmd.output()).await {
            Err(_) => {
                eprintln!("[tool] timeout after {timeout}s");
                format!("error: command timed out after {timeout}s and was killed")
            }
            Ok(Err(e)) => format!("error: failed to start command: {e}"),
            Ok(Ok(o)) => {
                let code = o.status.code().unwrap_or(-1);
                eprintln!("[tool] exit={code}");
                let stdout = self.redact_proxy(&String::from_utf8_lossy(&o.stdout));
                let stderr = self.redact_proxy(&String::from_utf8_lossy(&o.stderr));
                let mut out = format!("exit_code: {code}\n");
                if stdout.trim().is_empty() {
                    out.push_str("stdout: (empty)\n");
                } else {
                    out.push_str(&format!("stdout:\n{}\n", clip_text(&stdout, 6000)));
                }
                if !stderr.trim().is_empty() {
                    out.push_str(&format!("stderr:\n{}\n", clip_text(&stderr, 3000)));
                }
                out
            }
        }
    }

    async fn tool_read_file(&mut self, args: &Value) -> String {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) if !p.trim().is_empty() => self.resolve_path(p),
            _ => return "error: missing 'path'".to_string(),
        };
        let max_bytes = args
            .get("max_bytes")
            .and_then(|v| v.as_u64())
            .unwrap_or(32_768)
            .clamp(1, 131_072) as usize;
        let start_line = args.get("start_line").and_then(|v| v.as_u64()).map(|n| n.max(1) as usize);
        let end_line = args.get("end_line").and_then(|v| v.as_u64()).map(|n| n.max(1) as usize);

        let desc = match (start_line, end_line) {
            (Some(s), Some(e)) => format!("path: {} (lines {s}..={e})", path.display()),
            (Some(s), None) => format!("path: {} (from line {s})", path.display()),
            (None, Some(e)) => format!("path: {} (lines 1..={e})", path.display()),
            (None, None) => format!("path: {}", path.display()),
        };
        if !self.approve("read_file", &desc) {
            return "error: the user denied permission to read this file.".to_string();
        }
        let meta = match tokio::fs::metadata(&path).await {
            Ok(m) => m,
            Err(e) => return format!("error: cannot stat {}: {e}", path.display()),
        };
        if meta.is_dir() {
            return format!("error: {} is a directory (use list_directory)", path.display());
        }

        if start_line.is_some() || end_line.is_some() {
            if meta.len() > 8 * 1024 * 1024 {
                return format!("error: file {} is too large ({} bytes) for line-range scan", path.display(), meta.len());
            }
            let raw = match tokio::fs::read(&path).await {
                Ok(b) => b,
                Err(e) => return format!("error: read failed: {e}"),
            };
            let Some(full_text) = Self::decode_text_bytes(&raw) else {
                return format!("binary file: {} ({} bytes); not decoded as text", path.display(), meta.len());
            };
            let all_lines: Vec<&str> = full_text.lines().collect();
            let total = all_lines.len();
            let s = start_line.unwrap_or(1).min(total.max(1));
            let e = end_line.unwrap_or_else(|| (s + 400).min(total)).min(total);
            if s > e || total == 0 {
                return format!("path: {}\nsize: {} bytes (lines: {total})\n---\n(empty range)", path.display(), meta.len());
            }
            let mut body = String::new();
            for (idx, line) in all_lines[s - 1..e].iter().enumerate() {
                body.push_str(&format!("{}: {}\n", s + idx, line));
                if body.len() >= max_bytes {
                    body.push_str("...[truncated by max_bytes]...");
                    break;
                }
            }
            let out = format!(
                "path: {}\nsize: {} bytes (lines {s}..={e} of {total})\n---\n{}",
                path.display(),
                meta.len(),
                body
            );
            return self.redact_proxy(&out);
        }

        let file = match tokio::fs::File::open(&path).await {
            Ok(f) => f,
            Err(e) => return format!("error: cannot open {}: {e}", path.display()),
        };
        let mut buf = Vec::new();
        if let Err(e) = file.take(max_bytes as u64).read_to_end(&mut buf).await {
            return format!("error: read failed: {e}");
        }
        let Some(text_raw) = Self::decode_text_bytes(&buf) else {
            return format!(
                "binary file: {} ({} bytes); not decoded as text",
                path.display(),
                meta.len()
            );
        };
        let text = self.redact_proxy(&text_raw);
        let mut out = format!(
            "path: {}\nsize: {} bytes\n---\n{}",
            path.display(),
            meta.len(),
            clip_text(&text, max_bytes)
        );
        if meta.len() > max_bytes as u64 {
            out.push_str("\n...[truncated by max_bytes; use start_line/end_line to read further]...");
        }
        self.redact_proxy(&out)
    }

    async fn tool_write_file(&mut self, args: &Value) -> String {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) if !p.trim().is_empty() => self.resolve_path(p),
            _ => return "error: missing 'path'".to_string(),
        };
        let mut content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return "error: missing 'content'".to_string(),
        };
        if Self::is_powershell_file(&path) && !content.starts_with('\u{feff}') {
            content = format!("\u{feff}{content}");
        }
        if let Some(err) = self.check_write_guardrails(&path, &content) {
            return err.to_string();
        }
        let exists = path.exists();
        let details = format!(
            "path: {}\n{}{} bytes\ncontent:\n{}",
            path.display(),
            if exists { "overwrite existing file\n" } else { "" },
            content.len(),
            clip_text(&content, 1500)
        );
        if !self.approve("write_file", &details) {
            return "error: the user denied permission to write this file.".to_string();
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = tokio::fs::create_dir_all(parent).await {
                    return format!("error: cannot create {}: {e}", parent.display());
                }
            }
        }
        match tokio::fs::write(&path, content.as_bytes()).await {
            Ok(()) => {
                eprintln!("[tool] wrote {} bytes", content.len());
                format!("wrote {} bytes to {}", content.len(), path.display())
            }
            Err(e) => format!("error: cannot write {}: {e}", path.display()),
        }
    }

    async fn tool_edit_file(&mut self, args: &Value) -> String {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) if !p.trim().is_empty() => self.resolve_path(p),
            _ => return "error: missing 'path'".to_string(),
        };
        let target = match args
            .get("target")
            .or_else(|| args.get("old_string"))
            .and_then(|v| v.as_str())
        {
            Some(t) if !t.is_empty() => t,
            _ => return "error: missing or empty 'target'".to_string(),
        };
        let replacement = match args
            .get("replacement")
            .or_else(|| args.get("new_string"))
            .and_then(|v| v.as_str())
        {
            Some(r) => r,
            None => return "error: missing 'replacement'".to_string(),
        };
        let replace_all = args.get("replace_all").and_then(|v| v.as_bool()).unwrap_or(false);

        if let Some(err) = self.check_write_guardrails(&path, replacement) {
            return err.to_string();
        }
        let raw = match tokio::fs::read(&path).await {
            Ok(b) => b,
            Err(e) => return format!("error: cannot read {}: {e}", path.display()),
        };
        let Some(orig) = Self::decode_text_bytes(&raw) else {
            return format!("error: {} is binary or invalid text", path.display());
        };

        let (eff_target, eff_repl) = if orig.contains("\r\n") && !target.contains("\r\n") && orig.contains(&target.replace('\n', "\r\n")) {
            (target.replace('\n', "\r\n"), replacement.replace('\n', "\r\n"))
        } else {
            (target.to_string(), replacement.to_string())
        };

        let count = orig.matches(&eff_target).count();
        if count == 0 {
            return format!("error: 'target' not found in {}", path.display());
        }
        if count > 1 && !replace_all {
            return format!("error: 'target' matched {count} occurrences in {}; provide more context or set replace_all=true", path.display());
        }

        let details = format!(
            "path: {} ({count} match{})\n--- target:\n{}\n+++ replacement:\n{}",
            path.display(),
            if count == 1 { "" } else { "es" },
            clip_text(&eff_target, 600),
            clip_text(&eff_repl, 600),
        );
        if !self.approve("edit_file", &details) {
            return "error: the user denied permission to edit this file.".to_string();
        }

        let mut updated = if replace_all {
            orig.replace(&eff_target, &eff_repl)
        } else {
            orig.replacen(&eff_target, &eff_repl, 1)
        };
        if Self::is_powershell_file(&path) && !updated.starts_with('\u{feff}') {
            updated = format!("\u{feff}{updated}");
        }
        match tokio::fs::write(&path, updated.as_bytes()).await {
            Ok(()) => {
                let n = if replace_all { count } else { 1 };
                eprintln!("[tool] edited {} ({n} replacement{})", path.display(), if n == 1 { "" } else { "s" });
                format!("edited {} ({n} replacement{})", path.display(), if n == 1 { "" } else { "s" })
            }
            Err(e) => format!("error: cannot write {}: {e}", path.display()),
        }
    }

    fn tool_search_files(&mut self, args: &Value) -> String {
        let pattern = match args
            .get("pattern")
            .or_else(|| args.get("query"))
            .and_then(|v| v.as_str())
        {
            Some(p) if !p.is_empty() => p.to_string(),
            _ => return "error: missing 'pattern'".to_string(),
        };
        let base_dir = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) if !p.trim().is_empty() => self.resolve_path(p),
            _ => self.cwd.clone(),
        };
        let file_filter = args
            .get("file_pattern")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().trim_start_matches('*').to_ascii_lowercase())
            .filter(|s| !s.is_empty());
        let case_sensitive = args.get("case_sensitive").and_then(|v| v.as_bool()).unwrap_or(false);
        let max_results = args.get("max_results").and_then(|v| v.as_u64()).unwrap_or(50).clamp(1, 200) as usize;

        if !self.approve("search_files", &format!("path: {}\npattern: {pattern}", base_dir.display())) {
            return "error: the user denied permission to search files.".to_string();
        }
        if !base_dir.exists() {
            return format!("error: path does not exist: {}", base_dir.display());
        }

        let needle = if case_sensitive { pattern.clone() } else { pattern.to_lowercase() };
        let mut stack = vec![base_dir.clone()];
        let mut matches = Vec::new();
        let mut scanned_files = 0usize;
        const MAX_FILES_SCAN: usize = 5000;

        while let Some(cur) = stack.pop() {
            if matches.len() >= max_results || scanned_files >= MAX_FILES_SCAN {
                break;
            }
            if cur.is_file() {
                if let Some(flt) = &file_filter {
                    let fname = cur.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
                    if !fname.ends_with(flt) && !fname.contains(flt) {
                        continue;
                    }
                }
                scanned_files += 1;
                if let Ok(meta) = std::fs::metadata(&cur) {
                    if meta.len() > 2 * 1024 * 1024 {
                        continue;
                    }
                }
                if let Ok(bytes) = std::fs::read(&cur) {
                    if let Some(text) = Self::decode_text_bytes(&bytes) {
                        let rel = cur.strip_prefix(&base_dir).unwrap_or(&cur);
                        for (idx, line) in text.lines().enumerate() {
                            let hay = if case_sensitive {
                                line.to_string()
                            } else {
                                line.to_lowercase()
                            };
                            if hay.contains(&needle) {
                                let preview: String = line.trim().chars().take(160).collect();
                                matches.push(format!("{}:{}: {}", rel.display(), idx + 1, preview));
                                if matches.len() >= max_results {
                                    break;
                                }
                            }
                        }
                    }
                }
            } else if let Ok(rd) = std::fs::read_dir(&cur) {
                for entry in rd.flatten() {
                    let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                    let path = entry.path();
                    if path.is_dir() {
                        if matches!(
                            name.as_str(),
                            ".git" | "target" | "node_modules" | ".venv" | "venv" | "__pycache__" | "dist"
                        ) {
                            continue;
                        }
                        stack.push(path);
                    } else {
                        stack.push(path);
                    }
                }
            }
        }

        if matches.is_empty() {
            return format!("no matches for '{pattern}' in {} (scanned {scanned_files} files)", base_dir.display());
        }
        let truncated_note = if matches.len() >= max_results {
            format!("\n...(showing first {max_results} matches)")
        } else if scanned_files >= MAX_FILES_SCAN {
            format!("\n...(stopped after scanning {MAX_FILES_SCAN} files)")
        } else {
            String::new()
        };
        let out = format!(
            "matches in {} ({} found across {} files):\n{}{}",
            base_dir.display(),
            matches.len(),
            scanned_files,
            matches.join("\n"),
            truncated_note
        );
        self.redact_proxy(&out)
    }

    async fn tool_web_fetch(&mut self, args: &Value) -> String {
        let url = match args.get("url").and_then(|v| v.as_str()) {
            Some(u) if u.trim().starts_with("http://") || u.trim().starts_with("https://") => u.trim().to_string(),
            _ => return "error: missing or invalid 'url' (must start with http:// or https://)".to_string(),
        };
        let raw_html = args
            .get("raw_html")
            .or_else(|| args.get("raw"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let max_chars = args.get("max_chars").and_then(|v| v.as_u64()).unwrap_or(12_000).clamp(500, 40_000) as usize;
        let timeout_sec = args.get("timeout_sec").and_then(|v| v.as_u64()).unwrap_or(20).clamp(3, 60);

        let proxy_note = match &self.proxy {
            Some(p) => format!(" (via proxy {})", mask_proxy(p)),
            None => String::new(),
        };
        if !self.approve("web_fetch", &format!("url: {url}{proxy_note}")) {
            return "error: the user denied permission to fetch this URL.".to_string();
        }

        let is_socks = self
            .proxy
            .as_deref()
            .map(|p| p.starts_with("socks"))
            .unwrap_or(false);

        if is_socks {
            let Some(proxy) = &self.proxy else {
                return "error: proxy vanished".to_string();
            };
            let mut c = tokio::process::Command::new("curl.exe");
            c.args([
                "-sS",
                "-L",
                "--max-time",
                &timeout_sec.to_string(),
                "-A",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BonsaiCLI/0.2",
                "-x",
                proxy,
                &url,
            ])
            .stdin(std::process::Stdio::null())
            .creation_flags(CREATE_NO_WINDOW);
            match c.output().await {
                Ok(o) if o.status.success() => {
                    let body = String::from_utf8_lossy(&o.stdout).into_owned();
                    let text = if !raw_html && (body.contains("<html") || body.contains("<body") || body.contains("<div")) {
                        Self::strip_html_to_text(&body)
                    } else {
                        body
                    };
                    let safe = self.redact_proxy(&text);
                    return format!("url: {url}\n---\n{}", clip_text(&safe, max_chars));
                }
                Ok(o) => {
                    let err = self.redact_proxy(&String::from_utf8_lossy(&o.stderr));
                    return format!("error: curl.exe exited with {}: {}", o.status, err.trim());
                }
                Err(e) => return format!("error: failed to run curl.exe for SOCKS5 proxy: {e}"),
            }
        }

        let mut builder = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_sec));
        builder = match &self.proxy {
            Some(p) => match reqwest::Proxy::all(p) {
                Ok(pr) => builder.proxy(pr),
                Err(e) => return format!("error: invalid proxy configuration: {e}"),
            },
            None => builder.no_proxy(),
        };
        let client = match builder.build() {
            Ok(c) => c,
            Err(e) => return format!("error: HTTP client build failed: {e}"),
        };
        let resp = match client
            .get(&url)
            .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BonsaiCLI/0.2")
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return format!("error: request failed: {}", self.redact_proxy(&format!("{e:#}"))),
        };
        let status = resp.status();
        let ctype = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        let body = match resp.text().await {
            Ok(t) => t,
            Err(e) => return format!("error: failed reading response body (HTTP {status}): {e}"),
        };
        let is_html = ctype.contains("text/html") || body.trim_start().starts_with("<!DOCTYPE") || body.contains("<html");
        let formatted = if !raw_html && is_html {
            Self::strip_html_to_text(&body)
        } else {
            body
        };
        let safe = self.redact_proxy(&formatted);
        format!("status: {status}\nurl: {url}\n---\n{}", clip_text(&safe, max_chars))
    }

    fn tool_list_directory(&mut self, args: &Value) -> String {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) if !p.trim().is_empty() => self.resolve_path(p),
            _ => self.cwd.clone(),
        };
        if !self.approve("list_directory", &format!("path: {}", path.display())) {
            return "error: the user denied permission to list this directory.".to_string();
        }
        let rd = match std::fs::read_dir(&path) {
            Ok(rd) => rd,
            Err(e) => return format!("error: cannot list {}: {e}", path.display()),
        };
        let mut rows = Vec::new();
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            match entry.metadata() {
                Ok(m) if m.is_dir() => rows.push(format!("d {name}/")),
                Ok(m) => rows.push(format!("f {name} ({} bytes)", m.len())),
                Err(_) => rows.push(format!("? {name}")),
            }
        }
        rows.sort_by_key(|r| r.to_ascii_lowercase());
        let total = rows.len();
        rows.truncate(300);
        let mut out = format!("directory: {}\nentries: {total}\n", path.display());
        if rows.is_empty() {
            out.push_str("(empty)\n");
        } else {
            out.push_str(&rows.join("\n"));
            if total > rows.len() {
                out.push_str(&format!("\n...({} more entries omitted)", total - rows.len()));
            }
            out.push('\n');
        }
        self.redact_proxy(&out)
    }

    async fn execute_tool(&mut self, call: &ToolCall) -> String {
        let raw = if call.arguments.trim().is_empty() {
            "{}"
        } else {
            call.arguments.as_str()
        };
        let args: Value = match serde_json::from_str(raw) {
            Ok(v) => v,
            Err(e) => return format!("error: invalid JSON arguments: {e}"),
        };
        match call.name.as_str() {
            "run_command" => self.tool_run_command(&args).await,
            "read_file" => self.tool_read_file(&args).await,
            "write_file" => self.tool_write_file(&args).await,
            "edit_file" => self.tool_edit_file(&args).await,
            "search_files" => self.tool_search_files(&args),
            "web_fetch" => self.tool_web_fetch(&args).await,
            "list_directory" => self.tool_list_directory(&args),
            other => format!("error: unknown tool '{other}'"),
        }
    }

    async fn agent_loop(&mut self) -> Result<()> {
        const MAX_ROUNDS: usize = 16;
        for round in 0..MAX_ROUNDS {
            if let Some(b) = self.budget {
                if self.session_tokens >= b {
                    eprintln!("\n● Session token budget reached (used {} of max {} tokens).", self.session_tokens, b);
                    eprintln!("  Use '/budget <new_limit>' or '/budget off' to continue.");
                    return Ok(());
                }
            }
            self.auto_compact_if_needed();
            let msgs = self.request_messages();
            let body = chat_body(self, &msgs);
            let secrets = self.proxy_passwords();
            CANCEL.store(false, Ordering::SeqCst);
            let rem_budget = self.budget.map(|b| b.saturating_sub(self.session_tokens));
            let (stats, content, reasoning, mut calls) =
                match stream_chat(
                    &self.client,
                    &self.base,
                    body,
                    self.show_reasoning,
                    &CANCEL,
                    &secrets,
                    self.tokens_per_sec,
                    self.load_limit,
                    self.markdown,
                    rem_budget,
                )
                .await
                {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("error: {e:#}");
                        // keep the user turn so /retry can resend it
                        return Ok(());
                    }
                };
            self.print_stats(&stats);

            if CANCEL.load(Ordering::SeqCst) {
                eprintln!("\n[cancelled]");
                return Ok(());
            }

            if calls.is_empty() {
                if !content.is_empty() {
                    self.messages
                        .push(json!({"role": "assistant", "content": content}));
                }
                return Ok(());
            }

            for c in &mut calls {
                if c.id.is_empty() {
                    self.tool_seq += 1;
                    c.id = format!("call_{}", self.tool_seq);
                }
            }
            let tc_json: Vec<Value> = calls
                .iter()
                .map(|c| {
                    json!({
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments}
                    })
                })
                .collect();
            let mut assistant = json!({
                "role": "assistant",
                "content": if content.is_empty() { Value::Null } else { Value::String(content) },
                "tool_calls": tc_json
            });
            if !reasoning.is_empty() {
                assistant["reasoning_content"] = Value::String(reasoning);
            }
            self.messages.push(assistant);

            for (idx, c) in calls.iter().enumerate() {
                if CANCEL.load(Ordering::SeqCst) {
                    eprintln!("\n[cancelled]");
                    for unexecuted in &calls[idx..] {
                        self.messages.push(json!({
                            "role": "tool",
                            "tool_call_id": unexecuted.id,
                            "name": unexecuted.name,
                            "content": "error: tool execution cancelled by user"
                        }));
                    }
                    return Ok(());
                }
                let shown = self.redact_proxy(&c.arguments.replace('\n', " "));
                eprintln!("[agent] {} {}", c.name, clip_text(&shown, 240));
                let result = self.execute_tool(c).await;
                self.messages.push(json!({
                    "role": "tool",
                    "tool_call_id": c.id,
                    "name": c.name,
                    "content": result
                }));
            }
            if round + 1 == MAX_ROUNDS {
                if CANCEL.load(Ordering::SeqCst) {
                    eprintln!("\n[cancelled]");
                    return Ok(());
                }
                if let Some(b) = self.budget {
                    if self.session_tokens >= b {
                        eprintln!(
                            "\n● Session token budget reached (used {} of max {} tokens).",
                            self.session_tokens, b
                        );
                        eprintln!("  Use '/budget <new_limit>' or '/budget off' to continue.");
                        return Ok(());
                    }
                }
                eprintln!("[agent] stopped after {MAX_ROUNDS} tool rounds; finalizing summary...");
                self.auto_compact_if_needed();
                let msgs = self.request_messages();
                let mut body = chat_body(self, &msgs);
                if let Some(map) = body.as_object_mut() {
                    map.remove("tools");
                    map.remove("tool_choice");
                }
                let rem_budget = self.budget.map(|b| b.saturating_sub(self.session_tokens));
                match stream_chat(
                    &self.client,
                    &self.base,
                    body,
                    self.show_reasoning,
                    &CANCEL,
                    &secrets,
                    self.tokens_per_sec,
                    self.load_limit,
                    self.markdown,
                    rem_budget,
                )
                .await
                {
                    Ok((stats, final_content, _, _)) => {
                        self.print_stats(&stats);
                        if !final_content.is_empty() {
                            self.messages.push(json!({"role": "assistant", "content": final_content}));
                        }
                    }
                    Err(e) => eprintln!("summary error: {e:#}"),
                }
            }
        }
        Ok(())
    }

    async fn run_turn(&mut self, text: String, image: Option<PathBuf>) -> Result<()> {
        CANCEL.store(false, Ordering::SeqCst);
        if let Some(b) = self.budget {
            if self.session_tokens >= b {
                eprintln!("● Token budget exceeded: used {} of max {} session tokens.", self.session_tokens, b);
                eprintln!("  Increase or disable the budget with '/budget <new_limit>' or '/budget off' to proceed.");
                return Ok(());
            }
        }
        self.ensure_server(image.is_some()).await?;
        let msg = user_message(&text, image.as_deref())?;
        self.messages.push(msg);
        self.agent_loop().await
    }

    async fn single_shot(&mut self, prompt: String) -> Result<()> {
        let image = self.image.take();
        self.run_turn(prompt, image).await
    }

    fn stop_xray(&mut self) {
        if let Some(mut child) = self.xray_child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        let pid_path = bonsai_dir().join("xray.pid");
        if let Ok(pid_str) = std::fs::read_to_string(&pid_path) {
            if let Ok(pid) = pid_str.trim().parse::<u32>() {
                #[cfg(windows)]
                {
                    let _ = std::process::Command::new("taskkill")
                        .args(["/FI", "IMAGENAME eq xray.exe", "/PID", &pid.to_string(), "/F"])
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .output();
                }
            }
            let _ = std::fs::remove_file(&pid_path);
        }
        let cfg_path = bonsai_dir().join("xray_active.json");
        let _ = std::fs::remove_file(&cfg_path);
    }

    async fn start_xray_node(&mut self, node: &SubNode) -> Result<String> {
        self.stop_xray();
        kill_lingering_xray();

        let xray_exe = find_xray_exe()
            .context("Xray executable not found (expected at C:\\Bonsai\\tools\\xray\\xray.exe)")?;
        let outbound = build_xray_outbound(node)
            .context("failed to build Xray outbound configuration for node")?;

        let listen_port: u16 = 20808;
        let config = json!({
            "log": {
                "loglevel": "error"
            },
            "inbounds": [
                {
                    "port": listen_port,
                    "listen": "127.0.0.1",
                    "protocol": "http"
                }
            ],
            "outbounds": [
                outbound
            ]
        });

        let cfg_path = bonsai_dir().join("xray_active.json");
        let cfg_str = serde_json::to_string_pretty(&config)?;
        std::fs::write(&cfg_path, &cfg_str)
            .with_context(|| format!("failed to write {}", cfg_path.display()))?;

        let mut cmd = std::process::Command::new(&xray_exe);
        if let Some(dir) = xray_exe.parent() {
            cmd.current_dir(dir);
        }
        cmd.args(["run", "-c"])
            .arg(&cfg_path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null());

        #[cfg(windows)]
        {
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let child = cmd.spawn()
            .with_context(|| format!("failed to spawn {}", xray_exe.display()))?;
        let _ = std::fs::write(bonsai_dir().join("xray.pid"), child.id().to_string());
        self.xray_child = Some(child);

        tokio::time::sleep(Duration::from_millis(600)).await;

        // Check if Xray exited immediately (port conflict, bad config, etc.)
        if let Some(child) = self.xray_child.as_mut() {
            if let Ok(Some(status)) = child.try_wait() {
                self.xray_child = None;
                self.stop_xray();
                bail!("Xray bridge exited immediately with status {status}");
            }
        }

        let proxy_url = format!("http://127.0.0.1:{listen_port}");
        let test_client = reqwest::Client::builder()
            .proxy(reqwest::Proxy::all(&proxy_url)?)
            .timeout(Duration::from_secs(7))
            .build()?;
        let check_urls = [
            "https://api.ipify.org",
            "https://ifconfig.me/ip",
            "http://ip-api.com/line/?fields=query",
        ];
        let mut last_err = String::new();
        for url in check_urls {
            match test_client.get(url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    let ip = resp.text().await.unwrap_or_else(|_| "unknown".to_string());
                    let trimmed = ip.lines().next().unwrap_or("unknown").trim().to_string();
                    self.proxy = Some(proxy_url);
                    let _ = ensure_ps_shim(true);
                    return Ok(if trimmed.is_empty() { "connected".to_string() } else { trimmed });
                }
                Ok(resp) => {
                    last_err = format!("{url} returned HTTP {}", resp.status());
                }
                Err(e) => {
                    last_err = format!("{url}: {e}");
                }
            }
        }
        self.stop_xray();
        self.proxy = None;
        self.active_sub_idx = None;
        bail!("Xray proxy connectivity check failed: {last_err}")
    }

    async fn switch_to_sub_node(&mut self, idx: usize) -> Result<()> {
        let node = match self.sub_nodes.get(idx) {
            Some(n) => n.clone(),
            None => bail!("server index {} out of range (1..{})", idx + 1, self.sub_nodes.len()),
        };

        println!("connecting to [{}] {} ({})...", node.id, node.tag, node.proto);
        match self.start_xray_node(&node).await {
            Ok(ip) => {
                self.active_sub_idx = Some(idx);
                println!("[subscription] Connected successfully! Exit IP: {ip}");
                println!("proxy = http://127.0.0.1:20808 (routing via {})", node.tag);
                println!("agent commands & nested scripts now route through {}", node.tag);
            }
            Err(e) => {
                eprintln!("error: failed to start proxy for {}: {e:#}", node.tag);
            }
        }
        Ok(())
    }

    async fn load_subscription(&mut self, url: &str) -> Result<()> {
        println!("[subscription] Fetching server list from {}...", mask_proxy(url));
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(12))
            .no_proxy()
            .build()?;
        let resp = client.get(url)
            .header("User-Agent", "v2rayN/6.23")
            .send()
            .await
            .with_context(|| format!("failed to fetch subscription from {}", mask_proxy(url)))?;
        let status = resp.status();
        if !status.is_success() {
            bail!("subscription server returned HTTP status {status}");
        }
        let body = resp.text().await.context("failed to read subscription body")?;
        let nodes = parse_subscription_feed(&body);
        if nodes.is_empty() {
            bail!("no valid server nodes found in subscription");
        }
        println!("[subscription] Discovered {} servers", nodes.len());
        self.sub_url = Some(url.to_string());
        self.sub_nodes = nodes;

        if let Some(def_idx) = find_default_node(&self.sub_nodes) {
            println!("[subscription] Auto-selecting server [{}]: {}", self.sub_nodes[def_idx].id, self.sub_nodes[def_idx].tag);
            self.switch_to_sub_node(def_idx).await?;
        } else {
            println!("Loaded {} servers. Use '/proxy list' and '/proxy select <number>' to connect.", self.sub_nodes.len());
        }
        println!("Tip: use '/proxy list' to see all servers, or '/proxy select <num|name>' to switch.");
        Ok(())
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.stop_xray();
    }
}

fn clip_text(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max * 3 / 4).collect();
    let tail: String = {
        let t: String = s.chars().rev().take(max / 4).collect();
        t.chars().rev().collect()
    };
    format!("{head}\n...[truncated {} chars]...\n{tail}", count - max)
}

fn tools_spec() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a command on Windows. shell='powershell' (default) runs Windows PowerShell; shell='cmd' runs cmd.exe /C, which can start any executable, batch file, builtin or pipeline. Returns the exit code, stdout and stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command line to execute"},
                        "shell": {"type": "string", "enum": ["powershell", "cmd"], "description": "Shell to use (default powershell)"},
                        "cwd": {"type": "string", "description": "Working directory (absolute or relative to the current working directory)"},
                        "timeout_sec": {"type": "integer", "description": "Timeout in seconds (default 120, max 600)"}
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 or UTF-16 text file, optionally selecting a 1-based line range. Binary files are reported but not decoded.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "start_line": {"type": "integer", "description": "Optional 1-based start line (inclusive)"},
                        "end_line": {"type": "integer", "description": "Optional 1-based end line (inclusive)"},
                        "max_bytes": {"type": "integer", "description": "Maximum bytes to return (default 32768, max 131072)"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a text file (UTF-8). Parent directories are created automatically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "Full file content"}
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Edit an existing file by replacing exact text (target -> replacement) while preserving CRLF line endings and UTF-8 BOM. By default requires target to occur once; set replace_all=true to replace every match.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "target": {"type": "string", "description": "Exact existing text to replace"},
                        "replacement": {"type": "string", "description": "Replacement text"},
                        "replace_all": {"type": "boolean", "description": "Replace all occurrences instead of requiring a unique match (default false)"}
                    },
                    "required": ["path", "target", "replacement"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files and subdirectories of a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path (default current working directory)"}
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Recursively search files for text (fixed substring, skipping .git, target, node_modules). Returns file:line: text matches.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Text to search for"},
                        "path": {"type": "string", "description": "Root directory or file (default current working directory)"},
                        "file_pattern": {"type": "string", "description": "Optional filename/extension filter (e.g. .rs or main.rs)"},
                        "case_sensitive": {"type": "boolean", "description": "Case-sensitive search (default false)"},
                        "max_results": {"type": "integer", "description": "Maximum matches to return (default 50, max 200)"}
                    },
                    "required": ["pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch content from an HTTP or HTTPS URL (honors active Bonsai proxy). Automatically strips HTML tags unless raw_html=true.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "HTTP or HTTPS URL to fetch"},
                        "raw_html": {"type": "boolean", "description": "Return raw body without HTML stripping (default false)"},
                        "max_chars": {"type": "integer", "description": "Maximum characters to return (default 12000, max 40000)"},
                        "timeout_sec": {"type": "integer", "description": "Request timeout in seconds (default 20, max 60)"}
                    },
                    "required": ["url"]
                }
            }
        }
    ])
}

// ---------------------------------------------------------------------------
// Task-Aware Context Compression (TAC)
// Inspired by mengbin92/task-aware-compression:
// - Task-First & Structure-First strategy
// - Recent Window Protection (preserves immediate conversational context)
// - Noise Filtering (strips redundant reasoning traces and verbose tool logs)
// - Structured State Summary (Goal, files touched, commands executed, decisions)
// - Multi-tier modes: Light, Medium, Heavy
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompressionMode {
    Light,
    Medium,
    Heavy,
}

impl CompressionMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().trim() {
            "light" | "1" => Some(Self::Light),
            "medium" | "med" | "2" | "" => Some(Self::Medium),
            "heavy" | "deep" | "3" => Some(Self::Heavy),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Light => "Light",
            Self::Medium => "Medium",
            Self::Heavy => "Heavy",
        }
    }
}

pub struct CompressionResult {
    pub before_tokens: usize,
    pub after_tokens: usize,
    pub before_messages: usize,
    pub after_messages: usize,
    pub mode: CompressionMode,
}

pub fn estimate_tokens(s: &str) -> usize {
    if s.is_empty() {
        return 0;
    }
    let mut ascii_chars = 0usize;
    let mut non_ascii_chars = 0usize;
    for c in s.chars() {
        if c.is_ascii() {
            ascii_chars += 1;
        } else {
            non_ascii_chars += 1;
        }
    }
    (ascii_chars * 10 / 35) + (non_ascii_chars * 10 / 18) + 1
}

pub fn estimate_message_tokens(m: &Value) -> usize {
    let mut total = 4;
    if let Some(c) = m.get("content") {
        if let Some(s) = c.as_str() {
            total += estimate_tokens(s);
        } else if let Some(arr) = c.as_array() {
            for part in arr {
                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                    total += estimate_tokens(text);
                } else if part.get("image_url").is_some() {
                    total += 512;
                }
            }
        }
    }
    if let Some(r) = m.get("reasoning_content").and_then(|v| v.as_str()) {
        total += estimate_tokens(r);
    }
    if let Some(tc) = m.get("tool_calls").and_then(|v| v.as_array()) {
        for call in tc {
            total += 8;
            if let Some(f) = call.get("function") {
                if let Some(name) = f.get("name").and_then(|n| n.as_str()) {
                    total += estimate_tokens(name);
                }
                if let Some(args) = f.get("arguments").and_then(|a| a.as_str()) {
                    total += estimate_tokens(args);
                }
            }
        }
    }
    total
}

pub fn estimate_total_tokens(messages: &[Value], system: Option<&str>) -> usize {
    let mut total = 0;
    if let Some(sys) = system {
        total += estimate_tokens(sys) + 4;
    }
    for m in messages {
        total += estimate_message_tokens(m);
    }
    total
}

fn strip_reasoning_and_condense_tools(messages: &mut [Value]) {
    let len = messages.len();
    for i in 0..len {
        if let Some(obj) = messages[i].as_object_mut() {
            if i + 1 < len {
                obj.remove("reasoning_content");
            }
            if obj.get("role").and_then(|r| r.as_str()) == Some("tool") {
                if let Some(content) = obj.get("content").and_then(|c| c.as_str()) {
                    if content.chars().count() > 500 {
                        obj.insert("content".to_string(), json!(clip_text(content, 400)));
                    }
                }
            }
        }
    }
}

pub fn compress_task_aware(
    messages: &mut Vec<Value>,
    mode: CompressionMode,
    cwd: &Path,
) -> Option<CompressionResult> {
    if messages.is_empty() {
        return None;
    }

    let original = messages.clone();
    let before_tokens = estimate_total_tokens(messages, None);
    let before_msgs = messages.len();

    let protect_recent = match mode {
        CompressionMode::Light => 6,
        CompressionMode::Medium => 4,
        CompressionMode::Heavy => 2,
    };

    if messages.len() <= protect_recent && mode != CompressionMode::Light {
        strip_reasoning_and_condense_tools(messages);
        let after_tokens = estimate_total_tokens(messages, None);
        if after_tokens < before_tokens {
            return Some(CompressionResult {
                before_tokens,
                after_tokens,
                before_messages: before_msgs,
                after_messages: messages.len(),
                mode,
            });
        }
        *messages = original;
        return None;
    }

    match mode {
        CompressionMode::Light => {
            strip_reasoning_and_condense_tools(messages);
        }
        CompressionMode::Medium | CompressionMode::Heavy => {
            // Find a semantic split index to ensure we never orphan a tool message
            // or split in the middle of a tool-call sequence.
            let target_split = messages.len().saturating_sub(protect_recent);
            let mut split_idx = target_split;

            // Align split_idx to a clean turn boundary (preferably a user message)
            if split_idx > 0 && split_idx < messages.len() {
                let forward_user = (split_idx..messages.len())
                    .find(|&i| messages[i].get("role").and_then(|r| r.as_str()) == Some("user"));
                let backward_user = (1..=split_idx)
                    .rev()
                    .find(|&i| messages[i].get("role").and_then(|r| r.as_str()) == Some("user"));

                if let Some(u) = forward_user {
                    if messages.len() - u >= 2 || backward_user.is_none() {
                        split_idx = u;
                    } else if let Some(b) = backward_user {
                        split_idx = b;
                    }
                } else if let Some(b) = backward_user {
                    split_idx = b;
                }
            }

            // Ensure split_idx never lands on a "tool" message
            while split_idx < messages.len() && messages[split_idx].get("role").and_then(|r| r.as_str()) == Some("tool") {
                split_idx += 1;
            }

            if split_idx == 0 || split_idx >= messages.len() {
                // Fallback: strip reasoning and condense oversized tool outputs
                strip_reasoning_and_condense_tools(messages);
                let after_tokens = estimate_total_tokens(messages, None);
                if after_tokens < before_tokens {
                    return Some(CompressionResult {
                        before_tokens,
                        after_tokens,
                        before_messages: before_msgs,
                        after_messages: messages.len(),
                        mode,
                    });
                }
                *messages = original;
                return None;
            }

            let older: Vec<Value> = messages.drain(..split_idx).collect();

            let mut active_goal = String::new();
            for m in older.iter().rev() {
                if m.get("role").and_then(|r| r.as_str()) == Some("user") {
                    if let Some(c) = m.get("content").and_then(|v| v.as_str()) {
                        let trimmed = c.trim();
                        if trimmed.starts_with("[TASK-AWARE") {
                            if let Some(pos) = trimmed.find("\n---\n") {
                                let rest = trimmed[pos + 5..].trim();
                                if !rest.is_empty() {
                                    active_goal = rest.chars().take(300).collect();
                                    break;
                                }
                            }
                            if active_goal.is_empty() {
                                if let Some(pos) = trimmed.find("• Active Task / User Goal: ") {
                                    let rest = &trimmed[pos + "• Active Task / User Goal: ".len()..];
                                    let line = rest.lines().next().unwrap_or("").trim();
                                    if !line.is_empty() {
                                        active_goal = line.chars().take(300).collect();
                                        break;
                                    }
                                } else if let Some(pos) = trimmed.find("• Primary Goal: ") {
                                    let rest = &trimmed[pos + "• Primary Goal: ".len()..];
                                    let line = rest.lines().next().unwrap_or("").trim();
                                    if !line.is_empty() {
                                        active_goal = line.chars().take(300).collect();
                                        break;
                                    }
                                }
                            }
                        } else if !trimmed.is_empty() {
                            let is_short_ack = matches!(trimmed.to_ascii_lowercase().as_str(), "y" | "yes" | "ok" | "да" | "a" | "always" | "confirm");
                            if !is_short_ack {
                                active_goal = trimmed.chars().take(300).collect();
                                break;
                            }
                        }
                    }
                }
            }

            if active_goal.is_empty() {
                if let Some(first_user) = older.iter().find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user")) {
                    if let Some(c) = first_user.get("content").and_then(|v| v.as_str()) {
                        active_goal = c.chars().take(300).collect();
                    }
                }
            }

            let mut target_urls = BTreeSet::new();
            let mut files_touched = BTreeSet::new();
            let mut commands_run = Vec::new();
            let mut key_decisions = Vec::new();

            for m in &older {
                if let Some(c) = m.get("content").and_then(|v| v.as_str()) {
                    for word in c.split_whitespace() {
                        let clean = word.trim_matches(|ch| ch == '(' || ch == ')' || ch == '<' || ch == '>' || ch == '"' || ch == '\'' || ch == ',' || ch == '`' || ch == '.' || ch == '[' || ch == ']');
                        if (clean.starts_with("http://") || clean.starts_with("https://")) && clean.len() < 1024 {
                            target_urls.insert(clean.to_string());
                        }
                    }
                    if let Some(pos) = c.find("• Target URLs & Endpoints:") {
                        let after = &c[pos + "• Target URLs & Endpoints:".len()..];
                        for line in after.lines() {
                            let trimmed = line.trim();
                            if trimmed.starts_with("- ") {
                                target_urls.insert(trimmed[2..].trim().to_string());
                            } else if trimmed.starts_with('•') {
                                break;
                            }
                        }
                    }
                }

                let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
                if role == "assistant" {
                    if let Some(tcs) = m.get("tool_calls").and_then(|v| v.as_array()) {
                        for tc in tcs {
                            if let Some(name) = tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()) {
                                if let Some(args_str) = tc.get("function").and_then(|f| f.get("arguments")).and_then(|a| a.as_str()) {
                                    for word in args_str.split_whitespace() {
                                        let clean = word.trim_matches(|ch| ch == '\\' || ch == '"' || ch == '\'' || ch == ',' || ch == '}' || ch == '{' || ch == '`' || ch == '[' || ch == ']');
                                        if (clean.starts_with("http://") || clean.starts_with("https://")) && clean.len() < 1024 {
                                            target_urls.insert(clean.to_string());
                                        }
                                    }
                                    if let Ok(args) = serde_json::from_str::<Value>(args_str) {
                                        match name {
                                            "write_file" | "read_file" | "edit_file" => {
                                                if let Some(p) = args.get("path").and_then(|v| v.as_str()) {
                                                    files_touched.insert(format!("{name}: {p}"));
                                                }
                                            }
                                            "run_command" => {
                                                if let Some(cmd) = args.get("command").and_then(|v| v.as_str()) {
                                                    let short_cmd: String = cmd.lines().next().unwrap_or("").chars().take(80).collect();
                                                    commands_run.push(short_cmd);
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if let Some(content) = m.get("content").and_then(|v| v.as_str()) {
                        if !content.trim().is_empty() {
                            let first_line = content.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
                            if !first_line.is_empty()
                                && first_line.chars().count() < 160
                                && !first_line.starts_with('{')
                            {
                                key_decisions.push(first_line.to_string());
                            }
                        }
                    }
                }
            }

            let mut summary = String::from("[TASK-AWARE CONTEXT SUMMARY — PREVIOUS HISTORY COMPACTED]\n");
            if !active_goal.is_empty() {
                summary.push_str(&format!("• Active Task / User Goal: {}\n", active_goal.trim()));
            }
            if !target_urls.is_empty() {
                summary.push_str("• Target URLs & Endpoints:\n");
                for u in target_urls.iter().take(8) {
                    summary.push_str(&format!("  - {u}\n"));
                }
            }
            summary.push_str(&format!("• Working Directory: {}\n", cwd.display()));

            if !files_touched.is_empty() {
                summary.push_str("• Files Touched:\n");
                for f in files_touched.iter().take(8) {
                    summary.push_str(&format!("  - {f}\n"));
                }
                if files_touched.len() > 8 {
                    summary.push_str(&format!("  - ... (+{} more files)\n", files_touched.len() - 8));
                }
            }

            if !commands_run.is_empty() {
                summary.push_str("• Commands Executed:\n");
                for c in commands_run.iter().rev().take(5).rev() {
                    summary.push_str(&format!("  - {c}\n"));
                }
            }

            if !key_decisions.is_empty() {
                summary.push_str("• Key State & Findings:\n");
                for d in key_decisions.iter().rev().take(4).rev() {
                    summary.push_str(&format!("  - {d}\n"));
                }
            }
            summary.push_str("• Status: Continuing task from protected recent window below.\n");

            // Merge summary into first user message to preserve strict role alternation
            if !messages.is_empty() && messages[0].get("role").and_then(|r| r.as_str()) == Some("user") {
                if let Some(existing_content) = messages[0].get("content").and_then(|c| c.as_str()) {
                    let combined = format!("{summary}\n---\n{existing_content}");
                    messages[0]["content"] = json!(combined);
                } else if let Some(parts) = messages[0].get_mut("content").and_then(|c| c.as_array_mut()) {
                    parts.insert(0, json!({"type": "text", "text": format!("{summary}\n---\n")}));
                } else {
                    messages.insert(0, json!({ "role": "user", "content": summary }));
                }
            } else {
                messages.insert(0, json!({ "role": "user", "content": summary }));
            }

            strip_reasoning_and_condense_tools(messages);
        }
    }

    let after_tokens = estimate_total_tokens(messages, None);
    if after_tokens >= before_tokens {
        *messages = original;
        return None;
    }
    Some(CompressionResult {
        before_tokens,
        after_tokens,
        before_messages: before_msgs,
        after_messages: messages.len(),
        mode,
    })
}

// ---------------------------------------------------------------------------
// REPL
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct SessionFile {
    system: Option<String>,
    messages: Vec<Value>,
    effort: Effort,
    sampling: Sampling,
    show_reasoning: bool,
    stats: bool,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default)]
    bypass: Option<bool>,
    #[serde(default)]
    agent: Option<bool>,
    #[serde(default)]
    session_tokens: Option<usize>,
    #[serde(default)]
    budget: Option<usize>,
    #[serde(default)]
    think_budget: Option<usize>,
    #[serde(default)]
    tokens_per_sec: Option<f64>,
    #[serde(default)]
    load_limit: Option<u8>,
    #[serde(default)]
    kv_type: Option<String>,
    #[serde(default)]
    spec_type: Option<String>,
    #[serde(default)]
    markdown: Option<bool>,
}

const SLASH_COMMANDS: &[&str] = &[
    "/help",
    "/exit",
    "/quit",
    "/clear",
    "/compact",
    "/compress",
    "/context",
    "/ctx",
    "/budget",
    "/think-budget",
    "/tb",
    "/tokens",
    "/speed",
    "/throttle",
    "/gpu",
    "/power",
    "/ngl",
    "/layers",
    "/kv",
    "/spec",
    "/server",
    "/status",
    "/markdown",
    "/md",
    "/system",
    "/temp",
    "/top-p",
    "/top-k",
    "/min-p",
    "/presence",
    "/reasoning",
    "/effort",
    "/chat",
    "/agent",
    "/bypass",
    "/proxy",
    "/paste",
    "/copy",
    "/export",
    "/cwd",
    "/show",
    "/stats",
    "/image",
    "/model",
    "/save",
    "/load",
    "/retry",
    "/undo",
];

struct BonsaiHelper {
    completer: FilenameCompleter,
    hinter: HistoryHinter,
    sub_nodes: Vec<(usize, String)>,
}

impl Helper for BonsaiHelper {}

impl Hinter for BonsaiHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, ctx: &RlContext<'_>) -> Option<String> {
        self.hinter.hint(line, pos, ctx)
    }
}

impl BonsaiHelper {
    fn new() -> Self {
        Self {
            completer: FilenameCompleter::new(),
            hinter: HistoryHinter::new(),
            sub_nodes: Vec::new(),
        }
    }
}

fn complete_word_from(start: usize, word: &str, candidates: &[&str]) -> (usize, Vec<Pair>) {
    let low = word.to_ascii_lowercase();
    let pairs = candidates
        .iter()
        .filter(|c| c.to_ascii_lowercase().starts_with(&low))
        .map(|c| Pair {
            display: (*c).to_string(),
            replacement: (*c).to_string(),
        })
        .collect();
    (start, pairs)
}

impl Completer for BonsaiHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &RlContext<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let prefix = &line[..pos];
        let trimmed = prefix.trim_start();
        if !trimmed.starts_with('/') {
            return self.completer.complete(line, pos, ctx);
        }

        let cmd_start = prefix.len() - trimmed.len();
        if !trimmed.contains(char::is_whitespace) {
            return Ok(complete_word_from(cmd_start, trimmed, SLASH_COMMANDS));
        }

        let (raw_cmd, rest_raw) = trimmed.split_once(char::is_whitespace).unwrap_or((trimmed, ""));
        let cmd = raw_cmd.to_ascii_lowercase();
        let arg_trimmed = rest_raw.trim_start();
        let arg_start = pos - arg_trimmed.len();

        match cmd.as_str() {
            "/compact" | "/compress" => {
                Ok(complete_word_from(arg_start, arg_trimmed, &["light", "medium", "heavy"]))
            }
            "/kv" => Ok(complete_word_from(
                arg_start,
                arg_trimmed,
                &["f16", "bf16", "q8_0", "q4_0", "q5_0", "iq4_nl", "off"],
            )),
            "/spec" => Ok(complete_word_from(
                arg_start,
                arg_trimmed,
                &["off", "ngram-cache", "ngram-simple", "ngram-mod", "ngram-map-k"],
            )),
            "/reasoning" | "/effort" => Ok(complete_word_from(
                arg_start,
                arg_trimmed,
                &["none", "low", "medium", "high", "xhigh"],
            )),
            "/copy" => Ok(complete_word_from(arg_start, arg_trimmed, &["last", "code"])),
            "/markdown" | "/md" | "/chat" | "/agent" | "/bypass" | "/show" | "/stats" => {
                Ok(complete_word_from(arg_start, arg_trimmed, &["on", "off"]))
            }
            "/budget" | "/think-budget" | "/tb" | "/tokens" | "/speed" | "/throttle" | "/gpu"
            | "/power" => Ok(complete_word_from(arg_start, arg_trimmed, &["off"])),
            "/system" => Ok(complete_word_from(arg_start, arg_trimmed, &["clear", "off"])),
            "/proxy" => {
                let low = arg_trimmed.to_ascii_lowercase();
                if let Some(sub_arg) = low
                    .strip_prefix("select ")
                    .or_else(|| low.strip_prefix("switch "))
                {
                    let sub_start = pos - sub_arg.len();
                    let pairs = self
                        .sub_nodes
                        .iter()
                        .filter(|(id, tag)| {
                            id.to_string().starts_with(sub_arg)
                                || tag.to_ascii_lowercase().contains(sub_arg)
                        })
                        .map(|(id, tag)| Pair {
                            display: format!("{id}: {tag}"),
                            replacement: id.to_string(),
                        })
                        .collect();
                    Ok((sub_start, pairs))
                } else {
                    Ok(complete_word_from(
                        arg_start,
                        arg_trimmed,
                        &["list", "select ", "off"],
                    ))
                }
            }
            "/cwd" | "/image" | "/model" | "/save" | "/load" | "/export" => {
                self.completer.complete(line, pos, ctx)
            }
            _ => Ok((pos, Vec::new())),
        }
    }
}

impl Highlighter for BonsaiHelper {
    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Owned(format!("\x1b[90m{hint}\x1b[0m"))
    }
}

fn is_multiline_incomplete(input: &str) -> bool {
    let trimmed_right = input.trim_end_matches(&['\r', '\n'][..]);
    let is_slash = trimmed_right.trim_start().starts_with('/');

    // Explicit line continuation with trailing `\` (not for slash paths like `/cwd C:\foo\`)
    if !is_slash && trimmed_right.ends_with('\\') {
        return true;
    }

    // Unclosed triple-backtick code fence
    let fence_count = input
        .lines()
        .filter(|l| l.trim_start().starts_with("```"))
        .count();
    if fence_count % 2 == 1 {
        return true;
    }

    // Unclosed triple-quoted block (ignored inside slash commands)
    if !is_slash && input.matches("\"\"\"").count() % 2 == 1 {
        return true;
    }

    false
}

impl Validator for BonsaiHelper {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        if is_multiline_incomplete(ctx.input()) {
            Ok(ValidationResult::Incomplete)
        } else {
            Ok(ValidationResult::Valid(None))
        }
    }
}

fn normalize_user_prompt(raw: &str) -> String {
    let clean = raw.trim_start_matches('\u{feff}').trim();
    if clean.starts_with('/') {
        return clean.to_string();
    }
    clean.replace("\\\r\n", "\n").replace("\\\n", "\n")
}

fn extract_last_assistant_text(messages: &[Value]) -> Option<String> {
    messages.iter().rev().find_map(|m| {
        if m.get("role").and_then(|r| r.as_str()) != Some("assistant") {
            return None;
        }
        let text = m.get("content").and_then(|c| c.as_str())?;
        if text.trim().is_empty() {
            None
        } else {
            Some(text.to_string())
        }
    })
}

fn extract_last_code_block(markdown: &str) -> Option<String> {
    let mut in_fence = false;
    let mut fence_len = 0usize;
    let mut current = String::new();
    let mut last_complete: Option<String> = None;

    for line in markdown.lines() {
        let trimmed = line.trim_start();
        let ticks = trimmed.bytes().take_while(|&b| b == b'`').count();
        if !in_fence && ticks >= 3 {
            in_fence = true;
            fence_len = ticks;
            current.clear();
        } else if in_fence && ticks >= fence_len && trimmed[ticks..].trim().is_empty() {
            in_fence = false;
            last_complete = Some(current.trim_end_matches('\n').to_string());
        } else if in_fence {
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
        }
    }
    last_complete
}

fn format_session_markdown(app: &App) -> String {
    let mut out = String::from("# Bonsai Session Export\n\n");
    if let Some(sys) = &app.system {
        out.push_str(&format!("## System Prompt\n\n{}\n\n---\n\n", app.redact_proxy(sys)));
    }
    for m in &app.messages {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("unknown");
        match role {
            "user" => {
                out.push_str("## User\n\n");
                if let Some(s) = m.get("content").and_then(|c| c.as_str()) {
                    out.push_str(&app.redact_proxy(s));
                    out.push_str("\n\n");
                } else if let Some(parts) = m.get("content").and_then(|c| c.as_array()) {
                    for p in parts {
                        if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                            out.push_str(&app.redact_proxy(t));
                            out.push('\n');
                        } else if p.get("image_url").is_some() {
                            out.push_str("[image attached]\n");
                        }
                    }
                    out.push('\n');
                }
            }
            "assistant" => {
                out.push_str("## Assistant\n\n");
                if let Some(r) = m.get("reasoning_content").and_then(|v| v.as_str()) {
                    if !r.trim().is_empty() {
                        out.push_str("<details>\n<summary>Reasoning</summary>\n\n");
                        out.push_str(&app.redact_proxy(r.trim()));
                        out.push_str("\n\n</details>\n\n");
                    }
                }
                if let Some(s) = m.get("content").and_then(|c| c.as_str()) {
                    if !s.trim().is_empty() {
                        out.push_str(&app.redact_proxy(s));
                        out.push_str("\n\n");
                    }
                }
                if let Some(tcs) = m.get("tool_calls").and_then(|v| v.as_array()) {
                    for tc in tcs {
                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            .unwrap_or("tool");
                        let args = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|a| a.as_str())
                            .unwrap_or("{}");
                        out.push_str(&format!(
                            "```json\n// tool_call: {name}\n{}\n```\n\n",
                            app.redact_proxy(args)
                        ));
                    }
                }
            }
            "tool" => {
                let name = m.get("name").and_then(|n| n.as_str()).unwrap_or("tool");
                let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
                out.push_str(&format!(
                    "### Tool Result (`{name}`)\n\n```text\n{}\n```\n\n",
                    app.redact_proxy(content)
                ));
            }
            _ => {}
        }
    }
    out
}

fn print_help() {
    println!(
        r#"commands:
  /help                     this help
  /exit                     quit (alias /quit)
  /clear                    clear conversation and pending image
  /compact [mode]           compress context: light|medium|heavy (alias /compress)
  /context [size]           show context meter or set size: e.g. /ctx 32k (alias /ctx)
  /budget [tokens|off]      session token budget (e.g. /budget 50k, /budget off)
  /think-budget [tok|off]   max thinking tokens per turn (alias /tb)
  /tokens [rate|off]        limit generation speed in tok/s (alias /speed, default off)
  /throttle [percent|off]   limit GPU/hardware load % (e.g. /throttle 50, alias /gpu)
  /ngl [layers]             number of GPU layers offloaded (e.g. /ngl 99, /ngl 32)
  /kv [type|off]            KV cache type: f16, bf16, q8_0, q4_0, off
  /spec [type|off]          speculative decoding: off, ngram-cache, ngram-simple, ngram-mod
  /server                   show local llama-server config, slots and health (alias /status)
  /markdown [on|off]        terminal markdown styling: bold, code, lists (alias /md)
  /system [text|clear]      set/show/clear system prompt
  /reasoning [level]        none|low|medium|high|xhigh (alias /effort)
  /temp <f>                 temperature (0..2)
  /top-p <f>                top_p (0..1)
  /top-k <n>                top_k (>=0)
  /min-p <f>                min_p (0..1)
  /presence <f>             presence_penalty (-2..2)
  /chat [on|off]            plain chat mode (disables agent tools)
  /agent [on|off]           toggle agent mode (tools enabled)
  /bypass [on|off]          auto-approve every tool call (dangerous)
  /proxy [URL|off]          proxy for all agent commands (http/socks5 or VPN subscription URL)
  /proxy list               list VPN subscription servers
  /proxy select <num|name>  switch to a specific VPN subscription server
  /paste                    attach an image from the clipboard
  /copy [last|code]         copy last reply or last code block to clipboard
  /export [file.md]         export conversation to a Markdown file
  /image <path|none>        image for the next message
  /cwd [path]               show/set the agent working directory
  /show [on|off]            show/hide reasoning
  /stats [on|off]           stats line on/off
  /model <path>             switch model (restarts owned server)
  /save <file>              save session JSON
  /load <file>              load session JSON
  /retry                    resend the last user message
  /undo                     drop the last exchange

notes:
  Tab completes slash commands, subcommands, VPN node IDs and file paths.
  Multiline input: end a prompt line with \ or leave ``` / """ open.
  Ctrl+V pastes text; copying an image attaches it to the next message,
  and /paste grabs the current clipboard image explicitly."#
    );
}

async fn repl(app: &mut App) -> Result<()> {
    let config = Config::builder()
        .completion_type(CompletionType::List)
        .edit_mode(rustyline::EditMode::Emacs)
        .build();
    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(BonsaiHelper::new()));
    let hist = history_path();
    let _ = rl.load_history(&hist);

    let (mode_badge, mode_color) = if !app.agent {
        ("[Chat]", "\x1b[1;36m")
    } else if app.bypass {
        ("[Agent:Bypass]", "\x1b[1;33m")
    } else {
        ("[Agent]", "\x1b[1;32m")
    };

    let proxy_badge = if let Some(idx) = app.active_sub_idx {
        if let Some(node) = app.sub_nodes.get(idx) {
            format!("[{}]", node.tag)
        } else {
            "[on]".to_string()
        }
    } else if app.proxy.is_some() {
        "[on]".to_string()
    } else {
        "[off]".to_string()
    };

    eprintln!();
    eprintln!("  \x1b[1;32m🌿 BONSAI\x1b[0m \x1b[90mv{}\x1b[0m", env!("CARGO_PKG_VERSION"));
    let budget_info = if let Some(b) = app.budget {
        format!("  \x1b[90mBudget:\x1b[0m \x1b[33m[{}/{}]\x1b[0m", app.session_tokens, b)
    } else {
        "".to_string()
    };
    let load_info = if let Some(l) = app.load_limit {
        format!("  \x1b[90mLoad:\x1b[0m \x1b[33m[{l}%]\x1b[0m")
    } else {
        "".to_string()
    };
    let kv_info = if let Some(kv) = &app.kv_type {
        format!("  \x1b[90mKV:\x1b[0m \x1b[36m[{kv}]\x1b[0m")
    } else {
        "".to_string()
    };
    let spec_info = if let Some(sp) = &app.spec_type {
        format!("  \x1b[90mSpec:\x1b[0m \x1b[36m[{sp}]\x1b[0m")
    } else {
        "".to_string()
    };
    eprintln!(
        "  \x1b[90mMode:\x1b[0m {mode_color}{mode_badge}\x1b[0m  \x1b[90mReasoning:\x1b[0m \x1b[35m[{}]\x1b[0m  \x1b[90mProxy:\x1b[0m \x1b[36m{proxy_badge}\x1b[0m{budget_info}{load_info}{kv_info}{spec_info}",
        app.effort.as_str()
    );
    eprintln!("  \x1b[90mCommands: /help  ·  Toggle Chat: /chat  ·  Compact: /compact  ·  Quit: /exit\x1b[0m");
    eprintln!();
    update_top_right_session_badge(app.session_tokens, app.budget);

    loop {
        update_top_right_session_badge(app.session_tokens, app.budget);
        if let Some(helper) = rl.helper_mut() {
            helper.sub_nodes = app
                .sub_nodes
                .iter()
                .map(|n| (n.id, n.tag.clone()))
                .collect();
        }
        let prompt = if !app.agent {
            "bonsai[chat]> "
        } else if app.bypass {
            "bonsai[bypass]> "
        } else {
            "bonsai[agent]> "
        };
        match rl.readline(prompt) {
            Ok(line) => {
                let line = normalize_user_prompt(&line);
                if line.is_empty() {
                    continue;
                }
                let hist_entry = if line.to_ascii_lowercase().starts_with("/proxy") {
                    let rest = line[6..].trim();
                    let rest_low = rest.to_ascii_lowercase();
                    if rest.is_empty()
                        || matches!(
                            rest_low.as_str(),
                            "off" | "none" | "clear" | "list" | "servers" | "nodes"
                        )
                        || rest_low.starts_with("select ")
                        || rest_low.starts_with("switch ")
                    {
                        line.clone()
                    } else {
                        format!("/proxy {}", mask_proxy(rest))
                    }
                } else {
                    line.clone()
                };
                let _ = rl.add_history_entry(hist_entry.as_str());
                let _ = rl.save_history(&hist);
                if line.starts_with('/') {
                    match handle_command(app, &line, &mut rl).await {
                        Ok(true) => break,
                        Ok(false) => {}
                        Err(e) => eprintln!("command error: {e:#}"),
                    }
                    continue;
                } else {
                    if app.image.is_none() {
                        if let Some(p) = app.try_auto_clipboard() {
                            println!("image attached from clipboard: {}", p.display());
                            app.image = Some(p);
                        }
                    }
                    let image = app.image.take();
                    if let Err(e) = app.run_turn(line, image).await {
                        eprintln!("turn error: {e:#}");
                    }
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("^C");
            }
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(e) => bail!("readline: {e}"),
        }
    }

    let _ = rl.save_history(&hist);
    Ok(())
}

/// Returns Ok(true) when the REPL should exit.
async fn handle_command(
    app: &mut App,
    line: &str,
    _rl: &mut Editor<BonsaiHelper, rustyline::history::DefaultHistory>,
) -> Result<bool> {
    let (cmd, rest) = match line.split_once(char::is_whitespace) {
        Some((c, r)) => (c, r.trim()),
        None => (line, ""),
    };
    let cmd_low = cmd.to_ascii_lowercase();
    match cmd_low.as_str() {
        "/help" => print_help(),
        "/exit" | "/quit" => return Ok(true),
        "/clear" => {
            app.messages.clear();
            app.image = None;
            println!("conversation cleared");
        }
        "/compact" | "/compress" => {
            let mode = if rest.is_empty() {
                CompressionMode::Medium
            } else if let Some(m) = CompressionMode::parse(rest) {
                m
            } else {
                eprintln!("unknown compression mode '{rest}'. Available: light, medium, heavy");
                return Ok(false);
            };
            if app.messages.is_empty() {
                println!("conversation is empty, nothing to compact.");
            } else if let Some(res) = compress_task_aware(&mut app.messages, mode, &app.cwd) {
                let pct = if res.before_tokens > 0 {
                    (res.before_tokens.saturating_sub(res.after_tokens) * 100) / res.before_tokens
                } else {
                    0
                };
                println!("● Context compacted ({} mode):", res.mode.name());
                println!("  Tokens:   {} → {} ({}% reduced)", res.before_tokens, res.after_tokens, pct);
                println!("  Messages: {} → {} (preserved task state & recent window)", res.before_messages, res.after_messages);
            } else {
                let cur = app.estimate_current_tokens();
                println!("● Context is already optimal ({} tokens, {} messages).", cur, app.messages.len());
            }
        }
        "/context" | "/ctx" => {
            let arg = rest.trim();
            if !arg.is_empty() {
                match parse_ctx(arg) {
                    Ok(new_ctx) => {
                        let old_ctx = app.cli.ctx;
                        app.cli.ctx = new_ctx;
                        app.server_config_dirty = true;
                        let had_server = app.server.is_some() || read_pid(app.cli.port).is_some();
                        if had_server && !app.cli.no_server {
                            app.restart_tracked_or_pid_server().await;
                            println!("● Context size changed: {old_ctx} → {new_ctx} tokens.");
                            println!("  Local model engine stopped; restarts with -c {new_ctx} on next request.");
                        } else {
                            println!("● Context size set to {new_ctx} tokens (auto-compact threshold: {} tokens).", (new_ctx * 3) / 4);
                        }
                    }
                    Err(e) => {
                        eprintln!("error: {e}. Example: /ctx 32k or /context 16384");
                    }
                }
                return Ok(false);
            }
            let current = app.estimate_current_tokens();
            let max = app.cli.ctx as usize;
            let pct = (current * 100) / max.max(1);
            let bar_len = 20;
            let filled = (pct * bar_len / 100).min(bar_len);
            let bar: String = "█".repeat(filled) + &"░".repeat(bar_len.saturating_sub(filled));
            let color = if pct >= 80 {
                "\x1b[1;31m"
            } else if pct >= 60 {
                "\x1b[1;33m"
            } else {
                "\x1b[1;32m"
            };
            println!("  Context: [{color}{bar}\x1b[0m] {current} / {max} tokens ({pct}%)");
            println!("  Messages: {}  ·  Auto-compact threshold: 75% ({} tokens)", app.messages.len(), (max * 3) / 4);
            println!("  Strategy: Task-Aware Compression (protects recent turns & task state)");
            println!("  Change size: /ctx <size> (e.g. /ctx 32k or /context 16384)");
        }
        "/budget" => {
            let arg = rest.trim();
            if arg.is_empty() {
                match app.budget {
                    Some(b) => {
                        let pct = (app.session_tokens * 100) / b.max(1);
                        println!("budget = {} tokens (used {} of max {}, {}% spent)", b, app.session_tokens, b, pct);
                        println!("  Change or disable with: /budget <tokens> (e.g. /budget 50k, /budget 100000) or /budget off");
                    }
                    None => {
                        println!("budget = off (unlimited session tokens)");
                        println!("  Set limit with: /budget <tokens> (e.g. /budget 50k, /budget 100000) or /budget off");
                    }
                }
            } else if matches!(arg.to_ascii_lowercase().as_str(), "off" | "none" | "0" | "disable" | "clear") {
                app.budget = None;
                println!("budget = off (session token budget disabled)");
                update_top_right_session_badge(app.session_tokens, app.budget);
            } else {
                match parse_token_limit(arg) {
                    Ok(n) => {
                        app.budget = Some(n);
                        println!("● Session token budget set to {} tokens (current spend: {} tokens).", n, app.session_tokens);
                        update_top_right_session_badge(app.session_tokens, app.budget);
                    }
                    Err(e) => {
                        eprintln!("error: {e}. Usage: /budget <tokens|off> (e.g. /budget 32k, /budget 256, /budget off)");
                    }
                }
            }
        }
        "/think-budget" | "/tb" => {
            let arg = rest.trim();
            if arg.is_empty() {
                match app.think_budget {
                    Some(n) => println!("thinking budget = {n} tokens per turn\n  Change with: /tb <tokens|off>"),
                    None => println!("thinking budget = off (unlimited reasoning tokens)\n  Set with: /tb <tokens> (e.g. /tb 1024, /tb 4k) or /tb off"),
                }
            } else if matches!(arg.to_ascii_lowercase().as_str(), "off" | "none" | "0" | "disable" | "clear") {
                app.think_budget = None;
                println!("thinking budget = off (unlimited reasoning tokens)");
            } else {
                match parse_token_limit(arg) {
                    Ok(n) => {
                        app.think_budget = Some(n);
                        println!("● Per-turn thinking budget set to {n} tokens.");
                    }
                    Err(e) => {
                        eprintln!("error: {e}. Usage: /tb <tokens|off> (e.g. /tb 1024, /tb 4k, /tb off)");
                    }
                }
            }
        }
        "/tokens" | "/speed" => {
            let arg = rest.trim();
            if arg.is_empty() {
                match app.tokens_per_sec {
                    Some(r) => println!("tokens/sec limit = {:.1} tok/s (generation paced)", r),
                    None => println!("tokens/sec limit = off (unlimited generation speed)\n  Set limit with: /tokens <rate> (e.g. /tokens 20 or /tokens 35.5) or /tokens off"),
                }
            } else if matches!(arg.to_ascii_lowercase().as_str(), "off" | "none" | "0" | "disable" | "clear" | "max" | "unlimited") {
                app.tokens_per_sec = None;
                println!("tokens/sec limit = off (unlimited generation speed)");
            } else {
                match parse_tps(arg) {
                    Ok(r) => {
                        app.tokens_per_sec = Some(r);
                        println!("● Generation speed limited to {:.1} tokens/sec.", r);
                    }
                    Err(e) => {
                        eprintln!("error: {e}. Usage: /tokens <rate|off> (e.g. /tokens 25 or /tokens off)");
                    }
                }
            }
        }
        "/throttle" | "/gpu" | "/power" => {
            let arg = rest.trim();
            if arg.is_empty() {
                match app.load_limit {
                    Some(pct) => {
                        println!("load limit = {pct}% (GPU duty cycle throttled to ~{pct}%)");
                        println!("  Change or disable with: /throttle <percent> (e.g. /throttle 50) or /throttle off");
                    }
                    None => {
                        println!("load limit = off (unlimited hardware/GPU utilization, 100% full speed)");
                        println!("  Set limit with: /throttle <percent> (e.g. /throttle 50) or /throttle off");
                    }
                }
            } else if matches!(arg.to_ascii_lowercase().as_str(), "off" | "none" | "clear" | "100" | "100%" | "max" | "unlimited") {
                app.load_limit = None;
                println!("load limit = off (unlimited hardware/GPU utilization, 100% full speed)");
            } else {
                let clean = arg.trim_end_matches('%');
                match clean.parse::<u8>() {
                    Ok(pct) if pct > 0 && pct < 100 => {
                        app.load_limit = Some(pct);
                        println!("● Hardware / GPU load limit set to {pct}% (duty cycle throttled).");
                        println!("  Active inference compute is throttled to approximately {pct}% GPU load.");
                    }
                    _ => {
                        eprintln!("error: load percentage must be between 1 and 99 (e.g. /throttle 50 or /throttle off)");
                    }
                }
            }
        }
        "/ngl" | "/layers" => {
            let arg = rest.trim();
            if arg.is_empty() {
                println!("ngl = {} GPU layers", app.cli.ngl);
                println!("  Change with: /ngl <layers> (e.g. /ngl 99, /ngl 32, /ngl 0)");
            } else {
                match arg.parse::<u32>() {
                    Ok(n) => {
                        let old = app.cli.ngl;
                        app.cli.ngl = n;
                        app.server_config_dirty = true;
                        let had_server = app.server.is_some() || read_pid(app.cli.port).is_some();
                        if had_server && !app.cli.no_server {
                            app.restart_tracked_or_pid_server().await;
                            println!("● GPU layers changed: {old} → {n} layers.");
                            println!("  Local model engine stopped; restarts with -ngl {n} on next request.");
                        } else {
                            println!("● GPU layers set to {n} layers (takes effect when engine starts).");
                        }
                    }
                    Err(_) => {
                        eprintln!("error: ngl must be a non-negative integer (e.g. /ngl 99, /ngl 32, /ngl 0)");
                    }
                }
            }
        }
        "/kv" => {
            let arg = rest.trim();
            if arg.is_empty() {
                println!(
                    "kv cache = {} (use /kv f16|bf16|q8_0|q4_0|off)",
                    app.kv_type.as_deref().unwrap_or("default (f16)")
                );
            } else if matches!(arg.to_ascii_lowercase().as_str(), "off" | "none" | "clear") {
                app.kv_type = None;
                app.server_config_dirty = true;
                if (app.server.is_some() || read_pid(app.cli.port).is_some()) && !app.cli.no_server {
                    app.restart_tracked_or_pid_server().await;
                }
                println!("kv cache = default (f16)");
            } else {
                match parse_kv_type(arg) {
                    Ok(v) => {
                        app.kv_type = if v == "f16" { None } else { Some(v.clone()) };
                        app.server_config_dirty = true;
                        if (app.server.is_some() || read_pid(app.cli.port).is_some()) && !app.cli.no_server {
                            app.restart_tracked_or_pid_server().await;
                            println!("● KV cache quantization set to {v} (local engine restarts on next request).");
                        } else {
                            println!("● KV cache quantization set to {v}.");
                        }
                    }
                    Err(e) => eprintln!("error: {e}"),
                }
            }
        }
        "/spec" => {
            let arg = rest.trim();
            if arg.is_empty() {
                println!(
                    "speculative decoding = {} (use /spec off|ngram-cache|ngram-simple|ngram-mod)",
                    app.spec_type.as_deref().unwrap_or("off")
                );
            } else {
                match parse_spec_type(arg) {
                    Ok(v) => {
                        let next = if v == "none" { None } else { Some(v.clone()) };
                        app.spec_type = next;
                        app.server_config_dirty = true;
                        if (app.server.is_some() || read_pid(app.cli.port).is_some()) && !app.cli.no_server {
                            app.restart_tracked_or_pid_server().await;
                            println!("● Speculative decoding set to {v} (local engine restarts on next request).");
                        } else {
                            println!("● Speculative decoding set to {v}.");
                        }
                    }
                    Err(e) => eprintln!("error: {e}"),
                }
            }
        }
        "/server" | "/status" => {
            let pid = read_pid(app.cli.port);
            let alive = health(&app.client, &app.base).await;
            println!("server endpoint: {}", app.base);
            println!(
                "  status: {}  pidfile({}): {}  pending_restart: {}",
                if alive { "online" } else { "offline" },
                app.cli.port,
                pid.map(|p| p.to_string()).unwrap_or_else(|| "none".to_string()),
                app.server_config_dirty
            );
            println!(
                "  config: ctx={}  ngl={}  kv={}  spec={}  think_budget={}",
                app.cli.ctx,
                app.cli.ngl,
                app.kv_type.as_deref().unwrap_or("default"),
                app.spec_type.as_deref().unwrap_or("off"),
                app.think_budget.map(|b| b.to_string()).unwrap_or_else(|| "off".to_string())
            );
            println!("  model:  {}", app.cli.model);
            if alive {
                if let Some(props) = fetch_server_props(&app.client, &app.base).await {
                    let r_ctx = props
                        .get("default_generation_settings")
                        .and_then(|d| d.get("n_ctx"))
                        .and_then(|v| v.as_u64());
                    let total_slots = props.get("total_slots").and_then(|v| v.as_u64());
                    let vision = props
                        .get("modalities")
                        .and_then(|m| m.get("vision"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let r_model = props.get("model_path").and_then(|v| v.as_str()).unwrap_or("unknown");
                    println!(
                        "  live /props: n_ctx={}  slots={}  vision={}  model={}",
                        r_ctx.map(|c| c.to_string()).unwrap_or_else(|| "?".to_string()),
                        total_slots.map(|s| s.to_string()).unwrap_or_else(|| "?".to_string()),
                        vision,
                        r_model
                    );
                }
                if let Some(slots) = fetch_server_slots(&app.client, &app.base).await.and_then(|v| v.as_array().cloned()) {
                    let busy = slots
                        .iter()
                        .filter(|s| s.get("is_processing").and_then(|v| v.as_bool()) == Some(true))
                        .count();
                    println!("  live /slots: {busy} busy / {} total", slots.len());
                }
            }
        }
        "/markdown" | "/md" => {
            if rest.is_empty() {
                app.markdown = !app.markdown;
            } else {
                app.markdown = toggle(rest, app.markdown, "markdown rendering");
            }
            if app.markdown {
                println!("  \x1b[1;32m●\x1b[0m \x1b[32mMarkdown rendering active\x1b[0m \x1b[90m(bold **text**, `code`, lists, and headers styled)\x1b[0m");
            } else {
                println!("  \x1b[1;33m●\x1b[0m \x1b[33mMarkdown rendering disabled\x1b[0m \x1b[90m(raw markdown text output)\x1b[0m");
            }
        }
        "/system" => {
            if rest.is_empty() {
                match &app.system {
                    Some(s) => println!("system: {s}"),
                    None => println!("system: (none)"),
                }
            } else if matches!(rest.to_ascii_lowercase().as_str(), "off" | "none" | "clear") {
                app.system = None;
                println!("system prompt cleared (default applies in agent mode)");
            } else {
                app.system = Some(rest.to_string());
                println!("system prompt set");
            }
        }
        "/temp" => set_f32_range(&mut app.sampling.temp, rest, "temp", 0.0, 2.0)?,
        "/top-p" => set_f32_range(&mut app.sampling.top_p, rest, "top_p", 0.0, 1.0)?,
        "/top-k" => {
            if !rest.is_empty() {
                let k: i32 = rest.parse().context("top_k must be an integer")?;
                if k < 0 {
                    bail!("top_k must be >= 0 (got {k})");
                }
                app.sampling.top_k = k;
            }
            println!("top_k = {}", app.sampling.top_k);
        }
        "/min-p" => set_f32_range(&mut app.sampling.min_p, rest, "min_p", 0.0, 1.0)?,
        "/presence" => set_f32_range(&mut app.sampling.presence, rest, "presence", -2.0, 2.0)?,
        "/reasoning" | "/effort" => {
            if rest.is_empty() {
                println!(
                    "reasoning = {} ({})",
                    app.effort.as_str(),
                    if app.effort.thinking() {
                        "thinking preset"
                    } else {
                        "instruct preset"
                    }
                );
            } else if let Some(level) = Effort::parse(rest) {
                app.effort = level;
                app.sampling = preset(level);
                println!(
                    "reasoning = {} ({})",
                    level.as_str(),
                    if level.thinking() {
                        "thinking preset"
                    } else {
                        "instruct preset"
                    }
                );
            } else {
                println!("usage: /reasoning none|low|medium|high|xhigh");
            }
        }
        "/chat" => {
            let sub = rest.to_ascii_lowercase();
            if sub.is_empty() {
                app.agent = !app.agent;
            } else if matches!(sub.as_str(), "on" | "1" | "true") {
                app.agent = false;
            } else if matches!(sub.as_str(), "off" | "0" | "false") {
                app.agent = true;
            } else {
                println!("usage: /chat [on|off]");
                return Ok(false);
            }
            if !app.agent {
                println!("  \x1b[1;36m●\x1b[0m \x1b[36mChat mode active\x1b[0m \x1b[90m(plain conversation, agent tools disabled)\x1b[0m");
            } else {
                println!("  \x1b[1;32m●\x1b[0m \x1b[32mAgent mode active\x1b[0m \x1b[90m(tools enabled, system prompt active)\x1b[0m");
            }
        }
        "/agent" => {
            if rest.is_empty() {
                app.agent = !app.agent;
            } else {
                app.agent = toggle(rest, app.agent, "agent");
            }
            if app.agent {
                println!("  \x1b[1;32m●\x1b[0m \x1b[32mAgent mode active\x1b[0m \x1b[90m(tools enabled, system prompt active)\x1b[0m");
            } else {
                println!("  \x1b[1;36m●\x1b[0m \x1b[36mChat mode active\x1b[0m \x1b[90m(plain conversation, tools disabled)\x1b[0m");
            }
        }
        "/bypass" => {
            if rest.is_empty() {
                app.bypass = !app.bypass;
                println!("bypass = {}", if app.bypass { "on" } else { "off" });
            } else {
                app.bypass = toggle(rest, app.bypass, "bypass");
            }
            if app.bypass {
                println!("WARNING: bypass is on - every tool call runs without confirmation");
            }
        }
        "/proxy" => {
            let arg = rest.trim().trim_matches(['"', '\'']);
            let arg_low = arg.to_ascii_lowercase();
            if arg.is_empty() {
                if let Some(idx) = app.active_sub_idx {
                    if let Some(node) = app.sub_nodes.get(idx) {
                        println!("proxy: http://127.0.0.1:20808 (Subscription server [{}]: {} [{}])", node.id, node.tag, node.proto);
                        println!("  total servers: {} (use '/proxy list' or '/proxy select <n>')", app.sub_nodes.len());
                        println!("  NO_PROXY: localhost,127.0.0.1,::1");
                        return Ok(false);
                    }
                }
                match &app.proxy {
                    Some(p) => {
                        println!("proxy: {}", mask_proxy(p));
                        println!("  applies to agent commands (env + PowerShell .NET proxy)");
                        println!("  NO_PROXY: localhost,127.0.0.1,::1");
                    }
                    None => println!("proxy: off (agent commands connect directly)\nTo enable, enter: /proxy <http/socks5 URL> or /proxy <subscription URL>"),
                }
            } else if matches!(arg_low.as_str(), "off" | "none" | "clear") {
                app.stop_xray();
                app.proxy = None;
                app.active_sub_idx = None;
                let _ = std::fs::remove_dir_all(proxy_shim_dir());
                println!("proxy = off");
                println!("agent system prompt updated: proxy requirement removed");
            } else if matches!(arg_low.as_str(), "list" | "servers" | "nodes") {
                if app.sub_nodes.is_empty() {
                    println!("no subscription loaded yet. Enter: /proxy <subscription URL>");
                } else {
                    println!("Subscription servers ({}):", app.sub_nodes.len());
                    for (i, node) in app.sub_nodes.iter().enumerate() {
                        let is_active = app.active_sub_idx == Some(i);
                        let marker = if is_active { "* " } else { "  " };
                        let active_label = if is_active { " (active)" } else { "" };
                        println!("{}{:2}. [{}] {}{}", marker, node.id, node.proto, node.tag, active_label);
                    }
                    println!("\nUse '/proxy select <number|name>' to switch servers.");
                }
            } else if arg_low.starts_with("select ") || arg_low.starts_with("switch ") {
                let query = arg.split_once(char::is_whitespace).map(|x| x.1.trim()).unwrap_or("");
                if app.sub_nodes.is_empty() {
                    println!("no subscription loaded yet. Enter: /proxy <subscription URL>");
                } else if let Some((idx, _)) = find_node_by_query(&app.sub_nodes, query) {
                    app.switch_to_sub_node(idx).await?;
                } else {
                    println!("server matching '{query}' not found. Use '/proxy list' to see available servers.");
                }
            } else if is_subscription_url(arg) {
                if let Err(e) = app.load_subscription(arg).await {
                    eprintln!("error: failed to load subscription: {e:#}");
                }
            } else if let Some((idx, _)) = if !app.sub_nodes.is_empty() { find_node_by_query(&app.sub_nodes, arg) } else { None } {
                app.switch_to_sub_node(idx).await?;
            } else {
                match normalize_proxy(arg) {
                    Ok(p) => {
                        app.stop_xray();
                        app.active_sub_idx = None;
                        match ensure_ps_shim(true) {
                            Ok(()) => {
                                println!("proxy = {}", mask_proxy(&p));
                                println!("agent commands now use HTTP_PROXY/HTTPS_PROXY/ALL_PROXY, the PowerShell .NET proxy, and a nested-PowerShell shim");
                                println!("agent system prompt updated: every network action must use this proxy");
                                app.proxy = Some(p);
                            }
                            Err(e) => println!("error: cannot install PowerShell proxy shim: {e:#}"),
                        }
                    }
                    Err(e) => println!("error: {e:#}"),
                }
            }
        }
        "/paste" => {
            if let Some(p) = app.attach_clipboard() {
                app.image = Some(p.clone());
                println!("image attached: {} (next message)", p.display());
            } else {
                println!("clipboard does not contain an image");
            }
        }
        "/copy" => {
            let Some(reply) = extract_last_assistant_text(&app.messages) else {
                println!("nothing to copy yet");
                return Ok(false);
            };
            let mode = rest.trim().to_ascii_lowercase();
            if matches!(mode.as_str(), "code" | "block") {
                if let Some(code) = extract_last_code_block(&reply) {
                    copy_text_to_clipboard(&code)?;
                    println!("copied last code block ({} chars) to clipboard", code.chars().count());
                } else {
                    println!("no fenced code block found in last reply");
                }
            } else {
                copy_text_to_clipboard(&reply)?;
                println!("copied last reply ({} chars) to clipboard", reply.chars().count());
            }
        }
        "/export" => {
            let raw_name = rest.trim().trim_matches(['"', '\'']);
            let target_path = if raw_name.is_empty() {
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                app.resolve_path(&format!("bonsai_export_{ts}.md"))
            } else {
                app.resolve_path(raw_name)
            };
            let md = format_session_markdown(app);
            std::fs::write(&target_path, md)
                .with_context(|| format!("write {}", target_path.display()))?;
            println!("exported {} message(s) to {}", app.messages.len(), target_path.display());
        }
        "/cwd" => {
            let clean = rest.trim().trim_matches(['"', '\'']);
            if clean.is_empty() {
                println!("cwd: {}", app.cwd.display());
            } else {
                let p = app.resolve_path(clean);
                if p.is_dir() {
                    let canon = p.canonicalize().unwrap_or(p);
                    let s = canon.to_string_lossy();
                    let clean = s.strip_prefix(r"\\?\").unwrap_or(&s);
                    app.cwd = PathBuf::from(clean);
                    println!("cwd = {}", app.cwd.display());
                } else {
                    println!("not a directory: {}", p.display());
                }
            }
        }
        "/show" => {
            if rest.is_empty() {
                app.show_reasoning = !app.show_reasoning;
                println!("show reasoning = {}", if app.show_reasoning { "on" } else { "off" });
            } else {
                app.show_reasoning = toggle(rest, app.show_reasoning, "show reasoning");
            }
        }
        "/stats" => {
            if rest.is_empty() {
                app.stats = !app.stats;
                println!("stats = {}", if app.stats { "on" } else { "off" });
            } else {
                app.stats = toggle(rest, app.stats, "stats");
            }
        }
        "/image" => {
            let clean = rest.trim().trim_matches(['"', '\'']);
            if clean.is_empty() {
                match &app.image {
                    Some(p) => println!("image: {}", p.display()),
                    None => println!("image: (none)"),
                }
            } else if matches!(clean.to_ascii_lowercase().as_str(), "none" | "off" | "clear") {
                app.image = None;
                println!("image cleared");
            } else {
                let p = app.resolve_path(clean);
                if !p.is_file() {
                    println!("not found: {}", p.display());
                } else {
                    app.image = Some(p.clone());
                    println!("image set: {} (next message)", p.display());
                }
            }
        }
        "/model" => {
            let clean = rest.trim().trim_matches(['"', '\'']);
            if clean.is_empty() {
                println!("model: {}", app.cli.model);
            } else {
                let resolved = app.resolve_path(clean);
                if !resolved.is_file() {
                    println!("not found: {}", resolved.display());
                } else {
                    app.cli.model = resolved.to_string_lossy().into_owned();
                    app.server_config_dirty = true;
                    if (app.server.is_some() || read_pid(app.cli.port).is_some()) && !app.cli.no_server {
                        app.restart_tracked_or_pid_server().await;
                    }
                    println!("model set: {} (server restarts on next message)", app.cli.model);
                }
            }
        }
        "/save" => {
            if rest.is_empty() {
                println!("usage: /save <file>");
            } else {
                let target_path = app.resolve_path(rest.trim_matches(['"', '\'']));
                let s = SessionFile {
                    system: app.system.clone(),
                    messages: app.messages.clone(),
                    effort: app.effort,
                    sampling: app.sampling,
                    show_reasoning: app.show_reasoning,
                    stats: app.stats,
                    cwd: Some(app.cwd.clone()),
                    bypass: Some(app.bypass),
                    agent: Some(app.agent),
                    session_tokens: Some(app.session_tokens),
                    budget: app.budget,
                    think_budget: app.think_budget,
                    tokens_per_sec: app.tokens_per_sec,
                    load_limit: app.load_limit,
                    kv_type: app.kv_type.clone(),
                    spec_type: app.spec_type.clone(),
                    markdown: Some(app.markdown),
                };
                std::fs::write(&target_path, serde_json::to_string_pretty(&s)?)
                    .with_context(|| format!("write {}", target_path.display()))?;
                println!("saved {} message(s) to {}", app.messages.len(), target_path.display());
            }
        }
        "/load" => {
            let arg = rest.trim();
            let clean_path = arg.trim_matches(['"', '\'']);
            let resolved_file = if clean_path.is_empty() {
                false
            } else {
                app.resolve_path(clean_path).is_file()
            };
            let arg_low = arg.to_ascii_lowercase();
            let is_throttle_off = matches!(arg_low.as_str(), "off" | "none" | "clear" | "100" | "100%" | "max" | "unlimited");
            let is_percent = arg.ends_with('%');
            let parsed_u8 = arg.trim_end_matches('%').parse::<u8>().ok();

            if !resolved_file && (is_throttle_off || is_percent || parsed_u8.is_some()) {
                if is_throttle_off {
                    app.load_limit = None;
                    println!("load limit = off (unlimited hardware/GPU utilization, 100% full speed)");
                } else if let Some(pct) = parsed_u8 {
                    if pct > 0 && pct < 100 {
                        app.load_limit = Some(pct);
                        println!("● Hardware / GPU load limit set to {pct}% (duty cycle throttled).");
                        println!("  Active inference compute is throttled to approximately {pct}% GPU load.");
                    } else {
                        eprintln!("error: load percentage must be between 1 and 99 (or 100% / off)");
                    }
                } else {
                    eprintln!("error: invalid load percentage '{arg}' (use 1-99% or off)");
                }
            } else if rest.is_empty() {
                println!("usage: /load <file.json> (or /throttle <percent|off> to limit GPU load)");
            } else {
                let target_path = app.resolve_path(clean_path);
                let data = std::fs::read_to_string(&target_path)
                    .with_context(|| format!("read {}", target_path.display()))?;
                let s: SessionFile = serde_json::from_str(&data).context("parse session")?;
                app.system = s.system;
                app.messages = s.messages;
                app.effort = s.effort;
                app.sampling = s.sampling;
                app.show_reasoning = s.show_reasoning;
                app.stats = s.stats;
                if let Some(c) = s.cwd {
                    app.cwd = c;
                }
                if let Some(b) = s.bypass {
                    app.bypass = b;
                }
                if let Some(a) = s.agent {
                    app.agent = a;
                }
                if let Some(st) = s.session_tokens {
                    app.session_tokens = st;
                }
                app.budget = s.budget;
                app.think_budget = s.think_budget;
                app.tokens_per_sec = s.tokens_per_sec;
                app.load_limit = s.load_limit;
                if app.kv_type != s.kv_type || app.spec_type != s.spec_type {
                    app.kv_type = s.kv_type;
                    app.spec_type = s.spec_type;
                    app.server_config_dirty = true;
                }
                if let Some(m) = s.markdown {
                    app.markdown = m;
                }
                update_top_right_session_badge(app.session_tokens, app.budget);
                println!("loaded {} message(s) from {}", app.messages.len(), target_path.display());
            }
        }
        "/retry" => {
            CANCEL.store(false, Ordering::SeqCst);
            if let Some(b) = app.budget {
                if app.session_tokens >= b {
                    eprintln!("● Token budget exceeded: used {} of max {} session tokens.", app.session_tokens, b);
                    eprintln!("  Increase or disable the budget with '/budget <new_limit>' or '/budget off' to proceed.");
                    return Ok(false);
                }
            }
            while app
                .messages
                .last()
                .map(|m| m["role"] != "user")
                .unwrap_or(false)
            {
                app.messages.pop();
            }
            if let Some(last) = app
                .messages
                .last()
                .filter(|m| m["role"] == "user")
                .cloned()
            {
                let has_image = last
                    .get("content")
                    .and_then(|c| c.as_array())
                    .map(|parts| parts.iter().any(|p| p["type"] == "image_url"))
                    .unwrap_or(false);
                app.ensure_server(has_image).await?;
                app.agent_loop().await?;
            } else {
                println!("nothing to retry");
            }
        }
        "/undo" => {
            let mut removed = 0;
            while app
                .messages
                .last()
                .map(|m| m["role"] != "user")
                .unwrap_or(false)
            {
                app.messages.pop();
                removed += 1;
            }
            if app
                .messages
                .last()
                .map(|m| m["role"] == "user")
                .unwrap_or(false)
            {
                app.messages.pop();
                removed += 1;
            }
            println!("removed {removed} message(s)");
        }
        other => println!("unknown command {other} (try /help)"),
    }
    Ok(false)
}

fn set_f32_range(slot: &mut f32, rest: &str, name: &str, min: f32, max: f32) -> Result<()> {
    if !rest.is_empty() {
        let v: f32 = rest
            .parse()
            .with_context(|| format!("{name} must be a number"))?;
        if !v.is_finite() || v < min || v > max {
            bail!("{name} must be between {min} and {max} (got {v})");
        }
        *slot = v;
    }
    println!("{name} = {slot}");
    Ok(())
}

fn toggle(rest: &str, current: bool, name: &str) -> bool {
    let norm = rest.trim().to_ascii_lowercase();
    let v = match norm.as_str() {
        "on" | "1" | "true" => true,
        "off" | "0" | "false" => false,
        "" => current,
        _ => {
            println!("usage: /{name} [on|off]");
            return current;
        }
    };
    println!("{name} = {}", if v { "on" } else { "off" });
    v
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    if invoked_as_ps_shim() {
        return ps_shim_main();
    }

    enable_ansi_support();

    let cli = Cli::parse();

    if cli.stop {
        return stop_by_pidfile(cli.port).await;
    }

    if cli.start && cli.prompt.is_some() {
        bail!("--start is interactive; do not combine it with -p/--prompt");
    }

    let prompt = cli.prompt.clone();
    let interactive = prompt.is_none();
    let mut app = App::new(cli, interactive)?;

    if let Some(proxy_arg) = app.cli.proxy.clone() {
        if is_subscription_url(&proxy_arg) {
            if let Err(e) = app.load_subscription(&proxy_arg).await {
                eprintln!("error: failed to load subscription from CLI argument: {e:#}");
            }
        }
    }

    // Cancel generation on Ctrl-C instead of killing the process.
    let _ = ctrlc::set_handler(|| {
        CANCEL.store(true, Ordering::SeqCst);
    });

    if let Some(p) = prompt {
        let text = if p == "-" {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s.trim().to_string()
        } else {
            p
        };
        if text.is_empty() {
            bail!("empty prompt");
        }
        app.single_shot(text).await?;
        if let Some(mut h) = app.server.take() {
            if h.keep_alive {
                eprintln!("server: left running (pid {}, --keep-alive)", h.pid);
            } else {
                h.shutdown().await;
                eprintln!("server: stopped");
            }
        }
        return Ok(());
    }

    repl(&mut app).await?;
    if let Some(mut h) = app.server.take() {
        if h.keep_alive {
            eprintln!("server: left running (pid {}, --keep-alive)", h.pid);
        } else {
            h.shutdown().await;
            eprintln!("server: stopped");
        }
    }
    Ok(())
}

#[cfg(test)]
mod security_audit_tests {
    use super::*;

    #[test]
    fn test_percent_decode_simple() {
        assert_eq!(percent_decode_simple("hello%20world"), "hello world");
        assert_eq!(percent_decode_simple("p%40ssw0rd"), "p@ssw0rd");
        assert_eq!(percent_decode_simple("%21%23%24%25%26%27%28%29"), "!#$%&'()");
        assert_eq!(percent_decode_simple("plain_text"), "plain_text");
        assert_eq!(percent_decode_simple("incomplete%2"), "incomplete%2");
        assert_eq!(percent_decode_simple("invalid%ZZ"), "invalid%ZZ");
    }

    #[test]
    fn test_is_subscription_url() {
        assert!(is_subscription_url("https://alius-sub.com/sub/hudB-GoT849f8DJ9"));
        assert!(is_subscription_url("http://alius-sub.com/sub/test1234"));
        assert!(is_subscription_url("https://example.com/subscribe?token=abc"));
        assert!(is_subscription_url("https://example.com/get?sub=1&token=xyz"));
        assert!(!is_subscription_url("http://user:pass@1.2.3.4:8080"));
        assert!(!is_subscription_url("socks5://127.0.0.1:1080"));
    }

    #[test]
    fn test_mask_proxy() {
        // Subscription URL token masking
        let masked_sub = mask_proxy("https://alius-sub.com/sub/hudB-GoT849f8DJ9");
        assert_eq!(masked_sub, "https://alius-sub.com/sub/***");

        let masked_sub_query = mask_proxy("https://alius-sub.com/sub/hudB-GoT849f8DJ9?flag=1");
        assert_eq!(masked_sub_query, "https://alius-sub.com/sub/***");

        let masked_other_sub = mask_proxy("https://example.com/subscribe?token=hudB-GoT849f8DJ9");
        assert_eq!(masked_other_sub, "https://example.com/***");

        // Proxy credentials masking
        let masked_proxy = mask_proxy("http://tt3957:my_secret_pass@2.27.41.101:8443");
        assert_eq!(masked_proxy, "http://tt3957:***@2.27.41.101:8443/");

        let masked_no_creds = mask_proxy("http://127.0.0.1:20808");
        assert_eq!(masked_no_creds, "http://127.0.0.1:20808/");
    }

    #[test]
    fn test_proxy_passwords_and_redact_proxy() {
        let cli = Cli::parse_from(["bonsai", "--no-agent"]);
        let mut app = App::new(cli, false).unwrap();
        app.proxy = Some("http://tt3957:p%40ssw0rd@2.27.41.101:8443".to_string());
        app.sub_url = Some("https://alius-sub.com/sub/hudB-GoT849f8DJ9".to_string());
        app.sub_nodes = vec![
            SubNode {
                id: 1,
                proto: "vless".to_string(),
                tag: "Frankfurt".to_string(),
                raw: "vless://fb94685a-04ff-4230-b486-f39064d25fe7@updates.stream-online23.blog:8443?security=reality".to_string(),
            }
        ];

        let passwords = app.proxy_passwords();
        // Encoded & decoded proxy passwords
        assert!(passwords.contains(&"p%40ssw0rd".to_string()));
        assert!(passwords.contains(&"p@ssw0rd".to_string()));
        // Subscription token
        assert!(passwords.contains(&"hudB-GoT849f8DJ9".to_string()));
        // VLESS UUID node credential
        assert!(passwords.contains(&"fb94685a-04ff-4230-b486-f39064d25fe7".to_string()));

        // Test redact_proxy
        let sample = "Connecting with http://tt3957:p%40ssw0rd@2.27.41.101:8443, using password p@ssw0rd or encoded p%40ssw0rd, token hudB-GoT849f8DJ9, uuid fb94685a-04ff-4230-b486-f39064d25fe7, sub url https://alius-sub.com/sub/hudB-GoT849f8DJ9";
        let redacted = app.redact_proxy(sample);
        assert!(!redacted.contains("p@ssw0rd"));
        assert!(!redacted.contains("p%40ssw0rd"));
        assert!(!redacted.contains("hudB-GoT849f8DJ9"));
        assert!(!redacted.contains("fb94685a-04ff-4230-b486-f39064d25fe7"));
        assert!(redacted.contains("[REDACTED]"));
    }

    #[tokio::test]
    async fn test_write_file_credential_blocking() {
        let cli = Cli::parse_from(["bonsai", "--bypass", "--no-agent"]);
        let mut app = App::new(cli, false).unwrap();
        app.proxy = Some("http://tt3957:p%40ssw0rd@2.27.41.101:8443".to_string());
        app.sub_url = Some("https://alius-sub.com/sub/hudB-GoT849f8DJ9".to_string());

        let temp_dir = std::env::temp_dir();
        let target = temp_dir.join("test_leak.txt");

        // Try writing raw proxy url
        let res = app.tool_write_file(&json!({
            "path": target.to_str().unwrap(),
            "content": "proxy is http://tt3957:p%40ssw0rd@2.27.41.101:8443"
        })).await;
        assert!(res.contains("refusing to write proxy credentials into a file"));
        assert!(!target.exists());

        // Try writing token
        let res = app.tool_write_file(&json!({
            "path": target.to_str().unwrap(),
            "content": "token = hudB-GoT849f8DJ9"
        })).await;
        assert!(res.contains("refusing to write proxy credentials into a file"));
        assert!(!target.exists());

        // Try writing decoded password
        let res = app.tool_write_file(&json!({
            "path": target.to_str().unwrap(),
            "content": "password = p@ssw0rd"
        })).await;
        assert!(res.contains("refusing to write proxy credentials into a file"));
        assert!(!target.exists());

        // Try writing internal proxy shim files
        let shim_target = proxy_shim_dir().join("powershell.exe");
        let res = app.tool_write_file(&json!({
            "path": shim_target.to_str().unwrap(),
            "content": "malicious shim override"
        })).await;
        assert!(res.contains("modifying Bonsai internal proxy shim files is forbidden"));

        let wrapper_target = ps_wrapper_path();
        let res = app.tool_write_file(&json!({
            "path": wrapper_target.to_str().unwrap(),
            "content": "malicious wrapper override"
        })).await;
        assert!(res.contains("modifying Bonsai internal proxy shim files is forbidden"));
    }

    #[test]
    fn test_is_destructive_command() {
        assert!(App::is_destructive_command("format c: /fs:ntfs /q").is_some());
        assert!(App::is_destructive_command("format d: /q").is_some());
        assert!(App::is_destructive_command("rmdir /s /q c:\\").is_some());
        assert!(App::is_destructive_command("rd /s /q c:/").is_some());
        assert!(App::is_destructive_command("rd /s %systemdrive%").is_some());
        assert!(App::is_destructive_command("Remove-Item -Recurse -Force C:\\").is_some());
        assert!(App::is_destructive_command("remove-item c:\\* -recurse").is_some());
        assert!(App::is_destructive_command("vssadmin delete shadows /all /quiet").is_some());
        assert!(App::is_destructive_command("bcdedit /delete {current}").is_some());

        // Harmless commands
        assert!(App::is_destructive_command("dir").is_none());
        assert!(App::is_destructive_command("echo hello").is_none());
        assert!(App::is_destructive_command("Get-ChildItem -Path C:\\Bonsai").is_none());
        assert!(App::is_destructive_command("rmdir /s my_subfolder").is_none());
    }

    #[tokio::test]
    async fn test_powershell_absolute_path_bypass_guardrail() {
        let cli = Cli::parse_from(["bonsai", "--bypass", "--no-agent"]);
        let mut app = App::new(cli, false).unwrap();
        app.proxy = Some("http://127.0.0.1:20808".to_string());

        // Attempting to bypass shim via System32 PowerShell
        let res = app.tool_run_command(&json!({
            "command": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -File script.ps1"
        })).await;
        assert!(res.contains("running PowerShell via absolute system path bypasses the mandatory proxy shim"));

        // Attempting to bypass shim via SysWOW64 PowerShell
        let res = app.tool_run_command(&json!({
            "command": "c:\\windows\\syswow64\\windowspowershell\\v1.0\\powershell.exe -Command Write-Output 1"
        })).await;
        assert!(res.contains("running PowerShell via absolute system path bypasses the mandatory proxy shim"));
    }

    #[test]
    fn test_parse_ctx_formats() {
        assert_eq!(parse_ctx("8192").unwrap(), 8192);
        assert_eq!(parse_ctx("16k").unwrap(), 16384);
        assert_eq!(parse_ctx("32k").unwrap(), 32768);
        assert_eq!(parse_ctx("32768").unwrap(), 32768);
        assert_eq!(parse_ctx("64K").unwrap(), 65536);
        assert!(parse_ctx("invalid").is_err());
    }

    #[test]
    fn test_task_aware_compression_medium() {
        let mut msgs = vec![
            json!({"role": "user", "content": "Build a REST API server in Rust with tokio"}),
            json!({
                "role": "assistant",
                "content": "I will examine the workspace and create the server.",
                "reasoning_content": "Deep thinking about architecture... ".repeat(50),
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"Cargo.toml\"}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "cargo contents ".repeat(20)}),
            json!({
                "role": "assistant",
                "content": "Now writing main.rs.",
                "tool_calls": [{
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{\"path\":\"src/main.rs\",\"content\":\"fn main(){}\"}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "c2", "name": "write_file", "content": "wrote 14 bytes"}),
            json!({"role": "user", "content": "Now add the router endpoint"}),
            json!({"role": "assistant", "content": "Adding router endpoints."}),
            json!({"role": "user", "content": "What is the status?"}),
            json!({"role": "assistant", "content": "The server is ready."}),
        ];

        let before = estimate_total_tokens(&msgs, None);
        let res = compress_task_aware(&mut msgs, CompressionMode::Medium, Path::new("C:\\Project")).unwrap();

        assert!(res.after_tokens < before);
        // Protected recent window (4 messages) with summary merged into first user message
        assert_eq!(msgs.len(), 4);

        // Verify task state was preserved in summary merged into first user message
        let first_msg = msgs[0]["content"].as_str().unwrap();
        assert!(first_msg.contains("[TASK-AWARE CONTEXT SUMMARY"));
        assert!(first_msg.contains("Build a REST API server"));
        assert!(first_msg.contains("Cargo.toml"));
        assert!(first_msg.contains("src/main.rs"));
        assert!(first_msg.contains("Now add the router endpoint"));

        // Verify recent conversation was preserved verbatim
        assert_eq!(msgs[1]["content"].as_str().unwrap(), "Adding router endpoints.");
        assert_eq!(msgs[2]["content"].as_str().unwrap(), "What is the status?");
        assert_eq!(msgs[3]["content"].as_str().unwrap(), "The server is ready.");
    }

    #[test]
    fn test_split_shell_args() {
        let args = App::split_shell_args("-NoProfile -ExecutionPolicy Bypass -Command \"Write-Host 'Hello World'\"");
        assert_eq!(
            args,
            vec![
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "Write-Host 'Hello World'"
            ]
        );

        let args2 = App::split_shell_args("curl.exe -s -L -o \"C:\\Users\\Дима\\page.html\" http://opencode.ai/");
        assert_eq!(
            args2,
            vec![
                "curl.exe",
                "-s",
                "-L",
                "-o",
                "C:\\Users\\Дима\\page.html",
                "http://opencode.ai/"
            ]
        );
    }

    #[test]
    fn test_task_aware_compression_preserves_target_urls() {
        let mut msgs = vec![
            json!({"role": "user", "content": "что находится на сайте http://opencode.ai/ ?"}),
            json!({
                "role": "assistant",
                "content": "Скачиваю содержимое сайта http://opencode.ai/",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "run_command", "arguments": "{\"command\":\"curl.exe -s -L http://opencode.ai/\"}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "c1", "name": "run_command", "content": "<html><title>OpenCode</title></html>".repeat(20)}),
            json!({"role": "assistant", "content": "Сайт получен, анализирую"}),
            json!({"role": "user", "content": "И что там написано?"}),
            json!({"role": "assistant", "content": "Там страница OpenCode."}),
        ];

        let res = compress_task_aware(&mut msgs, CompressionMode::Medium, Path::new(r"C:\Users\Дима")).unwrap();
        assert!(res.after_tokens < res.before_tokens);
        let first_msg = msgs[0]["content"].as_str().unwrap();
        assert!(first_msg.contains("http://opencode.ai/"), "Target URL must be preserved across compression!");
        assert!(first_msg.contains("что находится на сайте"), "Active user goal must be preserved!");
    }

    #[test]
    fn test_session_file_serde_with_budget_and_tokens() {
        let original = SessionFile {
            system: Some("test system".to_string()),
            messages: vec![json!({"role": "user", "content": "hi"})],
            effort: Effort::High,
            sampling: preset(Effort::High),
            show_reasoning: true,
            stats: true,
            cwd: Some(PathBuf::from(r"C:\Bonsai")),
            bypass: Some(true),
            agent: Some(true),
            session_tokens: Some(12345),
            budget: Some(50000),
            think_budget: Some(2048),
            tokens_per_sec: Some(25.5),
            load_limit: Some(50),
            kv_type: Some("q8_0".to_string()),
            spec_type: Some("ngram-cache".to_string()),
            markdown: Some(true),
        };

        let json_str = serde_json::to_string(&original).unwrap();
        let parsed: SessionFile = serde_json::from_str(&json_str).unwrap();

        assert_eq!(parsed.session_tokens, Some(12345));
        assert_eq!(parsed.budget, Some(50000));
        assert_eq!(parsed.think_budget, Some(2048));
        assert_eq!(parsed.tokens_per_sec, Some(25.5));
        assert_eq!(parsed.load_limit, Some(50));
        assert_eq!(parsed.kv_type.as_deref(), Some("q8_0"));
        assert_eq!(parsed.spec_type.as_deref(), Some("ngram-cache"));
        assert_eq!(parsed.markdown, Some(true));
    }

    #[test]
    fn test_markdown_streamer_bold_and_russian() {
        let mut streamer = MarkdownStreamer::new(true, true);
        let mut out = Vec::new();
        streamer.write("Ваш публичный IP: **157.173.27.200**", &mut out).unwrap();
        streamer.finish(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("\x1b[1;37m157.173.27.200\x1b[0m"));
        assert!(!s.contains("**"));

        let mut out2 = Vec::new();
        streamer.write("**Привет**, мир!", &mut out2).unwrap();
        streamer.finish(&mut out2).unwrap();
        let s2 = String::from_utf8(out2).unwrap();
        assert!(s2.contains("\x1b[1;37mПривет\x1b[0m, мир!"));
        assert!(!s2.contains("**"));
    }

    #[test]
    fn test_markdown_streamer_chunked() {
        let mut streamer = MarkdownStreamer::new(true, true);
        let mut out = Vec::new();
        let chunks = ["**", "При", "вет", "**", " всем"];
        for c in chunks {
            streamer.write(c, &mut out).unwrap();
        }
        streamer.finish(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("\x1b[1;37mПривет\x1b[0m всем"));
        assert!(!s.contains("**"));
    }

    #[test]
    fn test_markdown_streamer_bullet_and_inline_code() {
        let mut streamer = MarkdownStreamer::new(true, true);
        let mut out = Vec::new();
        streamer.write("- **Страна:** Германия\n`ipconfig`\n", &mut out).unwrap();
        streamer.finish(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("•"));
        assert!(s.contains("\x1b[1;37mСтрана:\x1b[0m"));
        assert!(s.contains("\x1b[36mipconfig\x1b[0m"));
        assert!(!s.contains("**"));
        assert!(!s.contains('`'));
    }

    #[test]
    fn test_markdown_streamer_math_multiplication() {
        let mut streamer = MarkdownStreamer::new(true, true);
        let mut out = Vec::new();
        streamer.write("2 * 3 = 6", &mut out).unwrap();
        streamer.finish(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert_eq!(s, "2 * 3 = 6");
    }

    #[test]
    fn test_markdown_streamer_disabled() {
        let mut streamer = MarkdownStreamer::new(false, true);
        let mut out = Vec::new();
        streamer.write("**Привет**", &mut out).unwrap();
        streamer.finish(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert_eq!(s, "**Привет**");
    }

    #[test]
    fn test_markdown_streamer_underscore_bold_and_italic() {
        let mut streamer = MarkdownStreamer::new(true, true);
        let mut out = Vec::new();
        streamer.write("__жирный текст__ и _курсив_", &mut out).unwrap();
        streamer.finish(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("\x1b[1;37mжирный текст\x1b[0m"));
        assert!(s.contains("\x1b[3mкурсив\x1b[0m"));
        assert!(!s.contains("__"));
        assert!(!s.contains('_'));
    }

    #[test]
    fn test_markdown_streamer_identifiers_with_underscores() {
        let mut streamer = MarkdownStreamer::new(true, true);
        let mut out = Vec::new();
        streamer.write("Файл my_cool_variable_name.rs готов", &mut out).unwrap();
        streamer.finish(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert_eq!(s, "Файл my_cool_variable_name.rs готов");
    }

    #[test]
    fn test_markdown_streamer_standalone_backtick_no_leak() {
        let mut streamer = MarkdownStreamer::new(true, true);
        let mut out = Vec::new();
        streamer.write("Символ ` в тексте", &mut out).unwrap();
        streamer.finish(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert_eq!(s, "Символ ` в тексте");
    }

    #[test]
    fn test_markdown_streamer_multidigit_numbered_list() {
        let mut streamer = MarkdownStreamer::new(true, true);
        let mut out = Vec::new();
        streamer.write("9. Девятый\n10. Десятый\n11. Одиннадцатый\n", &mut out).unwrap();
        streamer.finish(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("  \x1b[33m10.\x1b[0m Десятый"));
        assert!(s.contains("  \x1b[33m11.\x1b[0m Одиннадцатый"));
    }

    #[test]
    fn test_markdown_streamer_nested_italic_and_inline_code() {
        let mut streamer = MarkdownStreamer::new(true, true);
        let mut out = Vec::new();
        streamer.write("*курсив с `кодом` внутри*", &mut out).unwrap();
        streamer.finish(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("\x1b[36mкодом\x1b[0m\x1b[3m внутри\x1b[0m"));
    }

    #[test]
    fn test_markdown_streamer_unclosed_code_block_finish() {
        let mut streamer = MarkdownStreamer::new(true, true);
        let mut out = Vec::new();
        streamer.write("```rust\nfn main() {}\n", &mut out).unwrap();
        streamer.finish(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("└──"), "Рамка блока кода должна быть корректно закрыта в finish()");
    }

    #[test]
    fn test_markdown_streamer_redact_secrets_no_markdown_corruption() {
        let secrets = vec!["SuperSecretToken123".to_string()];
        let delta = "Bearer SuperSecretToken123 next text";
        let safe = redact_secrets(delta, &secrets);
        
        let mut streamer = MarkdownStreamer::new(true, true);
        let mut out = Vec::new();
        streamer.write(&safe, &mut out).unwrap();
        streamer.finish(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(!s.contains("\x1b[1;3;37mnext text"));
        assert!(s.contains("[REDACTED]"));
    }

    #[test]
    fn test_is_destructive_command_precise_paths() {
        assert!(App::is_destructive_command("Remove-Item -Recurse -Force C:\\Users\\Дима\\tmp").is_none());
        assert!(App::is_destructive_command("rm -rf C:\\Bonsai\\tmp_build").is_none());
        assert!(App::is_destructive_command("echo confirm -r C:\\temp && dir").is_none());
        assert!(App::is_destructive_command("Remove-Item -Recurse -Force C:\\Windows").is_some());
        assert!(App::is_destructive_command("rmdir /s /q C:\\").is_some());
        assert!(App::is_destructive_command("rm -rf c:/users").is_some());
        assert!(App::is_destructive_command("format C:").is_some());
    }

    #[test]
    fn test_markdown_streamer_fixes_and_extensions() {
        // Intraword underscores/asterisks preserved without false formatting
        let mut st = MarkdownStreamer::new(true, true);
        let mut out = Vec::new();
        st.write("`__init__` and foo__bar and foo_bar_baz and a*b*c", &mut out).unwrap();
        st.finish(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("\x1b[36m__init__\x1b[0m"), "unexpected mangling in {s:?}");
        assert!(s.contains("foo__bar"), "unexpected mangling in {s:?}");
        assert!(s.contains("foo_bar_baz"), "unexpected mangling in {s:?}");
        assert!(s.contains("a*b*c"), "unexpected mangling in {s:?}");

        // Strikethrough, link, backslash escape, table pipe, HR
        let mut st2 = MarkdownStreamer::new(true, true);
        let mut out2 = Vec::new();
        st2.write("~~old~~ [docs](https://example.com) \\*lit\\*\n---\n| a | b |\n", &mut out2)
            .unwrap();
        st2.finish(&mut out2).unwrap();
        let s2 = String::from_utf8(out2).unwrap();
        assert!(s2.contains("\x1b[9mold\x1b[0m"));
        assert!(s2.contains("docs") && s2.contains("https://example.com"));
        assert!(s2.contains("*lit*"));
        assert!(s2.contains("────────"));
        assert!(s2.contains("│"));

        // 4-backtick outer fence preserves inner triple backticks, no blank line before └──
        let mut st3 = MarkdownStreamer::new(true, true);
        let mut out3 = Vec::new();
        st3.write("````md\n```rust\nlet x = 1;\n```\n````\n", &mut out3)
            .unwrap();
        st3.finish(&mut out3).unwrap();
        let s3 = String::from_utf8(out3).unwrap();
        assert!(s3.contains("```rust"));
        assert!(s3.contains("let x = 1;"));
        assert!(!s3.contains("\n\n  \x1b[90m└──"));

        // Blank line resets unclosed italic/bold state
        let mut st4 = MarkdownStreamer::new(true, true);
        let mut out4 = Vec::new();
        st4.write("**unclosed bold\n\nNext paragraph", &mut out4).unwrap();
        st4.finish(&mut out4).unwrap();
        let s4 = String::from_utf8(out4).unwrap();
        assert!(s4.contains("\x1b[0m\nNext paragraph"), "got: {s4:?}");
    }

    #[test]
    fn test_parse_token_limit_kv_and_spec() {
        assert_eq!(parse_token_limit("1").unwrap(), 1);
        assert_eq!(parse_token_limit("256").unwrap(), 256);
        assert_eq!(parse_token_limit("32k").unwrap(), 32768);
        assert_eq!(parse_token_limit("1m").unwrap(), 1024 * 1024);
        assert!(parse_token_limit("0").is_err());

        assert_eq!(parse_kv_type("q8").unwrap(), "q8_0");
        assert_eq!(parse_kv_type("Q8_0").unwrap(), "q8_0");
        assert_eq!(parse_kv_type("q4").unwrap(), "q4_0");
        assert_eq!(parse_kv_type("bf16").unwrap(), "bf16");
        assert_eq!(parse_kv_type("f16").unwrap(), "f16");
        assert!(parse_kv_type("bogus").is_err());

        assert_eq!(parse_spec_type("off").unwrap(), "none");
        assert_eq!(parse_spec_type("on").unwrap(), "ngram-cache");
        assert_eq!(parse_spec_type("ngram-simple").unwrap(), "ngram-simple");
        assert_eq!(parse_spec_type("ngram-mod").unwrap(), "ngram-mod");
        assert!(parse_spec_type("nope").is_err());
    }

    #[test]
    fn test_build_xray_outbound_ws_grpc_and_tls_params() {
        let ws_node = SubNode {
            id: 1,
            tag: "DE-WS".to_string(),
            proto: "vless".to_string(),
            raw: "vless://11111111-1111-1111-1111-111111111111@1.2.3.4:443?type=ws&path=%2Fws&host=cdn.test&security=tls&sni=sni.test&alpn=h2%2Chttp%2F1.1&allowInsecure=1#DE".to_string(),
        };
        let ws_out = build_xray_outbound(&ws_node).unwrap();
        assert_eq!(ws_out["streamSettings"]["network"], "ws");
        assert_eq!(ws_out["streamSettings"]["wsSettings"]["path"], "/ws");
        assert_eq!(ws_out["streamSettings"]["wsSettings"]["headers"]["Host"], "cdn.test");
        assert_eq!(ws_out["streamSettings"]["tlsSettings"]["allowInsecure"], true);
        assert_eq!(ws_out["streamSettings"]["tlsSettings"]["alpn"][0], "h2");

        let grpc_node = SubNode {
            id: 2,
            tag: "NL-gRPC".to_string(),
            proto: "trojan".to_string(),
            raw: "trojan://secretpass@5.6.7.8:443?type=grpc&serviceName=tunSvc&security=tls&sni=g.test#NL".to_string(),
        };
        let grpc_out = build_xray_outbound(&grpc_node).unwrap();
        assert_eq!(grpc_out["protocol"], "trojan");
        assert_eq!(grpc_out["streamSettings"]["network"], "grpc");
        assert_eq!(grpc_out["streamSettings"]["grpcSettings"]["serviceName"], "tunSvc");
    }

    #[test]
    fn test_tac_no_bloat_condenses_tools_and_cyrillic_chars() {
        let mut short_msgs = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hello"}),
        ];
        assert!(compress_task_aware(&mut short_msgs, CompressionMode::Heavy, Path::new(".")).is_none());

        let big_tool = "x".repeat(2000);
        let mut tool_msgs = vec![
            json!({"role": "user", "content": "check"}),
            json!({"role": "tool", "name": "read_file", "content": big_tool}),
        ];
        let res = compress_task_aware(&mut tool_msgs, CompressionMode::Heavy, Path::new(".")).unwrap();
        assert!(res.after_tokens < res.before_tokens);
        assert!(tool_msgs[1]["content"].as_str().unwrap().len() < 600);

        // 100 Cyrillic chars (>200 bytes UTF-8, <160 chars) must not be dropped from Key State
        let cyr_decision = "Проверено: сервер запущен на локальном порту и отвечает корректно без ошибок ".to_string()
            + &"А".repeat(30);
        assert!(cyr_decision.len() > 160);
        assert!(cyr_decision.chars().count() < 160);
        let mut long_msgs = vec![
            json!({"role": "user", "content": "Запусти и проверь"}),
            json!({"role": "assistant", "content": cyr_decision.clone()}),
            json!({"role": "user", "content": "шаг 2 ".repeat(80)}),
            json!({"role": "assistant", "content": "готово 2 ".repeat(80)}),
            json!({"role": "user", "content": "последний вопрос"}),
            json!({"role": "assistant", "content": "ответ"}),
        ];
        let res2 = compress_task_aware(&mut long_msgs, CompressionMode::Heavy, Path::new(".")).unwrap();
        assert!(res2.after_tokens < res2.before_tokens);
        let summary = long_msgs[0]["content"].as_str().unwrap();
        assert!(summary.contains("Проверено: сервер запущен"));
    }

    #[test]
    fn test_validator_and_code_block_extraction() {
        assert!(!is_multiline_incomplete(r"/cwd C:\Users\Дима\"));
        assert!(is_multiline_incomplete("first line \\"));
        assert!(is_multiline_incomplete("```rust\nfn x() {}"));
        assert!(!is_multiline_incomplete("```rust\nfn x() {}\n```"));
        assert!(is_multiline_incomplete("text \"\"\"open"));
        assert!(!is_multiline_incomplete("text \"\"\"closed\"\"\""));

        assert_eq!(normalize_user_prompt(r"/cwd C:\Users\"), r"/cwd C:\Users\");
        assert_eq!(normalize_user_prompt("hello \\\r\nworld"), "hello \nworld");

        let reply = "text\n```python\nprint(1)\n```\nmore\n```rust\nlet a = 42;\n```\nend";
        assert_eq!(extract_last_code_block(reply).as_deref(), Some("let a = 42;"));
    }

    #[tokio::test]
    async fn test_tools_read_edit_search_and_guardrails() {
        let dir = std::env::temp_dir().join("bonsai_unit_tools_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(".git")).unwrap();

        let file = dir.join("sample.ps1");
        std::fs::write(&file, b"\xEF\xBB\xBFline1\r\nWrite-Host '\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82'\r\nline3\r\nline3\r\n").unwrap();
        std::fs::write(dir.join(".git").join("ignored.txt"), b"Write-Host secret_hit").unwrap();

        let cli = Cli::parse_from(["bonsai", "--bypass"]);
        let mut app = App::new(cli, false).unwrap();
        app.cwd = dir.clone();

        // read_file line range + UTF-8 BOM decoded cleanly
        let read_out = app
            .tool_read_file(&json!({"path": "sample.ps1", "start_line": 2, "end_line": 3}))
            .await;
        assert!(read_out.contains("2: Write-Host 'Привет'"), "got: {read_out}");
        assert!(!read_out.contains("1: line1"));

        // edit_file rejects ambiguous multiple matches unless replace_all=true
        let dup_err = app
            .tool_edit_file(&json!({"path": "sample.ps1", "target": "line3", "replacement": "lineX"}))
            .await;
        assert!(dup_err.contains("matched 2 occurrences"), "got: {dup_err}");

        // edit_file single replacement preserves CRLF & PowerShell BOM
        let ok_edit = app
            .tool_edit_file(&json!({"path": "sample.ps1", "target": "Write-Host 'Привет'", "replacement": "Write-Host 'Мир'"}))
            .await;
        assert!(ok_edit.contains("1 replacement"), "got: {ok_edit}");
        let disk_bytes = std::fs::read(&file).unwrap();
        assert!(disk_bytes.starts_with(b"\xEF\xBB\xBF"), "BOM must be kept for .ps1");
        let disk_str = String::from_utf8_lossy(&disk_bytes[3..]);
        assert!(disk_str.contains("\r\nWrite-Host 'Мир'\r\n"), "CRLF preserved: {disk_str:?}");

        // search_files skips .git and finds edits (case-insensitive Cyrillic)
        let search_out = app.tool_search_files(&json!({"pattern": "миР"}));
        assert!(search_out.contains("sample.ps1:2:"), "got: {search_out}");
        assert!(!search_out.contains("ignored.txt"), "must skip .git: {search_out}");

        let ignored_search = app.tool_search_files(&json!({"pattern": "secret_hit"}));
        assert!(ignored_search.contains("no matches"), "got: {ignored_search}");

        // guardrail blocks overwriting proxy shim or leaking proxy password
        app.proxy = Some("http://alice:TopSecret99@127.0.0.1:8888".to_string());
        let leak_err = app
            .tool_edit_file(&json!({"path": "sample.ps1", "target": "line3", "replacement": "TopSecret99", "replace_all": true}))
            .await;
        assert!(leak_err.contains("proxy credentials"), "got: {leak_err}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
