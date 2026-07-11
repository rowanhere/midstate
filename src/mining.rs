//! Mining coordination for the full node.
//!
//! This module was extracted from the `node` god module during the refactor.
//! It owns the types for miner config, the result of a mining attempt (block or pool share),
//! and helpers for coinbase generation and logging.
//!
//! The CPU-intensive extension mining itself lives in `crate::core::extension`.
//! Template building for *external* miners (RPC/light) remains in `node.rs` as
//! `build_block_template_inner` (due to its dependency on `rpc::types`).

use crate::core::types::*;
use crate::core::{block_reward, decompose_value, CoinbaseOutput};
use crate::wallet::{coinbase_seed, coinbase_salt};
use crate::core::wots;
use std::path::PathBuf;
use rayon::prelude::*;
use tokio::net::TcpStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Default, Clone)]
pub struct StratumStats {
    pub network_target: [u8; 32],
    pub accepted_shares: u64,
    pub rejected_shares: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MinerToml {
    pub mining: MiningConfig,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MiningConfig {
    pub mode: String,
    pub pool_url: Option<String>,
    pub payout_address: Option<String>,
    pub pool_address: Option<String>,
    /// Optional rig name reported to the pool in mining.authorize for the per-worker
    /// breakdown. Absent in older miner.toml files -> deserializes to None -> "default".
    pub worker: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StratumClientOptions {
    pub pool_url: String,
    pub payout_address: String,
    pub worker: String,
    pub audit_url: Option<String>,
}

pub enum MinedResult {
    Block(crate::core::Batch),
    Share {
        batch: crate::core::Batch,
        pool_url: String,
        payout_address: String,
    },
}

/// Generate coinbase outputs for a new block being mined.
///
/// In solo mode: uses the node's mining seed to derive WOTS keys/addresses/salts.
/// In pool mode: pays the pool's MSS address and watermarks the miner's payout
/// address into the salt (so the pool can later prove who earned the share).
pub fn generate_coinbase(
    mining_seed: &[u8; 32],
    height: u64,
    total_fees: u64,
    pool_target: Option<([u8; 32], [u8; 32])>, // (Pool MSS Address, Miner Payout Address)
) -> Vec<CoinbaseOutput> {
    let reward = block_reward(height);
    let total_value = reward.saturating_add(total_fees);
    let denominations = decompose_value(total_value);

    denominations.into_par_iter()
        .enumerate()
        .map(move |(i, value)| {
            match pool_target {
                Some((pool_addr, miner_addr)) => {
                    // POOL MINING MODE
                    // Pay the pool's address, but embed the miner's address in the salt
                    // so the pool can cryptographically verify who did the work.
                    let mut salt = [0u8; 32];
                    let mut hasher = blake3::Hasher::new();
                    hasher.update(b"pool_share");
                    hasher.update(&miner_addr);
                    hasher.update(&height.to_le_bytes());
                    hasher.update(&(i as u64).to_le_bytes());
                    salt.copy_from_slice(hasher.finalize().as_bytes());

                    CoinbaseOutput { address: pool_addr, value, salt }
                }
                None => {
                    // SOLO MINING MODE (Original Logic)
                    let seed = coinbase_seed(mining_seed, height, i as u64);
                    let owner_pk = wots::keygen(&seed);
                    let address = compute_address(&owner_pk);
                    let salt = coinbase_salt(mining_seed, height, i as u64);
                    
                    CoinbaseOutput { address, value, salt }
                }
            }
        })
        .collect()
}

/// Append a JSONL entry for every coinbase output created at this height.
/// The seed itself is deliberately NOT logged (it is derivable from the
/// node's persistent mining_seed + height + index).
pub fn log_coinbase(
    mining_seed: &[u8; 32],
    data_dir: &PathBuf,
    height: u64,
    total_fees: u64,
) {
    let reward = block_reward(height);
    let total_value = reward + total_fees;
    let denominations = decompose_value(total_value);
    let log_path = data_dir.join("coinbase_seeds.jsonl");

    let entries: Vec<String> = denominations.into_par_iter()
        .enumerate()
        .map(move |(i, value)| {
            let seed = coinbase_seed(mining_seed, height, i as u64);
            let owner_pk = wots::keygen(&seed);
            let address = compute_address(&owner_pk);
            let salt = coinbase_salt(mining_seed, height, i as u64);
            let coin_id = compute_coin_id(&address, value, &salt);
            // NOTE: We intentionally do NOT log the seed (private key).
            // It is derivable from (mining_seed, height, index) when
            // the wallet needs to spend. Logging it in cleartext would
            // allow anyone with filesystem or RPC access to steal funds.
            format!(
                r#"{{"height":{},"index":{},"address":"{}","coin":"{}","value":{},"salt":"{}"}}"#,
                height, i,
                hex::encode(address),
                hex::encode(coin_id),
                value,
                hex::encode(salt)
            )
        })
        .collect();

    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true).append(true).open(&log_path)
    {
        use std::io::Write;
        for entry in entries {
            let _ = writeln!(file, "{}", entry);
        }
    }
}

/// Lightweight coordinator for the node's autonomous mining.
/// Holds the configuration that is stable across mining attempts.
/// The heavy per-block preparation + delegation to core::extension::mine_extension
/// is still driven from Node (to keep access to live mempool/state), but
/// this gives the mining logic a named home outside the god module.
#[derive(Clone)]
pub struct MiningCoordinator {
    pub threads: Option<usize>,
    seed: [u8; 32],
    data_dir: PathBuf,
}

impl MiningCoordinator {
    pub fn new(threads: Option<usize>, seed: [u8; 32], data_dir: PathBuf) -> Self {
        Self {
            threads,
            seed,
            data_dir,
        }
    }

    pub fn seed(&self) -> &[u8; 32] {
        &self.seed
    }

    pub fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    /// Convenience wrapper around the free function, using the coordinator's seed.
    pub fn generate_coinbase(
        &self,
        height: u64,
        total_fees: u64,
        pool_target: Option<([u8; 32], [u8; 32])>,
    ) -> Vec<CoinbaseOutput> {
        generate_coinbase(&self.seed, height, total_fees, pool_target)
    }

    /// Convenience wrapper for logging using the coordinator's seed + data_dir.
    pub fn log_coinbase(&self, height: u64, total_fees: u64) {
        log_coinbase(&self.seed, &self.data_dir, height, total_fees);
    }
}

fn normalize_pool_host(pool_url: &str) -> String {
    pool_url
        .trim()
        .trim_end_matches('/')
        .strip_prefix("stratum+tcp://")
        .or_else(|| pool_url.trim().trim_end_matches('/').strip_prefix("tcp://"))
        .or_else(|| pool_url.trim().trim_end_matches('/').strip_prefix("http://"))
        .or_else(|| pool_url.trim().trim_end_matches('/').strip_prefix("https://"))
        .unwrap_or_else(|| pool_url.trim().trim_end_matches('/'))
        .to_string()
}

fn default_audit_base_url(host: &str) -> String {
    let mut api_host = host.to_string();
    if let Some((ip, port_str)) = host.rsplit_once(':') {
        if let Ok(port) = port_str.parse::<u16>() {
            let offset = port.saturating_sub(3333);
            let api_port = 8081 + offset;
            api_host = format!("{}:{}", ip, api_port);
        }
    }
    format!("http://{}", api_host)
}

fn normalize_audit_base_url(audit_url: &str) -> String {
    let trimmed = audit_url.trim().trim_end_matches('/');
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{}", trimmed)
    }
}

fn default_share_target() -> [u8; 32] {
    let mut target = [0xff; 32];
    target[0] = 0x00;
    target[1] = 0x0f;
    target
}

fn decode_hex32(s: &str) -> Option<[u8; 32]> {
    let mut out = [0u8; 32];
    hex::decode_to_slice(s, &mut out).ok()?;
    Some(out)
}

async fn fetch_pool_targets(
    client: &reqwest::Client,
    audit_base_url: &str,
) -> Option<([u8; 32], [u8; 32])> {
    let stats = client
        .get(format!("{}/pool/stats", audit_base_url))
        .send()
        .await
        .ok()?
        .json::<serde_json::Value>()
        .await
        .ok()?;
    let network_target = stats
        .get("network_target")
        .and_then(|v| v.as_str())
        .and_then(decode_hex32)?;
    let share_target = stats
        .get("share_target")
        .and_then(|v| v.as_str())
        .and_then(decode_hex32)?;
    Some((network_target, share_target))
}

pub fn spawn_stratum_dashboard(
    hash_counter: Arc<AtomicU64>,
    stats: Arc<std::sync::RwLock<StratumStats>>,
    stats_interval_secs: Option<u64>,
) {
    let hc = hash_counter.clone();
    let stats_clone = stats.clone();

    tokio::spawn(async move {
        use tokio::io::AsyncBufReadExt;

        let mut lines = tokio::io::BufReader::new(tokio::io::stdin()).lines();
        let mut ticker = stats_interval_secs
            .filter(|s| *s > 0)
            .map(|s| tokio::time::interval(std::time::Duration::from_secs(s)));
        if let Some(ticker) = ticker.as_mut() {
            ticker.tick().await;
        }
        let mut last_hashes = 0;
        let mut last_time = Instant::now();
        let mut last_cuda_nonces: std::collections::BTreeMap<i32, u64> =
            std::collections::BTreeMap::new();

        // Hardcoded share target matching the current pool server (0x000f...).
        let share_target = default_share_target();

        fn u256_to_f64(u: primitive_types::U256) -> f64 {
            u.0[0] as f64
                + (u.0[1] as f64) * 2.0f64.powi(64)
                + (u.0[2] as f64) * 2.0f64.powi(128)
                + (u.0[3] as f64) * 2.0f64.powi(192)
        }

        fn format_time(secs: f64) -> String {
            if secs < 60.0 {
                return format!("{:.0}s", secs);
            }
            if secs < 3600.0 {
                return format!("{:.0}m {:.0}s", secs / 60.0, secs % 60.0);
            }
            if secs < 86400.0 {
                return format!("{:.0}h {:.0}m", secs / 3600.0, (secs % 3600.0) / 60.0);
            }
            if secs < 31536000.0 {
                return format!("{:.0}d {:.0}h", secs / 86400.0, (secs % 86400.0) / 3600.0);
            }
            format!("{:.1} years", secs / 31536000.0)
        }

        fn format_hashrate(rate: f64) -> String {
            let units = ["H/s", "kH/s", "MH/s", "GH/s", "TH/s", "PH/s"];
            let mut value = rate.max(0.0);
            let mut unit = 0usize;
            while value >= 1000.0 && unit + 1 < units.len() {
                value /= 1000.0;
                unit += 1;
            }
            if value >= 100.0 {
                format!("{value:.0} {}", units[unit])
            } else if value >= 10.0 {
                format!("{value:.1} {}", units[unit])
            } else {
                format!("{value:.2} {}", units[unit])
            }
        }

        let share_target_f64 = u256_to_f64(primitive_types::U256::from_big_endian(&share_target));

        loop {
            let mut manual = false;
            if let Some(ticker) = ticker.as_mut() {
                tokio::select! {
                    line = lines.next_line() => {
                        match line {
                            Ok(Some(_)) => manual = true,
                            Ok(None) => break,
                            Err(_) => break,
                        }
                    }
                    _ = ticker.tick() => {}
                }
            } else if lines.next_line().await.unwrap_or(None).is_none() {
                break;
            }

            let current = hc.load(Ordering::Relaxed);
            let now = Instant::now();
            let elapsed = now.duration_since(last_time).as_secs_f64();
            let rate = if elapsed > 0.0 {
                (current - last_hashes) as f64 / elapsed
            } else {
                0.0
            };
            let total_rate = format_hashrate(rate);
            let mut cuda_lines = Vec::new();
            #[cfg(not(target_arch = "wasm32"))]
            {
                let cuda = crate::core::cuda_mining::cuda_dashboard_snapshot();
                if !cuda.is_empty() {
                    for gpu in cuda {
                        let previous = last_cuda_nonces
                            .insert(gpu.ordinal, gpu.nonce_equivalents)
                            .unwrap_or(gpu.nonce_equivalents);
                        let gpu_rate = if elapsed > 0.0 {
                            gpu.nonce_equivalents.saturating_sub(previous) as f64 / elapsed
                        } else {
                            0.0
                        };
                        let gpu_rate_fmt = format_hashrate(gpu_rate);
                        cuda_lines.push(format!(
                            "  GPU {:>2}: {:>10} | shares {} | blocks {} | {}",
                            gpu.ordinal,
                            gpu_rate_fmt,
                            gpu.accepted_shares,
                            gpu.accepted_blocks,
                            gpu.name
                        ));
                    }
                }
            }

            let s = stats_clone.read().unwrap().clone();
            if !manual && current == last_hashes && s.network_target == [0u8; 32] {
                last_time = now;
                continue;
            }

            if s.network_target != [0u8; 32] {
                let target_f64 = u256_to_f64(primitive_types::U256::from_big_endian(&s.network_target));

                let expected_nonces = 2.0f64.powi(256) / target_f64.max(1.0);
                let network_nps = expected_nonces / 60.0;
                let share_pct = if network_nps > 0.0 {
                    (rate / network_nps) * 100.0
                } else {
                    0.0
                };

                let expected_shares_per_block = share_target_f64 / target_f64.max(1.0);
                let session_effort_pct = if expected_shares_per_block > 0.0 {
                    (s.accepted_shares as f64 / expected_shares_per_block) * 100.0
                } else {
                    0.0
                };

                println!("\n== MINER STATUS ==");
                println!("Hashrate:      {}", total_rate);
                if !cuda_lines.is_empty() {
                    println!("CUDA GPUs:");
                    for line in &cuda_lines {
                        println!("{line}");
                    }
                }
                println!("Network:       {}", format_hashrate(network_nps));

                if share_pct < 0.001 {
                    println!("Your Share:    < 0.001%");
                } else {
                    println!("Your Share:    {:.4}%", share_pct);
                }

                if rate > 0.0 {
                    println!("Solo ETA:      {}", format_time(expected_nonces / rate));
                } else {
                    println!("Solo ETA:      ---");
                }
                println!("----------------------------------------");
                println!("Shares:        {} acc / {} rej", s.accepted_shares, s.rejected_shares);
                println!("Expected:      1 block per {} shares", expected_shares_per_block.round() as u64);
                println!(
                    "Session Luck:  {:.2}% {}",
                    session_effort_pct,
                    if session_effort_pct >= 100.0 {
                        "due for a block"
                    } else {
                        "in progress"
                    }
                );
            } else {
                println!("\n== MINER STATUS ==");
                println!("Hashrate:      {}", total_rate);
                if !cuda_lines.is_empty() {
                    println!("CUDA GPUs:");
                    for line in &cuda_lines {
                        println!("{line}");
                    }
                }
                println!("Network:       Waiting for job...");
            }

            println!("========================================\n");

            last_hashes = current;
            last_time = now;
        }
    });
}

pub async fn run_stratum_client(
    pool_url: String,
    payout_address: String,
    worker: String,
    threads: usize,
    hash_counter: Arc<AtomicU64>,
    stats: Arc<std::sync::RwLock<StratumStats>>,
) {
    run_stratum_client_with_options(
        StratumClientOptions {
            pool_url,
            payout_address,
            worker,
            audit_url: None,
        },
        threads,
        hash_counter,
        stats,
    )
    .await;
}

pub async fn run_stratum_client_with_options(
    opts: StratumClientOptions,
    threads: usize,
    hash_counter: Arc<AtomicU64>,
    stats: Arc<std::sync::RwLock<StratumStats>>,
) {
    let host = normalize_pool_host(&opts.pool_url);
    let audit_base_url = opts
        .audit_url
        .as_deref()
        .map(normalize_audit_base_url)
        .unwrap_or_else(|| default_audit_base_url(&host));

    let http_client = reqwest::Client::new();

    // Declare the mutable cancel flag outside so it persists across reconnects
    let mut mining_cancel = Arc::new(AtomicBool::new(false));

    loop {
        // Kill any lingering threads before attempting to reconnect
        mining_cancel.store(true, Ordering::Relaxed);

        tracing::info!(
            "stratum client connecting to {} (audit: {})",
            host,
            audit_base_url
        );
        if let Ok(mut stream) = TcpStream::connect(&host).await {
            let (read_half, mut write_half) = stream.split();
            let mut reader = BufReader::new(read_half);

            let auth_req = serde_json::json!({
                "id": 1,
                "method": "mining.authorize",
                "params": [opts.payout_address.clone(), opts.worker.clone()]
            });
            let _ = write_half.write_all(format!("{}\n", auth_req).as_bytes()).await;

            let mut line = String::new();
            let current_job_id = Arc::new(std::sync::RwLock::new(0u64));
            
            // Create a fresh channel for shares on each successful connection
            let (share_tx, mut share_rx) = tokio::sync::mpsc::channel::<(u64, u64)>(100);

            loop {
                tokio::select! {
                    res = reader.read_line(&mut line) => {
                        if res.unwrap_or(0) == 0 { break; } 
                        let msg: serde_json::Value = serde_json::from_str(&line).unwrap_or_default();
                        line.clear();

                        if msg["method"] == "mining.notify" {
                            let params = msg["params"].as_array().unwrap();
                            let job_id = params[0].as_u64().unwrap();
                            let hash_hex = params[1].as_str().unwrap();
                            let template_val = &params[2];

                            let mut m_hash = [0u8; 32];
                            hex::decode_to_slice(hash_hex, &mut m_hash).unwrap();

                            let batch: crate::core::Batch = match serde_json::from_value(template_val.clone()) {
                                Ok(b) => b,
                                Err(e) => {
                                    tracing::error!("failed to parse batch template: {}", e);
                                    break;
                                }
                            };
                            let header = batch.header();
                            let mut n_target = params
                                .get(3)
                                .and_then(|v| v.as_str())
                                .and_then(decode_hex32)
                                .unwrap_or(batch.target);
                            let mut s_target = params
                                .get(4)
                                .and_then(|v| v.as_str())
                                .and_then(decode_hex32)
                                .unwrap_or_else(default_share_target);

                            if params.get(3).is_none() || params.get(4).is_none() {
                                let Some((network_target, share_target)) =
                                    fetch_pool_targets(&http_client, &audit_base_url).await
                                else {
                                    tracing::error!(
                                        "pool notify omitted explicit targets and audit stats were unavailable; refusing unsafe fallback target"
                                    );
                                    break;
                                };
                                n_target = network_target;
                                s_target = share_target;
                            }

                            let dashboard_target = if n_target >= s_target {
                                tracing::warn!(
                                    "pool network target {} is easier than share target {}; mining pool shares only",
                                    hex::encode(n_target),
                                    hex::encode(s_target)
                                );
                                [0u8; 32]
                            } else {
                                n_target
                            };
                            {
                                let mut s = stats.write().unwrap();
                                s.network_target = dashboard_target;
                            }
                            tracing::info!(
                                "job {} targets: share={} network={} mode=share-only",
                                job_id,
                                hex::encode(s_target),
                                hex::encode(n_target)
                            );
                            let calculated_hash = crate::core::types::compute_header_hash(&header);
                            
                            if calculated_hash != m_hash {
                                tracing::error!("audit failed: template header hash mismatch. disconnecting.");
                                break;
                            }

                            if let Some(pool_cb) = batch.coinbase.first() {
                                let claimed_root = hex::encode(pool_cb.salt);
                                
                                if let Ok(res) = http_client
                                    .get(format!(
                                        "{}/api/proof?address={}",
                                        audit_base_url, opts.payout_address
                                    ))
                                    .send()
                                    .await
                                {
                                    if let Ok(proof_data) = res.json::<serde_json::Value>().await {
                                        if proof_data.get("error").is_none() {
                                            
                                            let payout_bytes = crate::core::types::parse_address_flexible(&opts.payout_address).unwrap();
                                            let score = proof_data["score"].as_u64().unwrap_or(0);
                                            let mut data = [0u8; 40];
                                            data[0..32].copy_from_slice(&payout_bytes);
                                            data[32..40].copy_from_slice(&score.to_le_bytes());
                                            let mut current_hash = crate::core::types::hash(&data);

                                            if let (Some(proof_array), Some(mut current_idx)) = (proof_data["proof"].as_array(), proof_data["index"].as_u64()) {
                                                for sibling_hex in proof_array {
                                                    let mut sibling = [0u8; 32];
                                                    hex::decode_to_slice(sibling_hex.as_str().unwrap(), &mut sibling).unwrap();
                                                    if current_idx % 2 == 1 {
                                                        current_hash = crate::core::types::hash_concat(&sibling, &current_hash);
                                                    } else {
                                                        current_hash = crate::core::types::hash_concat(&current_hash, &sibling);
                                                    }
                                                    current_idx /= 2;
                                                }
                                            }

                                            if hex::encode(current_hash) != claimed_root {
                                                tracing::error!("audit failed: merkle root mismatch (computed {}, claimed {}). disconnecting.", hex::encode(current_hash), claimed_root);
                                                break;
                                            }
                                            
                                            if score > 0 {
                                                let mut found_payout = false;
                                                for cb in &batch.coinbase {
                                                    if cb.address == payout_bytes {
                                                        found_payout = true;
                                                        break;
                                                    }
                                                }
                                                if !found_payout {
                                                    tracing::error!("audit failed: omitted from payout array despite score of {}. disconnecting.", score);
                                                    break;
                                                }
                                            }
                                        } else if let Some(err) = proof_data.get("error").and_then(|v| v.as_str()) {
                                            tracing::warn!("audit proof unavailable: {}", err);
                                        }
                                    }
                                } else {
                                    tracing::warn!("failed to fetch audit proof from {}", audit_base_url);
                                }
                            }
                           tracing::info!("audit passed. starting job {}", job_id);
                            
                            // Cancel the PREVIOUS thread pool
                            mining_cancel.store(true, Ordering::Relaxed);
                            *current_job_id.write().unwrap() = job_id;
                            
                            let new_cancel = Arc::new(AtomicBool::new(false));
                            
                            // Store the new reference so we can cancel it next time!
                            mining_cancel = new_cancel.clone(); 
                            
                            let share_tx_clone = share_tx.clone();
                            let j_id = job_id;
                            let nc = new_cancel.clone();
                            let hc = hash_counter.clone();
                            let share_only_target = [0u8; 32];
                            
                            std::thread::spawn(move || {
                                loop {
                                    if nc.load(Ordering::Relaxed) { break; }
                                    
                                    if let Some(res) = crate::core::gpu_mining::mine(
                                        m_hash, share_only_target, Some(s_target), threads, nc.clone(), hc.clone()
                                    ) {
                                        let nonce = match res {
                                            crate::core::extension::MiningResult::Block(ext) => ext.nonce,
                                            crate::core::extension::MiningResult::Share(ext) => ext.nonce,
                                        };
                                        let _ = share_tx_clone.blocking_send((j_id, nonce));
                                    } else {
                                        break; 
                                    }
                                }
                            });
                        } else if msg["result"].as_bool() == Some(true) && msg["id"].as_u64() == Some(2) {
                            tracing::debug!("share accepted");
                            stats.write().unwrap().accepted_shares += 1; 
                        } else if let Some(err) = msg["error"].as_str() {
                            if err == "Low difficulty" {
                                tracing::debug!("share rejected: {}", err);
                            } else {
                                tracing::warn!("share rejected: {}", err);
                            }
                            stats.write().unwrap().rejected_shares += 1; 
                        }
                    }
                    Some((job_id, nonce)) = share_rx.recv() => {
                        let submit_req = serde_json::json!({
                            "id": 2,
                            "method": "mining.submit",
                            "params": [opts.payout_address.clone(), job_id, nonce]
                        });
                        let _ = write_half.write_all(format!("{}\n", submit_req).as_bytes()).await;
                    }
                }
            }
        }
        tracing::warn!("disconnected from stratum pool. reconnecting in 5s...");
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    }
}
