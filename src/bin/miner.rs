use anyhow::Result;
use clap::Parser;
use std::sync::{Arc, RwLock};
use std::sync::atomic::AtomicU64;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(name = "miner")]
#[command(about = "Standalone Midstate GPU/CPU Stratum miner")]
struct Cli {
    /// Stratum endpoint, e.g. stratum+tcp://pool.example.com:3333
    #[arg(long = "pool-url")]
    pool_url: String,

    /// MSS payout address for mining rewards
    #[arg(long = "address", alias = "payout-address")]
    address: String,

    /// Optional rig name reported to the pool
    #[arg(long, default_value = "default")]
    worker: String,

    /// Optional explicit audit API URL, e.g. http://pool.example.com:8081
    #[arg(long = "audit-url")]
    audit_url: Option<String>,

    /// Mining backend: auto, cuda, gpu, or cpu
    #[arg(long, default_value = "auto")]
    backend: String,

    /// CPU worker threads if CPU fallback is used. 0 = use all available cores.
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Optional GPU duty cycle between 0.02 and 1.0
    #[arg(long = "gpu-duty")]
    gpu_duty: Option<f32>,

    /// Print miner dashboard automatically every N seconds. 0 = only on ENTER.
    #[arg(long = "stats-interval", default_value_t = 0)]
    stats_interval: u64,

    /// Hide normal INFO logs so the status dashboard stays clean.
    #[arg(long = "dashboard-only", alias = "quiet", default_value_t = false)]
    dashboard_only: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let default_filter = if cli.dashboard_only {
        "midstate=warn,tower_http=warn"
    } else {
        "midstate=info,tower_http=debug"
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| default_filter.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    midstate::core::gpu_mining::set_backend(match cli.backend.to_ascii_lowercase().as_str() {
        "cuda" => midstate::core::gpu_mining::Backend::Cuda,
        "gpu" => midstate::core::gpu_mining::Backend::Gpu,
        "cpu" => midstate::core::gpu_mining::Backend::Cpu,
        "auto" => midstate::core::gpu_mining::Backend::Auto,
        other => {
            tracing::warn!("unknown --backend '{other}', using auto");
            midstate::core::gpu_mining::Backend::Auto
        }
    });

    if let Some(duty) = cli.gpu_duty {
        midstate::core::gpu_mining::set_gpu_duty(duty);
    }

    let hash_counter = Arc::new(AtomicU64::new(0));
    let stats = Arc::new(RwLock::new(midstate::mining::StratumStats::default()));

    midstate::mining::spawn_stratum_dashboard(
        hash_counter.clone(),
        stats.clone(),
        (cli.stats_interval > 0).then_some(cli.stats_interval),
    );

    tracing::info!(
        "starting standalone miner (backend: {}, threads: {})",
        cli.backend,
        if cli.threads == 0 {
            "max".to_string()
        } else {
            cli.threads.to_string()
        }
    );
    if cli.stats_interval > 0 {
        tracing::info!(
            "printing dashboard every {}s; press [ENTER] to view immediately",
            cli.stats_interval
        );
    } else {
        tracing::info!("press [ENTER] at any time to view dashboard");
    }

    midstate::mining::run_stratum_client_with_options(
        midstate::mining::StratumClientOptions {
            pool_url: cli.pool_url,
            payout_address: cli.address,
            worker: cli.worker,
            audit_url: cli.audit_url,
        },
        cli.threads,
        hash_counter,
        stats,
    )
    .await;

    Ok(())
}
