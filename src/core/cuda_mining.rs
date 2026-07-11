//! Native NVIDIA CUDA mining backend for the PoW extension chain.
//!
//! This backend bypasses Vulkan/wgpu completely. It dynamically loads the CUDA
//! Driver API and NVRTC at runtime, compiles a tiny CUDA C kernel, self-tests it
//! against the CPU consensus implementation, and mines across all visible CUDA
//! devices. The GPU only proposes candidate nonces; every candidate is
//! recomputed on CPU before being returned to the pool/node.

use super::extension::{create_extension, mine_extension, MiningResult};
use super::types::{Extension, EXTENSION_ITERATIONS};
use anyhow::{anyhow, bail, Result};
use libloading::Library;
use std::collections::VecDeque;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread;
use std::time::Instant;

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuStream = *mut c_void;
type CuDevicePtr = u64;
type NvrtcResult = c_int;
type NvrtcProgram = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const NVRTC_SUCCESS: NvrtcResult = 0;
const MAX_WINNERS: u32 = 256;
const WINNERS_WORDS: usize = 4 + (MAX_WINNERS as usize) * 3;
const WINNERS_BYTES: usize = WINNERS_WORDS * 4;
const SELFTEST_N: u32 = 8;
const THREADS_PER_BLOCK: u32 = 64;
const DEFAULT_BATCH_NONCES: u32 = 1 << 13;
const DEFAULT_ITERS_PER_DISPATCH: u32 = 2_000;
const DEFAULT_RESPONSIVE_ITERS: u32 = 384;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    midstate: [u32; 8],
    target: [u32; 8],
    pool: [u32; 8],
    base_lo: u32,
    base_hi: u32,
    n_nonces: u32,
    iters: u32,
    has_pool: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

type JobKey = ([u8; 32], [u8; 32], Option<[u8; 32]>);

struct MinerState {
    job: Option<JobKey>,
    pending: VecDeque<MiningResult>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
struct CudaSettings {
    batch_nonces: u32,
    iters_per_dispatch: u32,
    responsive_iters: u32,
    duty: f32,
}

impl Default for CudaSettings {
    fn default() -> Self {
        Self {
            batch_nonces: DEFAULT_BATCH_NONCES,
            iters_per_dispatch: DEFAULT_ITERS_PER_DISPATCH,
            responsive_iters: DEFAULT_RESPONSIVE_ITERS,
            duty: 1.0,
        }
    }
}

impl CudaSettings {
    fn sanitized(mut self) -> Self {
        self.batch_nonces = self.batch_nonces.clamp(64, 1 << 24);
        self.iters_per_dispatch = self.iters_per_dispatch.max(1);
        self.responsive_iters = self.responsive_iters.max(1);
        self.duty = self.duty.clamp(0.02, 1.0);
        self
    }
}

fn settings() -> &'static CudaSettings {
    static SETTINGS: OnceLock<CudaSettings> = OnceLock::new();
    SETTINGS.get_or_init(|| {
        let path =
            std::env::var("GPU_OC_SETTINGS").unwrap_or_else(|_| "GPU_OC_SETTINGS.toml".to_string());
        match std::fs::read_to_string(&path) {
            Ok(text) => match toml::from_str::<CudaSettings>(&text) {
                Ok(s) => {
                    let s = s.sanitized();
                    tracing::info!("loaded CUDA mining settings from {path}: {s:?}");
                    s
                }
                Err(e) => {
                    tracing::warn!("failed to parse {path} ({e}); using CUDA defaults");
                    CudaSettings::default()
                }
            },
            Err(_) => CudaSettings::default(),
        }
    })
}

static CUDA_DUTY_MILLI: AtomicU32 = AtomicU32::new(0);
static CUDA_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);

pub fn set_cuda_duty(duty: f32) {
    CUDA_DUTY_MILLI.store((duty.clamp(0.02, 1.0) * 1000.0) as u32, Ordering::Relaxed);
}

fn cuda_duty() -> f32 {
    if let Ok(s) = std::env::var("GPU_MINE_DUTY") {
        if let Ok(v) = s.parse::<f32>() {
            return v.clamp(0.02, 1.0);
        }
    }
    let milli = CUDA_DUTY_MILLI.load(Ordering::Relaxed);
    if milli != 0 {
        return milli as f32 / 1000.0;
    }
    settings().duty
}

struct CudaApi {
    _cuda: Library,
    _nvrtc: Library,
    cu_init: unsafe extern "C" fn(c_uint) -> CuResult,
    cu_driver_get_version: unsafe extern "C" fn(*mut c_int) -> CuResult,
    cu_device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    cu_device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    cu_device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    cu_device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult,
    cu_ctx_create: unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult,
    cu_ctx_destroy: unsafe extern "C" fn(CuContext) -> CuResult,
    cu_ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    cu_ctx_synchronize: unsafe extern "C" fn() -> CuResult,
    cu_module_load_data_ex: unsafe extern "C" fn(
        *mut CuModule,
        *const c_void,
        c_uint,
        *mut c_uint,
        *mut *mut c_void,
    ) -> CuResult,
    cu_module_unload: unsafe extern "C" fn(CuModule) -> CuResult,
    cu_module_get_function:
        unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    cu_mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    cu_mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    cu_memcpy_htod: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult,
    cu_launch_kernel: unsafe extern "C" fn(
        CuFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CuStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    cu_get_error_string: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
    nvrtc_create_program: unsafe extern "C" fn(
        *mut NvrtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> NvrtcResult,
    nvrtc_compile_program:
        unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult,
    nvrtc_get_ptx_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    nvrtc_get_ptx: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    nvrtc_get_program_log_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    nvrtc_get_program_log: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    nvrtc_destroy_program: unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult,
    nvrtc_get_error_string: unsafe extern "C" fn(NvrtcResult) -> *const c_char,
}

unsafe impl Send for CudaApi {}
unsafe impl Sync for CudaApi {}

impl CudaApi {
    fn load() -> Result<Arc<Self>> {
        static API: OnceLock<Result<Arc<CudaApi>, String>> = OnceLock::new();
        match API
            .get_or_init(|| unsafe { Self::load_inner().map(Arc::new).map_err(|e| e.to_string()) })
        {
            Ok(api) => Ok(api.clone()),
            Err(e) => bail!("{e}"),
        }
    }

    unsafe fn load_inner() -> Result<Self> {
        let cuda = load_library(&["libcuda.so.1", "libcuda.so", "nvcuda.dll"])?;
        let nvrtc = load_library(&[
            "libnvrtc.so",
            "libnvrtc.so.13",
            "libnvrtc.so.13.0",
            "libnvrtc.so.12",
            "libnvrtc.so.12.0",
            "libnvrtc.so.11.2",
            "/usr/local/cuda/lib64/libnvrtc.so",
            "/usr/local/cuda/lib64/libnvrtc.so.13",
            "/usr/local/cuda/lib64/libnvrtc.so.12",
            "nvrtc64_130_0.dll",
            "nvrtc64_120_0.dll",
            "nvrtc64_112_0.dll",
        ])?;

        let api = Self {
            cu_init: sym(&cuda, b"cuInit\0")?,
            cu_driver_get_version: sym(&cuda, b"cuDriverGetVersion\0")?,
            cu_device_get_count: sym(&cuda, b"cuDeviceGetCount\0")?,
            cu_device_get: sym(&cuda, b"cuDeviceGet\0")?,
            cu_device_get_name: sym(&cuda, b"cuDeviceGetName\0")?,
            cu_device_get_attribute: sym(&cuda, b"cuDeviceGetAttribute\0")?,
            cu_ctx_create: sym_any(&cuda, &[b"cuCtxCreate_v2\0", b"cuCtxCreate\0"])?,
            cu_ctx_destroy: sym_any(&cuda, &[b"cuCtxDestroy_v2\0", b"cuCtxDestroy\0"])?,
            cu_ctx_set_current: sym(&cuda, b"cuCtxSetCurrent\0")?,
            cu_ctx_synchronize: sym(&cuda, b"cuCtxSynchronize\0")?,
            cu_module_load_data_ex: sym(&cuda, b"cuModuleLoadDataEx\0")?,
            cu_module_unload: sym(&cuda, b"cuModuleUnload\0")?,
            cu_module_get_function: sym(&cuda, b"cuModuleGetFunction\0")?,
            cu_mem_alloc: sym_any(&cuda, &[b"cuMemAlloc_v2\0", b"cuMemAlloc\0"])?,
            cu_mem_free: sym_any(&cuda, &[b"cuMemFree_v2\0", b"cuMemFree\0"])?,
            cu_memcpy_htod: sym_any(&cuda, &[b"cuMemcpyHtoD_v2\0", b"cuMemcpyHtoD\0"])?,
            cu_memcpy_dtoh: sym_any(&cuda, &[b"cuMemcpyDtoH_v2\0", b"cuMemcpyDtoH\0"])?,
            cu_launch_kernel: sym(&cuda, b"cuLaunchKernel\0")?,
            cu_get_error_string: sym(&cuda, b"cuGetErrorString\0")?,
            nvrtc_create_program: sym(&nvrtc, b"nvrtcCreateProgram\0")?,
            nvrtc_compile_program: sym(&nvrtc, b"nvrtcCompileProgram\0")?,
            nvrtc_get_ptx_size: sym(&nvrtc, b"nvrtcGetPTXSize\0")?,
            nvrtc_get_ptx: sym(&nvrtc, b"nvrtcGetPTX\0")?,
            nvrtc_get_program_log_size: sym(&nvrtc, b"nvrtcGetProgramLogSize\0")?,
            nvrtc_get_program_log: sym(&nvrtc, b"nvrtcGetProgramLog\0")?,
            nvrtc_destroy_program: sym(&nvrtc, b"nvrtcDestroyProgram\0")?,
            nvrtc_get_error_string: sym(&nvrtc, b"nvrtcGetErrorString\0")?,
            _cuda: cuda,
            _nvrtc: nvrtc,
        };
        api.check_cuda((api.cu_init)(0), "cuInit")?;
        Ok(api)
    }

    fn check_cuda(&self, code: CuResult, op: &str) -> Result<()> {
        if code == CUDA_SUCCESS {
            Ok(())
        } else {
            bail!("{op} failed: {}", self.cuda_error(code))
        }
    }

    fn check_nvrtc(&self, code: NvrtcResult, op: &str) -> Result<()> {
        if code == NVRTC_SUCCESS {
            Ok(())
        } else {
            bail!("{op} failed: {}", self.nvrtc_error(code))
        }
    }

    fn cuda_error(&self, code: CuResult) -> String {
        unsafe {
            let mut msg: *const c_char = ptr::null();
            if (self.cu_get_error_string)(code, &mut msg) == CUDA_SUCCESS && !msg.is_null() {
                CStr::from_ptr(msg).to_string_lossy().into_owned()
            } else {
                format!("CUDA error {code}")
            }
        }
    }

    fn nvrtc_error(&self, code: NvrtcResult) -> String {
        unsafe {
            let ptr = (self.nvrtc_get_error_string)(code);
            if ptr.is_null() {
                format!("NVRTC error {code}")
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }

    fn driver_version(&self) -> Result<i32> {
        unsafe {
            let mut version = 0;
            self.check_cuda(
                (self.cu_driver_get_version)(&mut version),
                "cuDriverGetVersion",
            )?;
            Ok(version)
        }
    }

    fn device_count(&self) -> Result<i32> {
        unsafe {
            let mut count = 0;
            self.check_cuda((self.cu_device_get_count)(&mut count), "cuDeviceGetCount")?;
            Ok(count)
        }
    }

    fn device(&self, ordinal: i32) -> Result<CuDevice> {
        unsafe {
            let mut dev = 0;
            self.check_cuda((self.cu_device_get)(&mut dev, ordinal), "cuDeviceGet")?;
            Ok(dev)
        }
    }

    fn device_name(&self, dev: CuDevice) -> Result<String> {
        unsafe {
            let mut name = [0i8; 256];
            self.check_cuda(
                (self.cu_device_get_name)(name.as_mut_ptr(), name.len() as c_int, dev),
                "cuDeviceGetName",
            )?;
            Ok(CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned())
        }
    }

    fn compute_capability(&self, dev: CuDevice) -> Result<(i32, i32)> {
        unsafe {
            let mut major = 0;
            let mut minor = 0;
            self.check_cuda(
                (self.cu_device_get_attribute)(
                    &mut major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    dev,
                ),
                "cuDeviceGetAttribute(major)",
            )?;
            self.check_cuda(
                (self.cu_device_get_attribute)(
                    &mut minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    dev,
                ),
                "cuDeviceGetAttribute(minor)",
            )?;
            Ok((major, minor))
        }
    }

    fn compile_ptx(&self, source: &str, major: i32, minor: i32) -> Result<Vec<u8>> {
        let mut candidates = vec![format!("compute_{major}{minor}")];
        for arch in [
            "compute_120",
            "compute_100",
            "compute_90",
            "compute_89",
            "compute_86",
            "compute_80",
            "compute_75",
            "compute_70",
            "compute_61",
            "compute_60",
        ] {
            if !candidates.iter().any(|a| a == arch) {
                candidates.push(arch.to_string());
            }
        }

        let mut last_err = None;
        for arch in candidates {
            match self.compile_ptx_once(source, &arch) {
                Ok(ptx) => {
                    tracing::info!("compiled CUDA kernel for {arch}");
                    return Ok(ptx);
                }
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("NVRTC could not compile CUDA kernel")))
    }

    fn compile_ptx_once(&self, source: &str, arch: &str) -> Result<Vec<u8>> {
        unsafe {
            let source = CString::new(source)?;
            let name = CString::new("midstate_cuda_miner.cu")?;
            let mut program = ptr::null_mut();
            self.check_nvrtc(
                (self.nvrtc_create_program)(
                    &mut program,
                    source.as_ptr(),
                    name.as_ptr(),
                    0,
                    ptr::null(),
                    ptr::null(),
                ),
                "nvrtcCreateProgram",
            )?;

            let opt_std = CString::new("--std=c++11")?;
            let opt_arch = CString::new(format!("--gpu-architecture={arch}"))?;
            let options = [opt_std.as_ptr(), opt_arch.as_ptr()];
            let compile_result =
                (self.nvrtc_compile_program)(program, options.len() as c_int, options.as_ptr());
            if compile_result != NVRTC_SUCCESS {
                let log = self.program_log(program);
                let _ = (self.nvrtc_destroy_program)(&mut program);
                bail!(
                    "nvrtcCompileProgram failed for {arch}: {}{}",
                    self.nvrtc_error(compile_result),
                    if log.is_empty() {
                        String::new()
                    } else {
                        format!("; log: {log}")
                    }
                );
            }

            let mut ptx_size = 0usize;
            self.check_nvrtc(
                (self.nvrtc_get_ptx_size)(program, &mut ptx_size),
                "nvrtcGetPTXSize",
            )?;
            let mut ptx = vec![0u8; ptx_size];
            self.check_nvrtc(
                (self.nvrtc_get_ptx)(program, ptx.as_mut_ptr().cast::<c_char>()),
                "nvrtcGetPTX",
            )?;
            self.check_nvrtc(
                (self.nvrtc_destroy_program)(&mut program),
                "nvrtcDestroyProgram",
            )?;
            Ok(ptx)
        }
    }

    fn program_log(&self, program: NvrtcProgram) -> String {
        unsafe {
            let mut log_size = 0usize;
            if (self.nvrtc_get_program_log_size)(program, &mut log_size) != NVRTC_SUCCESS
                || log_size == 0
            {
                return String::new();
            }
            let mut log = vec![0u8; log_size];
            if (self.nvrtc_get_program_log)(program, log.as_mut_ptr().cast::<c_char>())
                != NVRTC_SUCCESS
            {
                return String::new();
            }
            String::from_utf8_lossy(&log)
                .trim_matches(char::from(0))
                .trim()
                .to_string()
        }
    }
}

unsafe fn load_library(candidates: &[&str]) -> Result<Library> {
    let mut errors = Vec::new();
    for name in candidates {
        match Library::new(name) {
            Ok(lib) => return Ok(lib),
            Err(e) => errors.push(format!("{name}: {e}")),
        }
    }
    bail!("could not load CUDA library; tried {}", errors.join(", "))
}

unsafe fn sym<T: Copy>(lib: &Library, name: &[u8]) -> Result<T> {
    Ok(*lib.get::<T>(name)?)
}

unsafe fn sym_any<T: Copy>(lib: &Library, names: &[&[u8]]) -> Result<T> {
    let mut last = None;
    for name in names {
        match lib.get::<T>(name) {
            Ok(s) => return Ok(*s),
            Err(e) => last = Some(e),
        }
    }
    Err(last
        .map(anyhow::Error::from)
        .unwrap_or_else(|| anyhow!("empty CUDA symbol candidate list")))
}

pub struct CudaMiner {
    api: Arc<CudaApi>,
    ctx: CuContext,
    module: CuModule,
    k_init: CuFunction,
    k_step: CuFunction,
    k_test: CuFunction,
    d_params: CuDevicePtr,
    d_state: CuDevicePtr,
    d_winners: CuDevicePtr,
    adapter_name: String,
    state: Mutex<MinerState>,
}

unsafe impl Send for CudaMiner {}
unsafe impl Sync for CudaMiner {}

impl CudaMiner {
    fn new_for_device(api: Arc<CudaApi>, ordinal: i32) -> Result<Self> {
        unsafe {
            let dev = api.device(ordinal)?;
            let adapter_name = api.device_name(dev)?;
            let (major, minor) = api.compute_capability(dev)?;
            tracing::info!("CUDA adapter found: {adapter_name} [compute {major}.{minor}]");

            let mut ctx = ptr::null_mut();
            api.check_cuda((api.cu_ctx_create)(&mut ctx, 0, dev), "cuCtxCreate")?;
            api.check_cuda((api.cu_ctx_set_current)(ctx), "cuCtxSetCurrent")?;

            let ptx = match api.compile_ptx(cuda_source(), major, minor) {
                Ok(ptx) => ptx,
                Err(e) => {
                    let _ = (api.cu_ctx_destroy)(ctx);
                    return Err(e);
                }
            };

            let mut module = ptr::null_mut();
            api.check_cuda(
                (api.cu_module_load_data_ex)(
                    &mut module,
                    ptx.as_ptr().cast::<c_void>(),
                    0,
                    ptr::null_mut(),
                    ptr::null_mut(),
                ),
                "cuModuleLoadDataEx",
            )?;

            let k_init_name = CString::new("k_init")?;
            let k_step_name = CString::new("k_step")?;
            let k_test_name = CString::new("k_test")?;
            let mut k_init = ptr::null_mut();
            let mut k_step = ptr::null_mut();
            let mut k_test = ptr::null_mut();
            api.check_cuda(
                (api.cu_module_get_function)(&mut k_init, module, k_init_name.as_ptr()),
                "cuModuleGetFunction(k_init)",
            )?;
            api.check_cuda(
                (api.cu_module_get_function)(&mut k_step, module, k_step_name.as_ptr()),
                "cuModuleGetFunction(k_step)",
            )?;
            api.check_cuda(
                (api.cu_module_get_function)(&mut k_test, module, k_test_name.as_ptr()),
                "cuModuleGetFunction(k_test)",
            )?;

            let max_state_bytes = (settings().batch_nonces as usize) * 8 * 4;
            let mut d_params = 0;
            let mut d_state = 0;
            let mut d_winners = 0;
            api.check_cuda(
                (api.cu_mem_alloc)(&mut d_params, std::mem::size_of::<Params>()),
                "cuMemAlloc(params)",
            )?;
            api.check_cuda(
                (api.cu_mem_alloc)(&mut d_state, max_state_bytes),
                "cuMemAlloc(state)",
            )?;
            api.check_cuda(
                (api.cu_mem_alloc)(&mut d_winners, WINNERS_BYTES),
                "cuMemAlloc(winners)",
            )?;

            Ok(Self {
                api,
                ctx,
                module,
                k_init,
                k_step,
                k_test,
                d_params,
                d_state,
                d_winners,
                adapter_name,
                state: Mutex::new(MinerState {
                    job: None,
                    pending: VecDeque::new(),
                }),
            })
        }
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    fn groups(n_nonces: u32) -> u32 {
        (n_nonces + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
    }

    fn set_current(&self) -> Result<()> {
        unsafe {
            self.api
                .check_cuda((self.api.cu_ctx_set_current)(self.ctx), "cuCtxSetCurrent")
        }
    }

    fn write_params(&self, params: &Params) -> Result<()> {
        unsafe {
            self.api.check_cuda(
                (self.api.cu_memcpy_htod)(
                    self.d_params,
                    bytes_of(params).as_ptr().cast::<c_void>(),
                    std::mem::size_of::<Params>(),
                ),
                "cuMemcpyHtoD(params)",
            )
        }
    }

    fn launch2(&self, func: CuFunction, groups: u32) -> Result<()> {
        unsafe {
            let mut p_params = self.d_params;
            let mut p_state = self.d_state;
            let mut args = [
                (&mut p_params as *mut CuDevicePtr).cast::<c_void>(),
                (&mut p_state as *mut CuDevicePtr).cast::<c_void>(),
            ];
            self.api.check_cuda(
                (self.api.cu_launch_kernel)(
                    func,
                    groups,
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )
        }
    }

    fn launch3(&self, func: CuFunction, groups: u32) -> Result<()> {
        unsafe {
            let mut p_params = self.d_params;
            let mut p_state = self.d_state;
            let mut p_winners = self.d_winners;
            let mut args = [
                (&mut p_params as *mut CuDevicePtr).cast::<c_void>(),
                (&mut p_state as *mut CuDevicePtr).cast::<c_void>(),
                (&mut p_winners as *mut CuDevicePtr).cast::<c_void>(),
            ];
            self.api.check_cuda(
                (self.api.cu_launch_kernel)(
                    func,
                    groups,
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )
        }
    }

    fn synchronize(&self) -> Result<()> {
        unsafe {
            self.api
                .check_cuda((self.api.cu_ctx_synchronize)(), "cuCtxSynchronize")
        }
    }

    fn run_batch(
        &self,
        params: &mut Params,
        base: u64,
        n_nonces: u32,
        cancel: &AtomicBool,
        hash_counter: &AtomicU64,
        collect_winners: bool,
    ) -> Option<Vec<(u64, u32)>> {
        self.set_current().ok()?;
        params.base_lo = base as u32;
        params.base_hi = (base >> 32) as u32;
        params.n_nonces = n_nonces;
        params.iters = 0;
        self.write_params(params).ok()?;

        let mut winners_words = vec![0u32; WINNERS_WORDS];
        winners_words[1] = MAX_WINNERS;
        unsafe {
            self.api
                .check_cuda(
                    (self.api.cu_memcpy_htod)(
                        self.d_winners,
                        winners_words.as_ptr().cast::<c_void>(),
                        WINNERS_BYTES,
                    ),
                    "cuMemcpyHtoD(winners)",
                )
                .ok()?;
        }

        let groups = Self::groups(n_nonces);
        self.launch2(self.k_init, groups).ok()?;
        self.synchronize().ok()?;

        let total = EXTENSION_ITERATIONS.max(1);
        let duty = cuda_duty();
        let throttling = duty < 0.999;
        let active_iters = if throttling {
            settings().responsive_iters
        } else {
            settings().iters_per_dispatch
        };
        let mut remaining = EXTENSION_ITERATIONS;

        while remaining > 0 {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }

            let k = remaining.min(active_iters as u64) as u32;
            params.iters = k;
            self.write_params(params).ok()?;

            let t0 = Instant::now();
            self.launch2(self.k_step, groups).ok()?;
            self.synchronize().ok()?;

            if throttling {
                let work = t0.elapsed();
                let idle = work.mul_f32((1.0 / duty) - 1.0);
                if !idle.is_zero() {
                    thread::sleep(idle);
                }
            }

            remaining -= k as u64;
            let add = (n_nonces as u64).saturating_mul(k as u64) / total;
            hash_counter.fetch_add(add, Ordering::Relaxed);
        }

        if !collect_winners {
            return Some(Vec::new());
        }

        self.launch3(self.k_test, groups).ok()?;
        self.synchronize().ok()?;

        let mut words = vec![0u32; WINNERS_WORDS];
        unsafe {
            self.api
                .check_cuda(
                    (self.api.cu_memcpy_dtoh)(
                        words.as_mut_ptr().cast::<c_void>(),
                        self.d_winners,
                        WINNERS_BYTES,
                    ),
                    "cuMemcpyDtoH(winners)",
                )
                .ok()?;
        }

        let count = words[0].min(MAX_WINNERS) as usize;
        let lo_off = 4usize;
        let hi_off = lo_off + MAX_WINNERS as usize;
        let kind_off = hi_off + MAX_WINNERS as usize;
        let mut winners = Vec::with_capacity(count);
        for j in 0..count {
            let nonce = (words[lo_off + j] as u64) | ((words[hi_off + j] as u64) << 32);
            winners.push((nonce, words[kind_off + j]));
        }
        Some(winners)
    }

    pub fn mine_gpu(
        &self,
        midstate: [u8; 32],
        target: [u8; 32],
        pool_target: Option<[u8; 32]>,
        cancel: Arc<AtomicBool>,
        hash_counter: Arc<AtomicU64>,
    ) -> Option<MiningResult> {
        let job: JobKey = (midstate, target, pool_target);
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());

        if st.job.as_ref() != Some(&job) {
            st.job = Some(job);
            st.pending.clear();
        }
        if let Some(hit) = st.pending.pop_front() {
            return Some(hit);
        }

        let (pool_words, has_pool) = match pool_target {
            Some(p) => (words_be(&p), 1u32),
            None => ([0u32; 8], 0u32),
        };
        let mut params = Params {
            midstate: words_le(&midstate),
            target: words_be(&target),
            pool: pool_words,
            base_lo: 0,
            base_hi: 0,
            n_nonces: settings().batch_nonces,
            iters: 0,
            has_pool,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        loop {
            if cancel.load(Ordering::Relaxed) {
                tracing::debug!("CUDA mining cancelled");
                return None;
            }

            let base: u64 = rand::random();
            let winners = self.run_batch(
                &mut params,
                base,
                settings().batch_nonces,
                &cancel,
                &hash_counter,
                true,
            )?;

            let mut hits = Vec::new();
            for (nonce, _kind) in winners {
                let final_hash = create_extension(midstate, nonce).final_hash;
                if final_hash < target {
                    hits.push(MiningResult::Block(Extension { nonce, final_hash }));
                } else if let Some(pt) = pool_target {
                    if final_hash < pt {
                        hits.push(MiningResult::Share(Extension { nonce, final_hash }));
                    }
                }
            }

            if hits.is_empty() {
                continue;
            }
            hits.sort_by_key(|h| matches!(h, MiningResult::Share(_)));

            let mut it = hits.into_iter();
            let first = it.next().unwrap();
            match &first {
                MiningResult::Block(e) => tracing::info!(
                    "CUDA found valid block! nonce={} hash={} gpu={}",
                    e.nonce,
                    hex::encode(e.final_hash),
                    self.adapter_name
                ),
                MiningResult::Share(e) => tracing::info!(
                    "CUDA found valid pool share! nonce={} hash={} gpu={}",
                    e.nonce,
                    hex::encode(e.final_hash),
                    self.adapter_name
                ),
            }
            st.pending.extend(it);
            return Some(first);
        }
    }

    pub fn self_test(&self) -> Result<()> {
        let midstate = [0xA5u8; 32];
        let never = AtomicBool::new(false);
        let sink = AtomicU64::new(0);
        let base = 0u64;
        let mut params = Params {
            midstate: words_le(&midstate),
            target: [0u32; 8],
            pool: [0u32; 8],
            base_lo: 0,
            base_hi: 0,
            n_nonces: SELFTEST_N,
            iters: 0,
            has_pool: 0,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        self.run_batch(&mut params, base, SELFTEST_N, &never, &sink, false)
            .ok_or_else(|| anyhow!("self-test batch was unexpectedly cancelled"))?;

        let mut words = vec![0u32; SELFTEST_N as usize * 8];
        self.set_current()?;
        unsafe {
            self.api.check_cuda(
                (self.api.cu_memcpy_dtoh)(
                    words.as_mut_ptr().cast::<c_void>(),
                    self.d_state,
                    words.len() * 4,
                ),
                "cuMemcpyDtoH(state)",
            )?;
        }

        for gid in 0..SELFTEST_N as u64 {
            let expected = create_extension(midstate, base + gid).final_hash;
            let mut got = [0u8; 32];
            for i in 0..8usize {
                let w = words[gid as usize * 8 + i];
                got[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            if got != expected {
                bail!(
                    "CUDA self-test failed on {} at nonce {gid}: gpu={} expected={}",
                    self.adapter_name,
                    hex::encode(got),
                    hex::encode(expected)
                );
            }
        }

        tracing::info!(
            "CUDA self-test passed on {} ({} nonces)",
            self.adapter_name,
            SELFTEST_N
        );
        Ok(())
    }
}

impl Drop for CudaMiner {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.api.cu_ctx_set_current)(self.ctx);
            if self.d_params != 0 {
                let _ = (self.api.cu_mem_free)(self.d_params);
            }
            if self.d_state != 0 {
                let _ = (self.api.cu_mem_free)(self.d_state);
            }
            if self.d_winners != 0 {
                let _ = (self.api.cu_mem_free)(self.d_winners);
            }
            if !self.module.is_null() {
                let _ = (self.api.cu_module_unload)(self.module);
            }
            if !self.ctx.is_null() {
                let _ = (self.api.cu_ctx_destroy)(self.ctx);
            }
        }
    }
}

pub struct CudaFarm {
    miners: Vec<Arc<CudaMiner>>,
    adapter_names: String,
}

impl CudaFarm {
    pub fn new() -> Result<Self> {
        let api = CudaApi::load()?;
        let driver = api.driver_version().unwrap_or_default();
        tracing::info!("CUDA driver version reported by libcuda: {driver}");

        let count = api.device_count()?;
        if count <= 0 {
            bail!("CUDA driver reports no devices");
        }

        let mut miners = Vec::new();
        for ordinal in 0..count {
            match CudaMiner::new_for_device(api.clone(), ordinal) {
                Ok(miner) => match miner.self_test() {
                    Ok(()) => {
                        tracing::info!("CUDA mining enabled on {}", miner.adapter_name());
                        miners.push(Arc::new(miner));
                    }
                    Err(e) => {
                        tracing::warn!("CUDA adapter #{ordinal} disabled (self-test failed): {e}")
                    }
                },
                Err(e) => tracing::warn!("CUDA adapter #{ordinal} disabled (init failed): {e}"),
            }
        }

        if miners.is_empty() {
            bail!("no CUDA adapters passed initialization and self-test");
        }

        let adapter_names = miners
            .iter()
            .map(|m| m.adapter_name().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        Ok(Self {
            miners,
            adapter_names,
        })
    }

    pub fn len(&self) -> usize {
        self.miners.len()
    }

    pub fn adapter_names(&self) -> &str {
        &self.adapter_names
    }

    pub fn mine_gpu(
        &self,
        midstate: [u8; 32],
        target: [u8; 32],
        pool_target: Option<[u8; 32]>,
        external_cancel: Arc<AtomicBool>,
        hash_counter: Arc<AtomicU64>,
    ) -> Option<MiningResult> {
        if self.miners.len() == 1 {
            return self.miners[0].mine_gpu(
                midstate,
                target,
                pool_target,
                external_cancel,
                hash_counter,
            );
        }

        let race_cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::with_capacity(self.miners.len() + 1);

        {
            let race_cancel = race_cancel.clone();
            let external_cancel = external_cancel.clone();
            handles.push(thread::spawn(move || {
                while !race_cancel.load(Ordering::Relaxed)
                    && !external_cancel.load(Ordering::Relaxed)
                {
                    thread::sleep(std::time::Duration::from_millis(25));
                }
                if external_cancel.load(Ordering::Relaxed) {
                    race_cancel.store(true, Ordering::Relaxed);
                }
            }));
        }

        for miner in &self.miners {
            let miner = miner.clone();
            let tx = tx.clone();
            let cancel = race_cancel.clone();
            let hash_counter = hash_counter.clone();
            handles.push(thread::spawn(move || {
                let result = miner.mine_gpu(midstate, target, pool_target, cancel, hash_counter);
                let _ = tx.send(result);
            }));
        }
        drop(tx);

        let mut found = None;
        for _ in 0..self.miners.len() {
            match rx.recv() {
                Ok(Some(hit)) => {
                    found = Some(hit);
                    race_cancel.store(true, Ordering::Relaxed);
                    break;
                }
                Ok(None) => {}
                Err(_) => break,
            }
        }

        race_cancel.store(true, Ordering::Relaxed);
        for handle in handles {
            let _ = handle.join();
        }
        found
    }
}

pub fn shared() -> Option<&'static CudaFarm> {
    static SHARED: OnceLock<Option<CudaFarm>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            if std::env::var("MINER_DISABLE_GPU")
                .map(|v| v != "0")
                .unwrap_or(false)
            {
                tracing::info!("CUDA mining disabled via MINER_DISABLE_GPU");
                return None;
            }
            if std::env::var("MINER_DISABLE_CUDA")
                .map(|v| v != "0")
                .unwrap_or(false)
            {
                tracing::info!("CUDA mining disabled via MINER_DISABLE_CUDA");
                return None;
            }
            match CudaFarm::new() {
                Ok(farm) => {
                    tracing::info!(
                        "CUDA mining enabled on {} device(s): {}",
                        farm.len(),
                        farm.adapter_names()
                    );
                    Some(farm)
                }
                Err(e) => {
                    tracing::info!("CUDA mining disabled (no usable device): {e}");
                    None
                }
            }
        })
        .as_ref()
}

pub fn cuda_available() -> bool {
    shared().is_some()
}

pub fn mine_cuda(
    midstate: [u8; 32],
    target: [u8; 32],
    pool_target: Option<[u8; 32]>,
    cancel: Arc<AtomicBool>,
    hash_counter: Arc<AtomicU64>,
) -> Option<MiningResult> {
    shared()?.mine_gpu(midstate, target, pool_target, cancel, hash_counter)
}

pub fn mine_cuda_or_cpu(
    midstate: [u8; 32],
    target: [u8; 32],
    pool_target: Option<[u8; 32]>,
    threads: usize,
    cancel: Arc<AtomicBool>,
    hash_counter: Arc<AtomicU64>,
) -> Option<MiningResult> {
    if let Some(farm) = shared() {
        return farm.mine_gpu(midstate, target, pool_target, cancel, hash_counter);
    }
    if !CUDA_FALLBACK_WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!("CUDA backend requested but no usable CUDA GPU; mining on CPU");
    }
    mine_extension(midstate, target, pool_target, threads, cancel, hash_counter)
}

fn words_le(b: &[u8; 32]) -> [u32; 8] {
    let mut w = [0u32; 8];
    for i in 0..8 {
        w[i] = u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
    }
    w
}

fn words_be(b: &[u8; 32]) -> [u32; 8] {
    let mut w = [0u32; 8];
    for i in 0..8 {
        w[i] = u32::from_be_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
    }
    w
}

fn bytes_of<T>(value: &T) -> &[u8] {
    unsafe { slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>()) }
}

fn cuda_source() -> &'static str {
    r#"
typedef unsigned int u32;
typedef unsigned long long u64;

struct Params {
    u32 midstate[8];
    u32 target[8];
    u32 pool[8];
    u32 base_lo;
    u32 base_hi;
    u32 n_nonces;
    u32 iters;
    u32 has_pool;
    u32 pad0;
    u32 pad1;
    u32 pad2;
};

struct Winners {
    u32 count;
    u32 cap;
    u32 pad0;
    u32 pad1;
    u32 nonce_lo[256];
    u32 nonce_hi[256];
    u32 kind[256];
};

__constant__ int MSG[7][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
    { 2, 6, 3,10, 7, 0, 4,13, 1,11,12, 5, 9,14,15, 8 },
    { 3, 4,10,12,13, 2, 7,14, 6, 5, 9, 0,11,15, 8, 1 },
    {10, 7,12, 9,14, 3,13,15, 4, 0,11, 2, 5, 8, 1, 6 },
    {12,13, 9,11,15,10,14, 8, 7, 2, 5, 3, 0, 1, 6, 4 },
    { 9,14,11, 5, 8,12,15, 1,13, 3, 0,10, 2, 6, 4, 7 },
    {11,15, 5, 0, 1, 9, 8, 6,14,10, 2,12, 3, 4, 7,13 }
};

__device__ __forceinline__ u32 rotr32(u32 x, u32 n) {
    return (x >> n) | (x << (32u - n));
}

__device__ __forceinline__ u32 bswap32(u32 x) {
    return ((x & 0x000000ffu) << 24) |
           ((x & 0x0000ff00u) << 8)  |
           ((x & 0x00ff0000u) >> 8)  |
           ((x & 0xff000000u) >> 24);
}

__device__ __forceinline__ void g(u32 v[16], int a, int b, int c, int d, u32 x, u32 y) {
    v[a] = v[a] + v[b] + x;
    v[d] = rotr32(v[d] ^ v[a], 16u);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 12u);
    v[a] = v[a] + v[b] + y;
    v[d] = rotr32(v[d] ^ v[a], 8u);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 7u);
}

__device__ __forceinline__ void compress(const u32 m[16], u32 block_len, u32 out[8]) {
    u32 v[16] = {
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0u, 0u, block_len, 11u
    };

    #pragma unroll
    for (int r = 0; r < 7; ++r) {
        g(v, 0, 4,  8, 12, m[MSG[r][ 0]], m[MSG[r][ 1]]);
        g(v, 1, 5,  9, 13, m[MSG[r][ 2]], m[MSG[r][ 3]]);
        g(v, 2, 6, 10, 14, m[MSG[r][ 4]], m[MSG[r][ 5]]);
        g(v, 3, 7, 11, 15, m[MSG[r][ 6]], m[MSG[r][ 7]]);
        g(v, 0, 5, 10, 15, m[MSG[r][ 8]], m[MSG[r][ 9]]);
        g(v, 1, 6, 11, 12, m[MSG[r][10]], m[MSG[r][11]]);
        g(v, 2, 7,  8, 13, m[MSG[r][12]], m[MSG[r][13]]);
        g(v, 3, 4,  9, 14, m[MSG[r][14]], m[MSG[r][15]]);
    }

    out[0] = v[0] ^ v[8];
    out[1] = v[1] ^ v[9];
    out[2] = v[2] ^ v[10];
    out[3] = v[3] ^ v[11];
    out[4] = v[4] ^ v[12];
    out[5] = v[5] ^ v[13];
    out[6] = v[6] ^ v[14];
    out[7] = v[7] ^ v[15];
}

__device__ __forceinline__ u64 nonce_for(const Params* p, u32 gid) {
    u32 lo = p->base_lo + gid;
    u32 carry = lo < p->base_lo ? 1u : 0u;
    u32 hi = p->base_hi + carry;
    return ((u64)hi << 32) | (u64)lo;
}

__device__ __forceinline__ void first_compress(const Params* p, u32 gid, u32 h[8]) {
    u32 m[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = p->midstate[i];
    u64 nonce = nonce_for(p, gid);
    m[8] = (u32)nonce;
    m[9] = (u32)(nonce >> 32);
    #pragma unroll
    for (int i = 10; i < 16; ++i) m[i] = 0u;
    compress(m, 40u, h);
}

__device__ __forceinline__ void iterate_hash(u32 h[8]) {
    u32 m[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = h[i];
    #pragma unroll
    for (int i = 8; i < 16; ++i) m[i] = 0u;
    compress(m, 32u, h);
}

__device__ __forceinline__ bool lt8(const u32 h[8], const u32 target[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        u32 k = bswap32(h[i]);
        if (k < target[i]) return true;
        if (k > target[i]) return false;
    }
    return false;
}

extern "C" __global__ void k_init(const Params* p, u32* state) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p->n_nonces) return;
    u32 h[8];
    first_compress(p, gid, h);
    u32 off = gid * 8u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) state[off + i] = h[i];
}

extern "C" __global__ void k_step(const Params* p, u32* state) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p->n_nonces) return;
    u32 h[8];
    u32 off = gid * 8u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = state[off + i];
    for (u32 i = 0; i < p->iters; ++i) {
        iterate_hash(h);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) state[off + i] = h[i];
}

extern "C" __global__ void k_test(const Params* p, u32* state, Winners* out) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p->n_nonces) return;
    u32 h[8];
    u32 off = gid * 8u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = state[off + i];

    u32 kind = 0xffffffffu;
    if (lt8(h, p->target)) {
        kind = 0u;
    } else if (p->has_pool != 0u && lt8(h, p->pool)) {
        kind = 1u;
    }

    if (kind != 0xffffffffu) {
        u32 idx = atomicAdd(&out->count, 1u);
        if (idx < out->cap) {
            u64 nonce = nonce_for(p, gid);
            out->nonce_lo[idx] = (u32)nonce;
            out->nonce_hi[idx] = (u32)(nonce >> 32);
            out->kind[idx] = kind;
        }
    }
}
"#
}
