// ============================================================
//  HardwareDetector.cs — Auto-detect OS + CPU topology
//
//  Optimized for: VMware shared-host Windows VPS, 16 vCPU
//
//  Key design decisions for VM environment:
//  1. WMI physical core count is UNRELIABLE in VMware — vCPUs always
//     report as "1 core per socket" or similar virtual topology.
//     We treat LogicalProcessorCount as ground truth.
//  2. CPU affinity pinning (OMP_PROC_BIND/OMP_PLACES) is DISABLED for
//     VMs — VMware's own scheduler manages vCPU placement on pCPUs.
//     Pinning inside a VM can cause vCPU stalls when the hypervisor
//     migrates vCPUs between pCPUs (which happens on shared hosts).
//  3. Thread budget: on a shared host we reserve ~25% of vCPUs for
//     OS overhead, ASP.NET threadpool, and host-side contention.
//     16 vCPU → 12 ORT intra-op threads is the sweet spot.
//  4. AVX2 detection on Windows VM: Xeon E5-2690 v4 (Broadwell-EP)
//     confirmed supports AVX2 + FMA3. VMware exposes AVX2 to guests
//     by default since vHW 11 (ESXi 6.0+). Safe to hardcode true.
// ============================================================

public static class HardwareDetector
{
    public record HardwareInfo(
        string OS,
        int LogicalCores,
        int PhysicalCores,
        bool IsVirtualMachine,
        bool HasAVX2,
        bool HasAVX512
    );

    public static HardwareInfo Detect(ILogger logger)
    {
        int logical = Environment.ProcessorCount;

        // ── VM detection ─────────────────────────────────────────────────────
        // Check BIOS/manufacturer strings that VMware injects.
        // CPU-Z shows "VMware Inc." as manufacturer and "440BX Desktop
        // Reference Platform" as model — these are VMware's virtual DMI strings.
        bool isVm = DetectVirtualMachine(logger);

        // ── Physical core detection ───────────────────────────────────────────
        // On VMware, WMI Win32_Processor.NumberOfCores returns vCPU count
        // divided by virtual socket topology — not the host's physical cores.
        // For a 16 vCPU VM configured as 1 socket × 16 cores, WMI returns 16.
        // For 4 sockets × 4 cores, WMI returns 4. Either way, logical == physical
        // in a VM without SMT passthrough enabled. We just use logical as physical.
        int physical = isVm
            ? logical   // In VM: logical = physical (no HT exposed by default)
            : DetectPhysicalCoresWindows(logical, logger);

        // ── AVX2 detection ────────────────────────────────────────────────────
        // Xeon E5-2690 v4 is Broadwell-EP — confirmed AVX2 + FMA3.
        // VMware exposes these via CPUID passthrough. CPU-Z confirms:
        // "AVX, AVX2, FMA3" in the Instructions field.
        // AVX-512 is NOT available on Broadwell-EP (Skylake-SP introduced it).
        bool avx2 = true;   // Confirmed: Broadwell-EP + VMware CPUID passthrough
        bool avx512 = false;  // Confirmed: not available on Broadwell-EP

        var info = new HardwareInfo(
            OS: "Windows",
            LogicalCores: logical,
            PhysicalCores: physical,
            IsVirtualMachine: isVm,
            HasAVX2: avx2,
            HasAVX512: avx512);

        logger.LogInformation(
            "[Hardware] OS={OS} Logical={L} Physical={P} VM={VM} AVX2={A2} AVX512={A5}",
            info.OS, info.LogicalCores, info.PhysicalCores,
            info.IsVirtualMachine, info.HasAVX2, info.HasAVX512);

        return info;
    }

    public static void ApplyToConfig(ModelConfig cfg, HardwareInfo hw, ILogger logger)
    {
        // ── Thread budget calculation ─────────────────────────────────────────
        //
        //  Shared VMware host means we can't guarantee all vCPUs are free.
        //  Host-side overhead (other VMs, VMkernel, storage I/O) steals pCPU
        //  cycles from our vCPUs unpredictably.
        //
        //  16 vCPU budget breakdown:
        //    12 → ORT intra-op (ONNX compute)
        //     2 → ASP.NET Kestrel + request handling
        //     1 → .NET GC (server GC uses dedicated threads)
        //     1 → OS + VMware tools + disk/net I/O
        //
        //  Going to 14-16 on a shared host causes:
        //    - Thread context switching spikes when host is busy
        //    - L3 cache thrashing (pCPU L3 is shared with other VMs)
        //    - Higher P99 latency even if median is similar
        //
        int intra;
        if (hw.LogicalCores >= 16)
            intra = 12;         // 16 vCPU: use 12 for ORT, 4 for OS/ASP.NET/GC
        else if (hw.LogicalCores >= 12)
            intra = 10;         // 12 vCPU: use 10
        else if (hw.LogicalCores >= 8)
            intra = hw.LogicalCores - 2;  // 8 vCPU: leave 2 for overhead
        else if (hw.LogicalCores >= 4)
            intra = hw.LogicalCores - 1;  // 4 vCPU: leave 1
        else
            intra = Math.Max(1, hw.LogicalCores);

        // InterOp: 2 allows parallel execution of independent ONNX graph branches.
        // Keep at 2 — higher values don't help for sequential decoder inference
        // and add scheduler overhead.
        cfg.IntraOpThreads = intra;
        cfg.InterOpThreads = 2;

        // ── Windows VM environment variables ─────────────────────────────────

        // Thread counts — align OMP/MKL/ORT to same value as ORT session config
        Environment.SetEnvironmentVariable("OMP_NUM_THREADS", intra.ToString());
        Environment.SetEnvironmentVariable("MKL_NUM_THREADS", intra.ToString());
        Environment.SetEnvironmentVariable("ORT_NUM_THREADS", intra.ToString());

        // DNNL: force AVX2 path (confirmed available on Broadwell-EP + VMware)
        // Do NOT use AVX512 — not available on this CPU
        Environment.SetEnvironmentVariable("DNNL_MAX_CPU_ISA", "AVX2");

        // Disable DNNL JIT dump overhead — static model, no recompile needed
        Environment.SetEnvironmentVariable("DNNL_JIT_DUMP", "0");

        // MKL: explicitly target AVX2 instruction path
        Environment.SetEnvironmentVariable("MKL_ENABLE_INSTRUCTIONS", "AVX2");
        Environment.SetEnvironmentVariable("MKL_CBWR", "AVX2");

        // ── CRITICAL for VM: DO NOT set OMP_PROC_BIND or OMP_PLACES ─────────
        //
        // These env vars tell OpenMP to pin threads to specific CPU IDs.
        // On bare metal this reduces NUMA hops and improves cache locality.
        // On VMware, vCPU IDs are VIRTUAL — the hypervisor remaps them to
        // physical cores dynamically. When a vCPU gets migrated mid-execution
        // (common on shared hosts), a pinned OMP thread is now on the "wrong"
        // physical core but still thinks it's pinned — this can cause:
        //   - Artificial wait loops (spin-wait on wrong NUMA node)
        //   - vCPU ready stalls (VMware can't schedule the vCPU on the
        //     physical core the thread is "pinned" to)
        //   - Deadlocks in extreme cases (OMP barrier with migrated vCPU)
        //
        // Leave OMP_PROC_BIND and OMP_PLACES UNSET. VMware's NUMA scheduler
        // (NUMA Client feature) handles locality far better than guest pinning.
        //
        // Environment.SetEnvironmentVariable("OMP_PROC_BIND", "close");  // DISABLED
        // Environment.SetEnvironmentVariable("OMP_PLACES", "cores");     // DISABLED

        // ── .NET GC tuning for inference server ──────────────────────────────
        // ConserveMemory=5: moderate GC aggressiveness
        // On 64 GB VM, we have headroom — don't be too aggressive
        // Value 0-9: higher = more conservative memory use (more frequent GC)
        Environment.SetEnvironmentVariable("DOTNET_GCConserveMemory", "3");

        // ── DNNL primitive caching ────────────────────────────────────────────
        // Increase primitive cache size — Qwen3.5 hybrid model reuses many
        // DNNL primitives across decode steps. Default is 1024, bump to 4096.
        Environment.SetEnvironmentVariable("DNNL_PRIMITIVE_CACHE_CAPACITY", "4096");

        logger.LogInformation(
            "[Hardware] VM={VM} → ORT threads IntraOp={I} InterOp={IO} | AVX2={A2} | OMP affinity=DISABLED (VM safe)",
            hw.IsVirtualMachine, cfg.IntraOpThreads, cfg.InterOpThreads, hw.HasAVX2);
    }

    // =========================================================================
    //  VM DETECTION
    // =========================================================================

    private static bool DetectVirtualMachine(ILogger logger)
    {
        // Method 1: Check BIOS/System manufacturer via WMI
        // VMware injects "VMware, Inc." as SystemManufacturer in DMI/SMBIOS
        try
        {
            var psi = new System.Diagnostics.ProcessStartInfo("powershell",
                "-NoProfile -Command \"(Get-CimInstance Win32_ComputerSystem).Manufacturer\"")
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            using var proc = System.Diagnostics.Process.Start(psi);
            if (proc != null)
            {
                proc.WaitForExit(3000);
                var output = proc.StandardOutput.ReadToEnd().Trim().ToLowerInvariant();
                if (output.Contains("vmware") || output.Contains("microsoft") ||
                    output.Contains("xen") || output.Contains("kvm") ||
                    output.Contains("qemu") || output.Contains("virtualbox"))
                {
                    logger.LogInformation("[Hardware] VM detected via WMI manufacturer: {M}", output);
                    return true;
                }
            }
        }
        catch (Exception ex)
        {
            logger.LogWarning("[Hardware] VM detection via WMI failed: {E}", ex.Message);
        }

        // Method 2: Check for VMware-specific registry key
        try
        {
            var key = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"SOFTWARE\VMware, Inc.\VMware Tools");
            if (key != null)
            {
                logger.LogInformation("[Hardware] VM detected via VMware Tools registry key");
                return true;
            }
        }
        catch { /* ignore */ }

        return false;
    }

    // =========================================================================
    //  PHYSICAL CORE DETECTION (Windows, bare metal only)
    //  For VMs this is skipped — we use logical count directly.
    // =========================================================================

    private static int DetectPhysicalCoresWindows(int logicalFallback, ILogger logger)
    {
        try
        {
            var psi = new System.Diagnostics.ProcessStartInfo("powershell",
                "-NoProfile -Command \"(Get-CimInstance Win32_Processor | " +
                "Measure-Object -Property NumberOfCores -Sum).Sum\"")
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            using var proc = System.Diagnostics.Process.Start(psi);
            if (proc != null)
            {
                proc.WaitForExit(3000);
                var output = proc.StandardOutput.ReadToEnd().Trim();
                if (int.TryParse(output, out int n) && n > 0)
                {
                    logger.LogInformation("[Hardware] Physical cores from WMI: {N}", n);
                    return n;
                }
            }
        }
        catch (Exception ex)
        {
            logger.LogWarning("[Hardware] WMI core detection failed: {E}", ex.Message);
        }

        int physical = Math.Max(1, logicalFallback / 2);
        logger.LogInformation("[Hardware] Physical cores fallback (logical/2): {N}", physical);
        return physical;
    }
}