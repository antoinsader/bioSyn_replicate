import os, time, csv, json, logging, psutil, torch



class MetricsLogger:
    def __init__(self, use_cuda, logger, tag="train"):
        self.use_cuda = bool(use_cuda and torch.cuda.is_available())
        self.proc = psutil.Process(os.getpid())
        self.start_ts = None
        self.device = torch.device("cuda") if self.use_cuda else None
        self.max_rss_mb =0.0
        self.logger = logger
        self.tag =tag
    def rss_mb(self):
        rss = self.proc.memory_info().rss / (1024 ** 2)
        self.max_rss_mb = max(self.max_rss_mb, rss)
        return rss
    def gpu_numbers(self):
        if not self.use_cuda:
            return (None, None, None, None)
        torch.cuda.synchronize(self.device)
        alloc = torch.cuda.memory_allocated(self.device) / (1024**2)
        alloc_peak = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        res = torch.cuda.memory_reserved(self.device) / (1024**2)
        res_peak = torch.cuda.max_memory_reserved(self.device) / (1024**2)
        return (alloc, alloc_peak, res, res_peak)
    def start_run(self):
        self.start_ts = time.time()
        if self.use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self.logger.info(f"[{self.tag}] START")

    def log_event(self, event, epoch, loss, t0):
        if self.start_ts is None:
            self.start_ts = time.time()
        wall = (time.time() - (t0 if t0 is not None else self.start_ts))
        rss = self.rss_mb()
        g_alloc, g_peak, g_res, g_res_peak  = self.gpu_numbers()
        msg = f"[{self.tag}] {event}"
        if epoch is not None:
            msg += f" | epoch={epoch}"
        if loss is not None:
            msg += f" | loss={loss:.6f}"
        msg += f" | CPU_RSS={rss:.1f} MB"
        if self.use_cuda:
            msg += f"| GPU_alloc={g_alloc:.1f} MB (peak {g_peak:.1f}) | GPU reserved {g_res:.1f} MB (peak: {g_res_peak: .1f})" 
        msg += f" | elapsed={wall:.2f}s"
        self.logger.info(msg)

    def end_run(self):
        total = time.time() - self.start_ts if self.start_ts else 0.0
         # read final GPU peaks (since start_run)
        _, g_peak, _, g_res_peak = self._gpu_numbers()
        summary = (f"[{self.tag}] END | total={total:.2f}s  | CPU_peak_RSS={self.max_rss_mb:.1f} MB")
        if self.use_cuda:
            summary += (f" | GPU_peak_alloc={g_peak:.1f} MB "
                        f"| GPU_peak_reserved={g_res_peak:.1f} MB")
        self.logger.info(summary)