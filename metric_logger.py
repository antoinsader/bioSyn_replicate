import os, time, csv, json, logging, psutil, torch

class METRIC_EVENT:
    pass
# FOR FUTURE

class MetricsLogger:
    def __init__(self, use_cuda, logger, tag="train"):
        self.use_cuda = bool(use_cuda and torch.cuda.is_available())
        self.proc = psutil.Process(os.getpid())
        self.start_ts = None
        self.device = torch.device("cuda") if self.use_cuda else None
        self.max_rss_mb =0.0
        self.logger = logger
        self.tag =tag
        self.messages = []
        self.one_time_events_set = set()
    def _rss_mb(self):
        rss = self.proc.memory_info().rss / (1024 ** 2)
        self.max_rss_mb = max(self.max_rss_mb, rss)
        return rss
    def _gpu_numbers(self):
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
        self.logger.info(f"\n[{self.tag}] START")


    def show_gpu_memory(self, event):
        free = torch.cuda.mem_get_info()[0] / 1024**2   # free memory in MB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        self.logger.info(f"\n[{event}]: GPU free={free:.1f} MB / total={total:.1f} MB")


    def log_event(self, event, epoch=None, loss=None, t0= None, log_immediate=True, first_iteration_only=True, only_elapsed_time=False):
        if first_iteration_only and event in self.one_time_events_set:
            return True

        self.one_time_events_set.add(event)
        if self.start_ts is None:
            self.start_ts = time.time()

        wall = (time.time() - (t0 if t0 is not None else self.start_ts))

        msg = f"[{self.tag}] {event}"
        msg += f" | elapsed={wall:.5f}s({'t0 is calculated' if t0 else "t0 not calculated"})"
        if epoch is not None:
            msg += f" | epoch={epoch}"

        if only_elapsed_time:
            if log_immediate:
                self.logger.info(f"\n{msg}")
            else:
                self.messages.append(f"{msg}")
            
            return


        rss = self._rss_mb()
        g_alloc, g_peak, g_res, g_res_peak  = self._gpu_numbers()
        

        
        if loss is not None:
            msg += f" | loss={loss:.6f}"
        
        msg += f" | CPU_RSS={rss:.1f} MB"
        if self.use_cuda:
            msg += f"| GPU_alloc={g_alloc:.1f} MB (peak {g_peak:.1f}) | GPU reserved {g_res:.1f} MB (peak: {g_res_peak: .1f})" 
        
        if log_immediate:
            self.logger.info(f"\n{msg}")
        else:
            self.messages.append(f"{msg}")


    def end_run(self):
        total = time.time() - self.start_ts if self.start_ts else 0.0
        self.logger.info(f"\n\n***************ONE TIME MESSAGES:***************\n\n")
        for m in self.messages:
            self.logger.info(f"\n{m}")
        self.logger.info(f"\n\n***************END ONE TIME MESSAGES:***************\n\n")
        
         # read final GPU peaks (since start_run)
        _, g_peak, _, g_res_peak = self._gpu_numbers()
        summary = (f"[{self.tag}] END | total={total:.2f}s  | CPU_peak_RSS={self.max_rss_mb:.1f} MB")
        if self.use_cuda:
            summary += (f" | GPU_peak_alloc={g_peak:.1f} MB "
                        f"| GPU_peak_reserved={g_res_peak:.1f} MB")
        self.logger.info(summary)