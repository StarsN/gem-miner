
import ctypes, time, sys, os, argparse, struct

HERE = os.path.dirname(os.path.abspath(__file__))
def _load_opencl():
    """Load the system OpenCL runtime on Windows / Linux / macOS."""
    if sys.platform.startswith("win"):
        candidates = [("OpenCL.dll", ctypes.WinDLL)]
    elif sys.platform == "darwin":
        candidates = [("/System/Library/Frameworks/OpenCL.framework/OpenCL", ctypes.CDLL)]
    else:
        candidates = [("libOpenCL.so.1", ctypes.CDLL), ("libOpenCL.so", ctypes.CDLL)]
    err = None
    for name, loader in candidates:
        try:
            return loader(name)
        except OSError as exc:
            err = exc
    raise OSError(
        "OpenCL runtime not found (tried %s). On Debian/Ubuntu: "
        "apt install -y ocl-icd-libopencl1 clinfo" % [c[0] for c in candidates]
    ) from err


CL = _load_opencl()

CL_DEVICE_TYPE_GPU = 4
CL_MEM_READ_ONLY = 1 << 2
CL_MEM_WRITE_ONLY = 1 << 1
CL_MEM_COPY_HOST_PTR = 1 << 5
CL_TRUE = 1

def _setup():
    CL.clGetPlatformIDs.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint)]
    CL.clGetDeviceIDs.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint)]
    CL.clCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    CL.clCreateContext.restype = ctypes.c_void_p
    CL.clCreateCommandQueue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_int)]
    CL.clCreateCommandQueue.restype = ctypes.c_void_p
    CL.clCreateProgramWithSource.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int)]
    CL.clCreateProgramWithSource.restype = ctypes.c_void_p
    CL.clBuildProgram.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p]
    CL.clGetProgramBuildInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    CL.clCreateKernel.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    CL.clCreateKernel.restype = ctypes.c_void_p
    CL.clCreateBuffer.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    CL.clCreateBuffer.restype = ctypes.c_void_p
    CL.clSetKernelArg.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p]
    CL.clEnqueueNDRangeKernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    CL.clEnqueueReadBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    CL.clEnqueueWriteBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    CL.clFinish.argtypes = [ctypes.c_void_p]
    CL.clReleaseMemObject.argtypes = [ctypes.c_void_p]

_setup()

class GPU:
    def __init__(self, src_path):
        n = ctypes.c_uint(0)
        assert CL.clGetPlatformIDs(0, None, ctypes.byref(n)) == 0 and n.value > 0, "no OpenCL platform"
        plates = (ctypes.c_void_p * n.value)()
        CL.clGetPlatformIDs(n.value, plates, None)
        self.dev = None
        self.plat = None
        for p in plates:
            dn = ctypes.c_uint(0)
            if CL.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(dn)) == 0 and dn.value > 0:
                devs = (ctypes.c_void_p * dn.value)()
                CL.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, dn.value, devs, None)
                self.plat, self.dev = ctypes.c_void_p(p), ctypes.c_void_p(devs[0])
                break
        assert self.dev, "no GPU device"
        err = ctypes.c_int(0)
        self.ctx = CL.clCreateContext(None, 1, ctypes.byref(self.dev), None, None, ctypes.byref(err))
        assert err.value == 0, "context error %d" % err.value
        self.q = CL.clCreateCommandQueue(self.ctx, self.dev, 0, ctypes.byref(err))
        assert err.value == 0, "queue error %d" % err.value

        src = open(src_path, "rb").read()
        csrc = ctypes.c_char_p(src)
        self.prog = CL.clCreateProgramWithSource(self.ctx, 1, ctypes.byref(csrc), None, ctypes.byref(err))
        assert err.value == 0, "program error %d" % err.value
        st = CL.clBuildProgram(self.prog, 1, ctypes.byref(self.dev), None, None, None)
        if st != 0:
            size = ctypes.c_size_t(0)
            CL.clGetProgramBuildInfo(self.prog, self.dev, 0x1183, 0, None, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value + 1)
            CL.clGetProgramBuildInfo(self.prog, self.dev, 0x1183, size.value, log, None)
            raise RuntimeError("build failed:\n" + log.value.decode(errors="replace"))
        self.kern = CL.clCreateKernel(self.prog, b"grind", ctypes.byref(err))
        assert err.value == 0, "kernel error %d" % err.value

    def new_buffer(self, size, host_ptr=None, flags=CL_MEM_READ_ONLY):
        err = ctypes.c_int(0)
        if host_ptr is not None:
            flags |= CL_MEM_COPY_HOST_PTR
        b = CL.clCreateBuffer(self.ctx, flags, size, host_ptr, ctypes.byref(err))
        assert err.value == 0, "buffer error %d" % err.value
        return b

    def hash_words(self, seed: bytes, nonce: int):
        err = ctypes.c_int(0)
        k = CL.clCreateKernel(self.prog, b"hashwords", ctypes.byref(err))
        assert err.value == 0, "hashwords kernel missing"
        out = (ctypes.c_ulonglong * 4)()
        sb = self.new_buffer(64, ctypes.c_char_p(seed))
        ob = self.new_buffer(32, ctypes.cast(out, ctypes.c_void_p), flags=1)
        n = ctypes.c_ulonglong(nonce)
        CL.clSetKernelArg(k, 0, 8, ctypes.byref(ctypes.c_void_p(sb)))
        CL.clSetKernelArg(k, 1, 8, ctypes.byref(n))
        CL.clSetKernelArg(k, 2, 8, ctypes.byref(ctypes.c_void_p(ob)))
        gs = ctypes.c_size_t(1)
        CL.clEnqueueNDRangeKernel(self.q, k, 1, None, ctypes.byref(gs), None, 0, None, None)
        CL.clFinish(self.q)
        CL.clEnqueueReadBuffer(self.q, ob, CL_TRUE, 0, 32, ctypes.byref(out), 0, None, None)
        return [int(out[i]) for i in range(4)]

    def run(self, seed: bytes, base: int, count: int, target_words, max_found=8):
        assert len(seed) == 32
        found_count = (ctypes.c_uint * 1)(0)
        found_nonces = (ctypes.c_ulonglong * max_found)()
        target_arr = (ctypes.c_ulonglong * 4)(*target_words)

        seed_buf = self.new_buffer(32, ctypes.c_char_p(seed))
        tgt_buf = self.new_buffer(32, ctypes.cast(target_arr, ctypes.c_void_p))
        cnt_buf = self.new_buffer(4, ctypes.cast(found_count, ctypes.c_void_p), flags=1)  # READ_WRITE
        non_buf = self.new_buffer(max_found * 8, ctypes.cast(found_nonces, ctypes.c_void_p), flags=1)

        p_base = ctypes.c_ulonglong(base)
        p_count = ctypes.c_ulonglong(count)
        p_max = ctypes.c_uint(max_found)

        CL.clSetKernelArg(self.kern, 0, 8, ctypes.byref(ctypes.c_void_p(seed_buf)))
        CL.clSetKernelArg(self.kern, 1, 8, ctypes.byref(p_base))
        CL.clSetKernelArg(self.kern, 2, 8, ctypes.byref(p_count))
        CL.clSetKernelArg(self.kern, 3, 8, ctypes.byref(ctypes.c_void_p(tgt_buf)))
        CL.clSetKernelArg(self.kern, 4, 8, ctypes.byref(ctypes.c_void_p(cnt_buf)))
        CL.clSetKernelArg(self.kern, 5, 8, ctypes.byref(ctypes.c_void_p(non_buf)))
        CL.clSetKernelArg(self.kern, 6, 4, ctypes.byref(p_max))

        local = 256
        gsize = ctypes.c_size_t(((count + local - 1) // local) * local)
        lsize = ctypes.c_size_t(local)
        st = CL.clEnqueueNDRangeKernel(self.q, self.kern, 1, None, ctypes.byref(gsize), ctypes.byref(lsize), 0, None, None)
        assert st == 0, "enqueue error %d" % st
        CL.clFinish(self.q)

        n = ctypes.c_uint(0)
        CL.clEnqueueReadBuffer(self.q, cnt_buf, CL_TRUE, 0, 4, ctypes.byref(n), 0, None, None)
        out = []
        if n.value > 0:
            CL.clEnqueueReadBuffer(self.q, non_buf, CL_TRUE, 0, max_found * 8, ctypes.byref(found_nonces), 0, None, None)
            out = [int(found_nonces[i]) for i in range(min(n.value, max_found))]
        for b in (seed_buf, tgt_buf, cnt_buf, non_buf):
            CL.clReleaseMemObject(ctypes.c_void_p(b))
        return out


def words_of(hexstr):
    v = int(hexstr.replace("0x", ""), 16) & ((1 << 256) - 1)
    return [(v >> (64 * (3 - i))) & ((1 << 64) - 1) for i in range(4)]




RESET = (ctypes.c_uint * 1)(0)


def pad64(hexstr):
    return hexstr.replace("0x", "").lower().rjust(64, "0")


class Job:
    def __init__(self, gpu, address, seed_hex, target_hex, ref="0x" + "0" * 40):
        self.gpu = gpu
        self.address = address
        self.seed = bytes.fromhex(seed_hex.replace("0x", ""))
        assert len(self.seed) == 64, "prefix must be 64 bytes (agentId||seed)"
        self.target = words_of(target_hex)
        self.target_int = int(target_hex.replace("0x", ""), 16)
        self.ref = ref
        self.seed_buf = gpu.new_buffer(64, ctypes.c_char_p(self.seed))
        tgt_arr = (ctypes.c_ulonglong * 4)(*self.target)
        self.tgt_buf = gpu.new_buffer(32, ctypes.cast(tgt_arr, ctypes.c_void_p))
        self.cnt = (ctypes.c_uint * 1)(0)
        self.cnt_buf = gpu.new_buffer(4, ctypes.cast(self.cnt, ctypes.c_void_p), flags=1)
        self.nonces = (ctypes.c_ulonglong * 8)()
        self.nonce_buf = gpu.new_buffer(64, ctypes.cast(self.nonces, ctypes.c_void_p), flags=1)
        p_max = ctypes.c_uint(8)
        CL.clSetKernelArg(gpu.kern, 0, 8, ctypes.byref(ctypes.c_void_p(self.seed_buf)))
        CL.clSetKernelArg(gpu.kern, 3, 8, ctypes.byref(ctypes.c_void_p(self.tgt_buf)))
        CL.clSetKernelArg(gpu.kern, 4, 8, ctypes.byref(ctypes.c_void_p(self.cnt_buf)))
        CL.clSetKernelArg(gpu.kern, 5, 8, ctypes.byref(ctypes.c_void_p(self.nonce_buf)))
        CL.clSetKernelArg(gpu.kern, 6, 4, ctypes.byref(p_max))
        self._p_max = p_max

    def expected_hashes(self):
        return (1 << 256) / (self.target_int + 1)

    def scan(self, base, count, local=256):
        CL.clEnqueueWriteBuffer(self.gpu.q, self.cnt_buf, CL_TRUE, 0, 4, ctypes.byref(ctypes.c_uint(0)), 0, None, None)
        p_base = ctypes.c_ulonglong(base)
        p_count = ctypes.c_ulonglong(count)
        CL.clSetKernelArg(self.gpu.kern, 1, 8, ctypes.byref(p_base))
        CL.clSetKernelArg(self.gpu.kern, 2, 8, ctypes.byref(p_count))
        gsize = ctypes.c_size_t(((count + local - 1) // local) * local)
        lsize = ctypes.c_size_t(local)
        st = CL.clEnqueueNDRangeKernel(self.gpu.q, self.gpu.kern, 1, None, ctypes.byref(gsize), ctypes.byref(lsize), 0, None, None)
        assert st == 0, "enqueue error %d" % st
        CL.clFinish(self.gpu.q)
        CL.clEnqueueReadBuffer(self.gpu.q, self.cnt_buf, CL_TRUE, 0, 4, ctypes.byref(self.cnt), 0, None, None)
        n = int(self.cnt[0])
        if n == 0:
            return []
        CL.clEnqueueReadBuffer(self.gpu.q, self.nonce_buf, CL_TRUE, 0, 64, ctypes.byref(self.nonces), 0, None, None)
        return [int(self.nonces[i]) for i in range(min(n, 8))]

    def mint_calldata(self, nonce):
        return "0x96d481c9" + pad64(hex(nonce)) + pad64(self.ref)


def mine_job(job, batch=1 << 24, start=0, quiet=False, max_hashes=None):
    base = start
    total = 0
    t0 = time.time()
    last = t0
    while True:
        found = job.scan(base, batch)
        total += batch
        base += batch
        if found:
            dt = time.time() - t0
            return found, total, dt
        if not quiet and time.time() - last > 5:
            last = time.time()
            rate = total / (time.time() - t0)
            left = (job.expected_hashes() - total) / rate if rate else 0
            print("  %s  %.1f MH/s  %.2fB/%s  ETA %.0fs" % (
                job.address[:10], rate / 1e6, total / 1e9,
                ("%.2fB" % (job.expected_hashes() / 1e9)) if job.expected_hashes() > 1e9 else "%.2fM" % (job.expected_hashes() / 1e6),
                left), flush=True)
        if max_hashes and total >= max_hashes:
            return [], total, time.time() - t0


VECTORS = []  # set by mine.py


def selftest(gpu, vectors=None):
    vectors = vectors or VECTORS
    ok = True
    for prefix_hex, nonce, expect in vectors:
        got = gpu.hash_words(bytes.fromhex(prefix_hex), nonce)
        got_hex = "".join("%016x" % w for w in got)
        good = got_hex == expect
        ok = ok and good
        print(("  PASS " if good else "  FAIL ") + "nonce=%d -> %s..." % (nonce, got_hex[:16]))
    print("selftest:", "OK" if ok else "FAILED")
    return ok


def rate_test(gpu, seconds=8, prefix_hex=None, target_hex=None):
    job = Job(gpu, "bench", prefix_hex, target_hex)
    batch = 1 << 24
    total = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        job.scan(total, batch)
        total += batch
    dt = time.time() - t0
    print("hashrate: %.1f MH/s (%d hashes in %.1fs)" % (total / dt / 1e6, total, dt))
