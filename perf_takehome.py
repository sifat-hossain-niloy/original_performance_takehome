"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # Vectorized parallel tree traversal with VLIW instruction scheduling
        # Processes batch_size elements across rounds iterations
        
        # Local helper: allocate VLEN consecutive scratch words
        def vec_alloc(tag=None):
            return self.alloc_scratch(name=tag, length=VLEN)
        
        # Local helper: pack multiple operations into a single cycle
        def pack_instr(operations):
            packed = defaultdict(list)
            for eng, op in operations:
                packed[eng].append(op)
            self.instrs.append(dict(packed))
        
        # VLIW scheduler: arranges operations to maximize throughput
        def vliw_schedule(work_lists):
            pos = [0] * len(work_lists)
            prev_cycle = [-1] * len(work_lists)
            prev_grp = [None] * len(work_lists)
            
            # Pre-calculate distance to memory operations for priority scheduling
            mem_priority = []
            ctrl_priority = []
            for wl in work_lists:
                mp = [0] * len(wl)
                cp = [0] * len(wl)
                next_mem = next_ctrl = None
                for k in range(len(wl) - 1, -1, -1):
                    if wl[k][0] == "load":
                        next_mem = 0
                    elif next_mem is not None:
                        next_mem += 1
                    mp[k] = next_mem if next_mem is not None else 999999
                    if wl[k][0] == "flow":
                        next_ctrl = 0
                    elif next_ctrl is not None:
                        next_ctrl += 1
                    cp[k] = next_ctrl if next_ctrl is not None else 999999
                mem_priority.append(mp)
                ctrl_priority.append(cp)
            
            todo = sum(len(wl) for wl in work_lists)
            tick = 0
            ld_rr = st_rr = 0
            
            def ready(w, grp):
                return prev_cycle[w] < tick or prev_grp[w] == grp
            
            while todo > 0:
                bundle = defaultdict(list)
                
                # Schedule vector ALU operations
                cands = []
                for w, wl in enumerate(work_lists):
                    if pos[w] >= len(wl):
                        continue
                    item = wl[pos[w]]
                    if not ready(w, item[2]) or item[0] != "valu":
                        continue
                    grp = item[2]
                    cnt = 0
                    while pos[w] + cnt < len(wl):
                        nxt = wl[pos[w] + cnt]
                        if nxt[0] != "valu" or nxt[2] != grp:
                            break
                        cnt += 1
                    cands.append((mem_priority[w][pos[w]], ctrl_priority[w][pos[w]], -(tick - prev_cycle[w]), cnt, w))
                
                for _, _, _, _, w in sorted(cands):
                    wl = work_lists[w]
                    grp = wl[pos[w]][2]
                    while len(bundle["valu"]) < SLOT_LIMITS["valu"]:
                        if pos[w] >= len(wl):
                            break
                        item = wl[pos[w]]
                        if item[0] != "valu" or item[2] != grp:
                            break
                        bundle["valu"].append(item[1])
                        pos[w] += 1
                        prev_cycle[w] = tick
                        prev_grp[w] = grp
                        todo -= 1
                    if len(bundle["valu"]) >= SLOT_LIMITS["valu"]:
                        break
                
                # Schedule scalar ALU operations
                cands = []
                for w, wl in enumerate(work_lists):
                    if pos[w] >= len(wl):
                        continue
                    item = wl[pos[w]]
                    if not ready(w, item[2]) or item[0] != "alu":
                        continue
                    grp = item[2]
                    cnt = 0
                    while pos[w] + cnt < len(wl):
                        nxt = wl[pos[w] + cnt]
                        if nxt[0] != "alu" or nxt[2] != grp:
                            break
                        cnt += 1
                    pri = 0 if prev_cycle[w] == tick else 1
                    cands.append((pri, -cnt, w))
                
                for _, _, w in sorted(cands):
                    wl = work_lists[w]
                    grp = wl[pos[w]][2]
                    while len(bundle["alu"]) < SLOT_LIMITS["alu"]:
                        if pos[w] >= len(wl):
                            break
                        item = wl[pos[w]]
                        if item[0] != "alu" or item[2] != grp:
                            break
                        bundle["alu"].append(item[1])
                        pos[w] += 1
                        prev_cycle[w] = tick
                        prev_grp[w] = grp
                        todo -= 1
                    if len(bundle["alu"]) >= SLOT_LIMITS["alu"]:
                        break
                
                # Schedule memory load operations
                ld_pick = None
                for w, wl in enumerate(work_lists):
                    if pos[w] >= len(wl):
                        continue
                    item = wl[pos[w]]
                    if not ready(w, item[2]) or item[0] != "load":
                        continue
                    if prev_cycle[w] == tick:
                        ld_pick = w
                        break
                
                for off in range(len(work_lists)):
                    if ld_pick is not None:
                        break
                    w = (ld_rr + off) % len(work_lists)
                    if pos[w] >= len(work_lists[w]):
                        continue
                    item = work_lists[w][pos[w]]
                    if not ready(w, item[2]) or item[0] != "load":
                        continue
                    ld_pick = w
                    break
                
                if ld_pick is not None:
                    ld_rr = ld_pick
                    wl = work_lists[ld_pick]
                    grp = wl[pos[ld_pick]][2]
                    while len(bundle["load"]) < SLOT_LIMITS["load"]:
                        if pos[ld_pick] >= len(wl):
                            break
                        item = wl[pos[ld_pick]]
                        if item[0] != "load" or item[2] != grp:
                            break
                        bundle["load"].append(item[1])
                        pos[ld_pick] += 1
                        prev_cycle[ld_pick] = tick
                        prev_grp[ld_pick] = grp
                        todo -= 1
                
                if len(bundle["load"]) < SLOT_LIMITS["load"]:
                    for w, wl in enumerate(work_lists):
                        if w == ld_pick or pos[w] >= len(wl):
                            continue
                        item = wl[pos[w]]
                        if not ready(w, item[2]) or item[0] != "load":
                            continue
                        bundle["load"].append(item[1])
                        pos[w] += 1
                        prev_cycle[w] = tick
                        prev_grp[w] = item[2]
                        todo -= 1
                        break
                
                # Schedule memory store operations
                st_pick = None
                for w, wl in enumerate(work_lists):
                    if pos[w] >= len(wl):
                        continue
                    item = wl[pos[w]]
                    if not ready(w, item[2]) or item[0] != "store":
                        continue
                    if prev_cycle[w] == tick:
                        st_pick = w
                        break
                
                for off in range(len(work_lists)):
                    if st_pick is not None:
                        break
                    w = (st_rr + off) % len(work_lists)
                    if pos[w] >= len(work_lists[w]):
                        continue
                    item = work_lists[w][pos[w]]
                    if not ready(w, item[2]) or item[0] != "store":
                        continue
                    st_pick = w
                    break
                
                if st_pick is not None:
                    st_rr = st_pick
                    wl = work_lists[st_pick]
                    grp = wl[pos[st_pick]][2]
                    while len(bundle["store"]) < SLOT_LIMITS["store"]:
                        if pos[st_pick] >= len(wl):
                            break
                        item = wl[pos[st_pick]]
                        if item[0] != "store" or item[2] != grp:
                            break
                        bundle["store"].append(item[1])
                        pos[st_pick] += 1
                        prev_cycle[st_pick] = tick
                        prev_grp[st_pick] = grp
                        todo -= 1
                
                if len(bundle["store"]) < SLOT_LIMITS["store"]:
                    for w, wl in enumerate(work_lists):
                        if w == st_pick or pos[w] >= len(wl):
                            continue
                        item = wl[pos[w]]
                        if not ready(w, item[2]) or item[0] != "store":
                            continue
                        bundle["store"].append(item[1])
                        pos[w] += 1
                        prev_cycle[w] = tick
                        prev_grp[w] = item[2]
                        todo -= 1
                        break
                
                # Schedule control flow operations
                best = None
                for w, wl in enumerate(work_lists):
                    if pos[w] >= len(wl):
                        continue
                    item = wl[pos[w]]
                    if not ready(w, item[2]) or item[0] != "flow":
                        continue
                    score = (mem_priority[w][pos[w]], w)
                    if best is None or score < best[0]:
                        best = (score, w)
                
                if best is not None:
                    w = best[1]
                    item = work_lists[w][pos[w]]
                    bundle["flow"].append(item[1])
                    pos[w] += 1
                    prev_cycle[w] = tick
                    prev_grp[w] = item[2]
                    todo -= 1
                
                if bundle:
                    self.instrs.append(dict(bundle))
                tick += 1
        
        # Setup memory pointers
        tree_ptr = self.alloc_scratch("tree_ptr")
        idx_ptr = self.alloc_scratch("idx_ptr")
        val_ptr = self.alloc_scratch("val_ptr")
        
        tree_base = 7
        idx_base = tree_base + n_nodes
        val_base = idx_base + batch_size
        
        pack_instr([("load", ("const", tree_ptr, tree_base)), ("load", ("const", idx_ptr, idx_base))])
        pack_instr([("load", ("const", val_ptr, val_base))])
        
        num_chunks = batch_size // VLEN
        
        # Allocate working vectors
        idx_vecs = self.alloc_scratch("idx_vecs", batch_size)
        val_vecs = self.alloc_scratch("val_vecs", batch_size)
        scratch_a = self.alloc_scratch("scratch_a", batch_size)
        scratch_b = self.alloc_scratch("scratch_b", batch_size)
        
        setup_work = []
        
        def make_task():
            return {"items": [], "seq": 0}
        
        def task_emit(task, eng, op):
            task["items"].append((eng, op, task["seq"]))
            task["seq"] += 1
        
        def task_emit_batch(task, ops):
            for e, o in ops:
                task["items"].append((e, o, task["seq"]))
            task["seq"] += 1
        
        # Create broadcast constants
        def make_vec_const(value, tag):
            s = self.alloc_scratch(tag)
            v = vec_alloc(f"vec_{tag}")
            t = make_task()
            task_emit(t, "load", ("const", s, value))
            task_emit(t, "valu", ("vbroadcast", v, s))
            setup_work.append(t["items"])
            return v
        
        vec_1 = make_vec_const(1, "c1")
        vec_2 = make_vec_const(2, "c2")
        vec_n_nodes = make_vec_const(n_nodes, "cn")
        tree_base_vec = make_vec_const(tree_base, "ctb")
        
        # Cache tree nodes for depth 0-1
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_val = self.alloc_scratch("tmp_val")
        
        cached_nodes = [None] * 3
        for n in range(3):
            cached_nodes[n] = vec_alloc(f"cn_{n}")
        
        cache_task = make_task()
        for n in range(3):
            task_emit(cache_task, "flow", ("add_imm", tmp_addr, self.scratch["tree_ptr"], n))
            task_emit(cache_task, "load", ("load", tmp_val, tmp_addr))
            task_emit(cache_task, "valu", ("vbroadcast", cached_nodes[n], tmp_val))
        setup_work.append(cache_task["items"])
        
        # Precompute level base addresses for faster indexed loads
        level_bases = [None] * (forest_height + 1)
        level_addr = self.alloc_scratch("level_addr")
        level_task = make_task()
        for d in range(2, forest_height + 1):
            lbase = (1 << d) - 1
            task_emit(level_task, "flow", ("add_imm", level_addr, self.scratch["tree_ptr"], lbase))
            lv = vec_alloc(f"lb_{d}")
            level_bases[d] = lv
            task_emit(level_task, "valu", ("vbroadcast", lv, level_addr))
        setup_work.append(level_task["items"])
        
        # Hash function constants with fused multiply-add optimization
        h_c1 = []
        h_c3 = []
        h_fma = []
        for i, (o1, c1, o2, o3, c3) in enumerate(HASH_STAGES):
            h_c1.append(make_vec_const(c1, f"hc1_{i}"))
            h_c3.append(make_vec_const(c3, f"hc3_{i}"))
            if o1 == "+" and o2 == "+" and o3 == "<<":
                fma_mult = (1 + (1 << c3)) % (2**32)
                h_fma.append(make_vec_const(fma_mult, f"hfma_{i}"))
            else:
                h_fma.append(None)
        
        # Setup per-chunk memory pointers
        chunk_idx_ptr = []
        chunk_val_ptr = []
        for c in range(num_chunks):
            chunk_idx_ptr.append(self.alloc_scratch(f"cip_{c}"))
            chunk_val_ptr.append(self.alloc_scratch(f"cvp_{c}"))
        
        chunk_offsets = self.alloc_scratch("offs", num_chunks)
        ptr_p0 = self.alloc_scratch("pp0")
        ptr_p1 = self.alloc_scratch("pp1")
        ptr_stride = None
        
        ptr_task = make_task()
        
        for c in range(0, num_chunks, 2):
            ops = [("load", ("const", chunk_offsets + c, c * VLEN))]
            if c + 1 < num_chunks:
                ops.append(("load", ("const", chunk_offsets + c + 1, (c + 1) * VLEN)))
            task_emit_batch(ptr_task, ops)
        
        for c in range(0, num_chunks, 12):
            ops = []
            for j in range(c, min(c + 12, num_chunks)):
                ops.append(("alu", ("+", chunk_idx_ptr[j], self.scratch["idx_ptr"], chunk_offsets + j)))
            task_emit_batch(ptr_task, ops)
        
        for c in range(0, num_chunks, 12):
            ops = []
            for j in range(c, min(c + 12, num_chunks)):
                ops.append(("alu", ("+", chunk_val_ptr[j], self.scratch["val_ptr"], chunk_offsets + j)))
            task_emit_batch(ptr_task, ops)
        
        p1_off = chunk_offsets + 1 if num_chunks > 1 else chunk_offsets
        task_emit_batch(ptr_task, [
            ("alu", ("+", ptr_p0, self.scratch["val_ptr"], chunk_offsets)),
            ("alu", ("+", ptr_p1, self.scratch["val_ptr"], p1_off)),
        ])
        
        if num_chunks >= 3:
            ptr_stride = chunk_offsets + 2
        else:
            ptr_stride = self.alloc_scratch("stride_val")
            task_emit(ptr_task, "load", ("const", ptr_stride, 2 * VLEN))
        
        for c in range(0, num_chunks, 2):
            ops = [("valu", ("^", idx_vecs + c * VLEN, idx_vecs + c * VLEN, idx_vecs + c * VLEN))]
            if c + 1 < num_chunks:
                ops.append(("valu", ("^", idx_vecs + (c + 1) * VLEN, idx_vecs + (c + 1) * VLEN, idx_vecs + (c + 1) * VLEN)))
            ops.append(("alu", ("+", ptr_p0, ptr_p0, ptr_stride)))
            ops.append(("alu", ("+", ptr_p1, ptr_p1, ptr_stride)))
            ops.append(("load", ("vload", val_vecs + c * VLEN, ptr_p0)))
            if c + 1 < num_chunks:
                ops.append(("load", ("vload", val_vecs + (c + 1) * VLEN, ptr_p1)))
            task_emit_batch(ptr_task, ops)
        
        setup_work.append(ptr_task["items"])
        vliw_schedule(setup_work)
        
        self.add("flow", ("pause",))
        
        # Main computation: process each chunk
        main_work = [[] for _ in range(num_chunks)]
        for c in range(num_chunks):
            iv = idx_vecs + c * VLEN
            vv = val_vecs + c * VLEN
            sa = scratch_a + c * VLEN
            sb = scratch_b + c * VLEN
            work = main_work[c]
            step = 0
            
            def emit(eng, op, w=work):
                nonlocal step
                w.append((eng, op, step))
                step += 1
            
            def emit_par(ops, w=work):
                nonlocal step
                for e, o in ops:
                    w.append((e, o, step))
                step += 1
            
            tree_cycle = forest_height + 1
            for r in range(rounds):
                depth = r % tree_cycle
                
                # Load tree node value - use cache for early depths
                if depth == 0:
                    # All indices are 0 at depth 0
                    emit("valu", ("^", vv, vv, cached_nodes[0]))
                elif depth == 1:
                    # Indices are 1 or 2: use vselect on (idx & 1)
                    # idx=1: bit0=1 -> want node[1], idx=2: bit0=0 -> want node[2]
                    emit("valu", ("&", sa, iv, vec_1))
                    emit("flow", ("vselect", sb, sa, cached_nodes[1], cached_nodes[2]))
                    emit("valu", ("^", vv, vv, sb))
                else:
                    # Generic indexed load for deeper levels
                    emit("valu", ("+", sa, iv, tree_base_vec))
                    for lane in range(0, VLEN, 2):
                        ld_ops = [("load", ("load_offset", sb, sa, lane))]
                        if lane + 1 < VLEN:
                            ld_ops.append(("load", ("load_offset", sb, sa, lane + 1)))
                        emit_par(ld_ops)
                    emit("valu", ("^", vv, vv, sb))
                
                # Apply hash function
                for hi, (o1, _, o2, o3, _) in enumerate(HASH_STAGES):
                    if h_fma[hi] is not None:
                        emit("valu", ("multiply_add", vv, vv, h_fma[hi], h_c1[hi]))
                        continue
                    emit_par([("valu", (o1, sa, vv, h_c1[hi])), ("valu", (o3, sb, vv, h_c3[hi]))])
                    emit("valu", (o2, vv, sa, sb))
                
                # Compute next index: idx = 2*idx + 1 + (val & 1)
                emit("valu", ("&", sa, vv, vec_1))
                emit("valu", ("+", sa, sa, vec_1))
                emit("valu", ("multiply_add", iv, iv, vec_2, sa))
                
                # Wrap around: idx = 0 if idx >= n_nodes
                emit("valu", ("<", sa, iv, vec_n_nodes))
                emit("valu", ("*", iv, iv, sa))
            
            # Store final values and indices
            emit_par([("store", ("vstore", chunk_val_ptr[c], vv)), ("store", ("vstore", chunk_idx_ptr[c], iv))])
        
        vliw_schedule(main_work)
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
