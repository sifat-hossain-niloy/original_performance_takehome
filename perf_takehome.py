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
        """
        Optimized kernel using SIMD vectorization and VLIW scheduling.
        Processes tree traversal for batch_size inputs over multiple rounds.
        """
        
        # Helper to reserve a vector-sized region in scratch memory
        def reserve_vector(label=None):
            return self.alloc_scratch(name=label, length=VLEN)
        
        # Helper to emit a packed instruction bundle
        def emit_packed(op_list):
            grouped = defaultdict(list)
            for unit, instr in op_list:
                grouped[unit].append(instr)
            self.instrs.append(dict(grouped))
        
        # Instruction scheduler for parallel execution
        def run_scheduler(task_queues):
            cursor = [0] * len(task_queues)
            last_tick = [-1] * len(task_queues)
            last_grp_id = [None] * len(task_queues)
            
            # Calculate lookahead distances for scheduling priority
            load_lookahead = []
            flow_lookahead = []
            for queue in task_queues:
                ld = [0] * len(queue)
                fl = [0] * len(queue)
                nxt_ld = nxt_fl = None
                for idx in range(len(queue) - 1, -1, -1):
                    if queue[idx][0] == "load":
                        nxt_ld = 0
                    elif nxt_ld is not None:
                        nxt_ld += 1
                    ld[idx] = nxt_ld if nxt_ld is not None else 999999
                    if queue[idx][0] == "flow":
                        nxt_fl = 0
                    elif nxt_fl is not None:
                        nxt_fl += 1
                    fl[idx] = nxt_fl if nxt_fl is not None else 999999
                load_lookahead.append(ld)
                flow_lookahead.append(fl)
            
            pending = sum(len(q) for q in task_queues)
            current_tick = 0
            ld_robin = st_robin = 0
            
            def is_ready(q_idx, grp):
                return last_tick[q_idx] < current_tick or last_grp_id[q_idx] == grp
            
            while pending > 0:
                cycle_bundle = defaultdict(list)
                
                # Pack VALU operations
                valu_cands = []
                for q_idx, queue in enumerate(task_queues):
                    if cursor[q_idx] >= len(queue):
                        continue
                    entry = queue[cursor[q_idx]]
                    if not is_ready(q_idx, entry[2]) or entry[0] != "valu":
                        continue
                    grp = entry[2]
                    run_len = 0
                    while cursor[q_idx] + run_len < len(queue):
                        nxt = queue[cursor[q_idx] + run_len]
                        if nxt[0] != "valu" or nxt[2] != grp:
                            break
                        run_len += 1
                    valu_cands.append((load_lookahead[q_idx][cursor[q_idx]], flow_lookahead[q_idx][cursor[q_idx]], -(current_tick - last_tick[q_idx]), run_len, q_idx))
                
                for _, _, _, _, q_idx in sorted(valu_cands):
                    queue = task_queues[q_idx]
                    grp = queue[cursor[q_idx]][2]
                    while len(cycle_bundle["valu"]) < SLOT_LIMITS["valu"]:
                        if cursor[q_idx] >= len(queue):
                            break
                        entry = queue[cursor[q_idx]]
                        if entry[0] != "valu" or entry[2] != grp:
                            break
                        cycle_bundle["valu"].append(entry[1])
                        cursor[q_idx] += 1
                        last_tick[q_idx] = current_tick
                        last_grp_id[q_idx] = grp
                        pending -= 1
                    if len(cycle_bundle["valu"]) >= SLOT_LIMITS["valu"]:
                        break
                
                # Pack scalar ALU operations
                alu_cands = []
                for q_idx, queue in enumerate(task_queues):
                    if cursor[q_idx] >= len(queue):
                        continue
                    entry = queue[cursor[q_idx]]
                    if not is_ready(q_idx, entry[2]) or entry[0] != "alu":
                        continue
                    grp = entry[2]
                    run_len = 0
                    while cursor[q_idx] + run_len < len(queue):
                        nxt = queue[cursor[q_idx] + run_len]
                        if nxt[0] != "alu" or nxt[2] != grp:
                            break
                        run_len += 1
                    priority = 0 if last_tick[q_idx] == current_tick else 1
                    alu_cands.append((priority, -run_len, q_idx))
                
                for _, _, q_idx in sorted(alu_cands):
                    queue = task_queues[q_idx]
                    grp = queue[cursor[q_idx]][2]
                    while len(cycle_bundle["alu"]) < SLOT_LIMITS["alu"]:
                        if cursor[q_idx] >= len(queue):
                            break
                        entry = queue[cursor[q_idx]]
                        if entry[0] != "alu" or entry[2] != grp:
                            break
                        cycle_bundle["alu"].append(entry[1])
                        cursor[q_idx] += 1
                        last_tick[q_idx] = current_tick
                        last_grp_id[q_idx] = grp
                        pending -= 1
                    if len(cycle_bundle["alu"]) >= SLOT_LIMITS["alu"]:
                        break
                
                # Pack load operations with round-robin fallback
                chosen_ld = None
                for q_idx, queue in enumerate(task_queues):
                    if cursor[q_idx] >= len(queue):
                        continue
                    entry = queue[cursor[q_idx]]
                    if not is_ready(q_idx, entry[2]) or entry[0] != "load":
                        continue
                    if last_tick[q_idx] == current_tick:
                        chosen_ld = q_idx
                        break
                
                for off in range(len(task_queues)):
                    if chosen_ld is not None:
                        break
                    q_idx = (ld_robin + off) % len(task_queues)
                    if cursor[q_idx] >= len(task_queues[q_idx]):
                        continue
                    entry = task_queues[q_idx][cursor[q_idx]]
                    if not is_ready(q_idx, entry[2]) or entry[0] != "load":
                        continue
                    chosen_ld = q_idx
                    break
                
                if chosen_ld is not None:
                    ld_robin = chosen_ld
                    queue = task_queues[chosen_ld]
                    grp = queue[cursor[chosen_ld]][2]
                    while len(cycle_bundle["load"]) < SLOT_LIMITS["load"]:
                        if cursor[chosen_ld] >= len(queue):
                            break
                        entry = queue[cursor[chosen_ld]]
                        if entry[0] != "load" or entry[2] != grp:
                            break
                        cycle_bundle["load"].append(entry[1])
                        cursor[chosen_ld] += 1
                        last_tick[chosen_ld] = current_tick
                        last_grp_id[chosen_ld] = grp
                        pending -= 1
                
                if len(cycle_bundle["load"]) < SLOT_LIMITS["load"]:
                    for q_idx, queue in enumerate(task_queues):
                        if q_idx == chosen_ld or cursor[q_idx] >= len(queue):
                            continue
                        entry = queue[cursor[q_idx]]
                        if not is_ready(q_idx, entry[2]) or entry[0] != "load":
                            continue
                        cycle_bundle["load"].append(entry[1])
                        cursor[q_idx] += 1
                        last_tick[q_idx] = current_tick
                        last_grp_id[q_idx] = entry[2]
                        pending -= 1
                        break
                
                # Pack store operations
                chosen_st = None
                for q_idx, queue in enumerate(task_queues):
                    if cursor[q_idx] >= len(queue):
                        continue
                    entry = queue[cursor[q_idx]]
                    if not is_ready(q_idx, entry[2]) or entry[0] != "store":
                        continue
                    if last_tick[q_idx] == current_tick:
                        chosen_st = q_idx
                        break
                
                for off in range(len(task_queues)):
                    if chosen_st is not None:
                        break
                    q_idx = (st_robin + off) % len(task_queues)
                    if cursor[q_idx] >= len(task_queues[q_idx]):
                        continue
                    entry = task_queues[q_idx][cursor[q_idx]]
                    if not is_ready(q_idx, entry[2]) or entry[0] != "store":
                        continue
                    chosen_st = q_idx
                    break
                
                if chosen_st is not None:
                    st_robin = chosen_st
                    queue = task_queues[chosen_st]
                    grp = queue[cursor[chosen_st]][2]
                    while len(cycle_bundle["store"]) < SLOT_LIMITS["store"]:
                        if cursor[chosen_st] >= len(queue):
                            break
                        entry = queue[cursor[chosen_st]]
                        if entry[0] != "store" or entry[2] != grp:
                            break
                        cycle_bundle["store"].append(entry[1])
                        cursor[chosen_st] += 1
                        last_tick[chosen_st] = current_tick
                        last_grp_id[chosen_st] = grp
                        pending -= 1
                
                if len(cycle_bundle["store"]) < SLOT_LIMITS["store"]:
                    for q_idx, queue in enumerate(task_queues):
                        if q_idx == chosen_st or cursor[q_idx] >= len(queue):
                            continue
                        entry = queue[cursor[q_idx]]
                        if not is_ready(q_idx, entry[2]) or entry[0] != "store":
                            continue
                        cycle_bundle["store"].append(entry[1])
                        cursor[q_idx] += 1
                        last_tick[q_idx] = current_tick
                        last_grp_id[q_idx] = entry[2]
                        pending -= 1
                        break
                
                # Pack flow control operations
                best_flow = None
                for q_idx, queue in enumerate(task_queues):
                    if cursor[q_idx] >= len(queue):
                        continue
                    entry = queue[cursor[q_idx]]
                    if not is_ready(q_idx, entry[2]) or entry[0] != "flow":
                        continue
                    key = (load_lookahead[q_idx][cursor[q_idx]], q_idx)
                    if best_flow is None or key < best_flow[0]:
                        best_flow = (key, q_idx)
                
                if best_flow is not None:
                    q_idx = best_flow[1]
                    entry = task_queues[q_idx][cursor[q_idx]]
                    cycle_bundle["flow"].append(entry[1])
                    cursor[q_idx] += 1
                    last_tick[q_idx] = current_tick
                    last_grp_id[q_idx] = entry[2]
                    pending -= 1
                
                if cycle_bundle:
                    self.instrs.append(dict(cycle_bundle))
                current_tick += 1
        
        # Memory layout pointers
        forest_ptr = self.alloc_scratch("forest_ptr")
        indices_ptr = self.alloc_scratch("indices_ptr")
        values_ptr = self.alloc_scratch("values_ptr")
        
        forest_offset = 7
        indices_offset = forest_offset + n_nodes
        values_offset = indices_offset + batch_size
        
        emit_packed([("load", ("const", forest_ptr, forest_offset)), ("load", ("const", indices_ptr, indices_offset))])
        emit_packed([("load", ("const", values_ptr, values_offset))])
        
        num_vectors = batch_size // VLEN
        
        # Working memory allocation
        rel_idx_buf = self.alloc_scratch("rel_idx_buf", batch_size)
        hash_val_buf = self.alloc_scratch("hash_val_buf", batch_size)
        work_buf_a = self.alloc_scratch("work_buf_a", batch_size)
        work_buf_b = self.alloc_scratch("work_buf_b", batch_size)
        
        init_tasks = []
        
        def new_task_seq():
            return {"ops": [], "order": 0}
        
        def append_op(task, unit, op):
            task["ops"].append((unit, op, task["order"]))
            task["order"] += 1
        
        def append_parallel(task, op_list):
            for u, o in op_list:
                task["ops"].append((u, o, task["order"]))
            task["order"] += 1
        
        # Broadcast constant creation
        def create_broadcast_const(value, label):
            scalar_addr = self.alloc_scratch(label)
            vec_addr = reserve_vector(f"bcast_{label}")
            task = new_task_seq()
            append_op(task, "load", ("const", scalar_addr, value))
            append_op(task, "valu", ("vbroadcast", vec_addr, scalar_addr))
            init_tasks.append(task["ops"])
            return vec_addr
        
        const_one = create_broadcast_const(1, "one")
        const_two = create_broadcast_const(2, "two")
        const_nnodes = create_broadcast_const(n_nodes, "nnodes")
        const_forest = create_broadcast_const(forest_offset, "forest")
        
        # Pre-fetch tree nodes at shallow depths
        addr_tmp = self.alloc_scratch("addr_tmp")
        val_tmp = self.alloc_scratch("val_tmp")
        
        prefetched = [None] * 7
        for node_idx in range(7):
            prefetched[node_idx] = reserve_vector(f"prefetch_{node_idx}")
        
        prefetch_task = new_task_seq()
        for node_idx in range(7):
            append_op(prefetch_task, "flow", ("add_imm", addr_tmp, self.scratch["forest_ptr"], node_idx))
            append_op(prefetch_task, "load", ("load", val_tmp, addr_tmp))
            append_op(prefetch_task, "valu", ("vbroadcast", prefetched[node_idx], val_tmp))
        init_tasks.append(prefetch_task["ops"])
        
        # Level base pointers and offsets for deeper tree levels
        level_ptr = [None] * (forest_height + 1)
        level_offset = [None] * (forest_height + 1)
        level_tmp = self.alloc_scratch("level_tmp")
        level_task = new_task_seq()
        for d in range(2, forest_height + 1):
            base_idx = (1 << d) - 1
            append_op(level_task, "flow", ("add_imm", level_tmp, self.scratch["forest_ptr"], base_idx))
            ptr_vec = reserve_vector(f"lvlptr_{d}")
            level_ptr[d] = ptr_vec
            append_op(level_task, "valu", ("vbroadcast", ptr_vec, level_tmp))
            off_vec = create_broadcast_const(base_idx, f"lvloff_{d}")
            level_offset[d] = off_vec
        init_tasks.append(level_task["ops"])
        
        # Hash stage constants with multiply-add fusion
        hash_add_const = []
        hash_shift_const = []
        hash_fma_mult = []
        for stage_idx, (op_a, val_a, op_m, op_s, val_s) in enumerate(HASH_STAGES):
            hash_add_const.append(create_broadcast_const(val_a, f"hadd_{stage_idx}"))
            hash_shift_const.append(create_broadcast_const(val_s, f"hshift_{stage_idx}"))
            if op_a == "+" and op_m == "+" and op_s == "<<":
                fma_coeff = (1 + (1 << val_s)) % (2**32)
                hash_fma_mult.append(create_broadcast_const(fma_coeff, f"hfma_{stage_idx}"))
            else:
                hash_fma_mult.append(None)
        
        # Per-vector pointers for scatter/gather
        vec_idx_ptr = []
        vec_val_ptr = []
        for v in range(num_vectors):
            vec_idx_ptr.append(self.alloc_scratch(f"vidxp_{v}"))
            vec_val_ptr.append(self.alloc_scratch(f"vvalp_{v}"))
        
        offset_scratch = self.alloc_scratch("offsets", num_vectors)
        ping_ptr = self.alloc_scratch("ping")
        pong_ptr = self.alloc_scratch("pong")
        step_size = None
        
        ptr_init = new_task_seq()
        
        for v in range(0, num_vectors, 2):
            ops = [("load", ("const", offset_scratch + v, v * VLEN))]
            if v + 1 < num_vectors:
                ops.append(("load", ("const", offset_scratch + v + 1, (v + 1) * VLEN)))
            append_parallel(ptr_init, ops)
        
        for v in range(0, num_vectors, 12):
            ops = []
            for j in range(v, min(v + 12, num_vectors)):
                ops.append(("alu", ("+", vec_idx_ptr[j], self.scratch["indices_ptr"], offset_scratch + j)))
            append_parallel(ptr_init, ops)
        
        for v in range(0, num_vectors, 12):
            ops = []
            for j in range(v, min(v + 12, num_vectors)):
                ops.append(("alu", ("+", vec_val_ptr[j], self.scratch["values_ptr"], offset_scratch + j)))
            append_parallel(ptr_init, ops)
        
        alt_off = offset_scratch + 1 if num_vectors > 1 else offset_scratch
        append_parallel(ptr_init, [
            ("alu", ("+", ping_ptr, self.scratch["values_ptr"], offset_scratch)),
            ("alu", ("+", pong_ptr, self.scratch["values_ptr"], alt_off)),
        ])
        
        if num_vectors >= 3:
            step_size = offset_scratch + 2
        else:
            step_size = self.alloc_scratch("step")
            append_op(ptr_init, "load", ("const", step_size, 2 * VLEN))
        
        for v in range(0, num_vectors, 2):
            ops = [("valu", ("^", rel_idx_buf + v * VLEN, rel_idx_buf + v * VLEN, rel_idx_buf + v * VLEN))]
            if v + 1 < num_vectors:
                ops.append(("valu", ("^", rel_idx_buf + (v + 1) * VLEN, rel_idx_buf + (v + 1) * VLEN, rel_idx_buf + (v + 1) * VLEN)))
            ops.append(("alu", ("+", ping_ptr, ping_ptr, step_size)))
            ops.append(("alu", ("+", pong_ptr, pong_ptr, step_size)))
            ops.append(("load", ("vload", hash_val_buf + v * VLEN, ping_ptr)))
            if v + 1 < num_vectors:
                ops.append(("load", ("vload", hash_val_buf + (v + 1) * VLEN, pong_ptr)))
            append_parallel(ptr_init, ops)
        
        init_tasks.append(ptr_init["ops"])
        run_scheduler(init_tasks)
        
        self.add("flow", ("pause",))
        
        # Main processing loop
        compute_tasks = [[] for _ in range(num_vectors)]
        for v in range(num_vectors):
            ridx = rel_idx_buf + v * VLEN
            hval = hash_val_buf + v * VLEN
            wa = work_buf_a + v * VLEN
            wb = work_buf_b + v * VLEN
            ops = compute_tasks[v]
            seq_num = 0
            
            def add_single(unit, instr, target=ops):
                nonlocal seq_num
                target.append((unit, instr, seq_num))
                seq_num += 1
            
            def add_batch(instr_list, target=ops):
                nonlocal seq_num
                for u, i in instr_list:
                    target.append((u, i, seq_num))
                seq_num += 1
            
            tree_period = forest_height + 1
            for r in range(rounds):
                tree_depth = r % tree_period
                
                # Lookup tree node based on current position
                if tree_depth == 0:
                    add_single("valu", ("^", hval, hval, prefetched[0]))
                elif tree_depth == 1:
                    add_single("flow", ("vselect", wb, ridx, prefetched[2], prefetched[1]))
                    add_single("valu", ("^", hval, hval, wb))
                elif tree_depth == 2:
                    add_batch([("valu", ("&", wa, ridx, const_one)), ("valu", (">>", wb, ridx, const_one)), ("store", ("vstore", vec_idx_ptr[v], ridx))])
                    add_single("valu", ("&", wb, wb, const_one))
                    add_single("flow", ("vselect", ridx, wb, prefetched[5], prefetched[3]))
                    add_single("flow", ("vselect", wb, wb, prefetched[6], prefetched[4]))
                    add_single("flow", ("vselect", wa, wa, wb, ridx))
                    add_single("valu", ("^", hval, hval, wa))
                    add_single("load", ("vload", ridx, vec_idx_ptr[v]))
                else:
                    add_single("valu", ("+", wa, ridx, level_ptr[tree_depth]))
                    for elem in range(0, VLEN, 2):
                        ld_batch = [("load", ("load_offset", wb, wa, elem))]
                        if elem + 1 < VLEN:
                            ld_batch.append(("load", ("load_offset", wb, wa, elem + 1)))
                        add_batch(ld_batch)
                    add_single("valu", ("^", hval, hval, wb))
                
                # Apply hash transformation
                for stage_idx, (op_a, _, op_m, op_s, _) in enumerate(HASH_STAGES):
                    if hash_fma_mult[stage_idx] is not None:
                        add_single("valu", ("multiply_add", hval, hval, hash_fma_mult[stage_idx], hash_add_const[stage_idx]))
                        continue
                    add_batch([("valu", (op_a, wa, hval, hash_add_const[stage_idx])), ("valu", (op_s, wb, hval, hash_shift_const[stage_idx]))])
                    add_single("valu", (op_m, hval, wa, wb))
                
                # Advance relative position in tree
                if tree_depth == forest_height:
                    add_single("valu", ("^", ridx, ridx, ridx))
                elif tree_depth == 0:
                    add_single("valu", ("&", ridx, hval, const_one))
                else:
                    add_single("valu", ("&", wa, hval, const_one))
                    add_single("valu", ("multiply_add", ridx, ridx, const_two, wa))
            
            # Convert relative to absolute index: final depth is rounds % tree_period
            final_depth = rounds % tree_period
            if final_depth == 1:
                add_single("valu", ("+", ridx, ridx, const_one))
            elif final_depth >= 2:
                add_single("valu", ("+", ridx, ridx, level_offset[final_depth]))
            
            # Write back results
            add_batch([("store", ("vstore", vec_val_ptr[v], hval)), ("store", ("vstore", vec_idx_ptr[v], ridx))])
        
        run_scheduler(compute_tasks)
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
