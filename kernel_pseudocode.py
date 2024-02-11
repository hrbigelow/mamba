"""
This is Python pseudo-code to represent the structure of
selective_scan_fwd_kernel from 
csrc/selective_scan/selective_scan_fwd_kernel.cuh, line 69

Please see kernel_pseudocode.md for detail 
"""

def run_kernel(nthreads, nitems, u, delta, A, B, C, out):
    """
    Launches all thread blocks.  Each thread block receives a different (batch_id,
    dim_id) slice of the inputs data.

    nthreads: int, threads per thread block
    nitems: int, number of timesteps each thread handles per chunk

    u: [batch, dim, seqlen] 
    delta: [batch, dim, seqlen]
    A: [dim, dstate]
    B: [batch, n_groups, dstate, seqlen]
    C: [batch, n_groups, dstate, seqlen]

    Returns:
    out: [batch, dim, seqlen]
    """
    batch, dim, seqlen = u.shape
    _, dstate = A.shape
    n_chunks = (seqlen + 2048 - 1) / 2048;
    x = torch.empty(batch, dim, n_chunks, dstate * 2)

    for batch_id in range(batch):
        for dim_id in range(dim):
            x = x[batch_id, dim_id]
            u = u[batch_id, dim_id]
            delta = delta[batch_id, dim_id]
            A = A[dim_id]
            B = B[batch_id]
            C = C[batch_id]
            out = out[batch_id, dim_id]
            run_block(nthreads, nitems, x, u, delta, A, B, C, out)

def run_block(nthreads, nitems, x, u, delta, A, B, C, out):
    """
    Runs one thread block.  Launches nthreads individual threads to process

    x: [n_chunks, dstate*2]
    u: [seqlen]
    delta: [seqlen]
    A: [dstate]
    B: [n_groups, dstate, seqlen]
    C: [n_groups, dstate, seqlen]
    out: [seqlen]
    """
    seqlen, = u.shape
    n_chunks = (seqlen + 2048 - 1) / 2048;
    for thread_id in range(nthreads):
        run_thread(thread_id, nitems, x, u, delta, A, B, C, out)

def run_thread(thread_id, nitems, x, u, delta, A, B, C, out):
    """
    Computes a scalar temporal sequence from the input using the SSM mechanism.
    Processes the seqlen temporal sequence in chunks

    x: [n_chunks, dstate*2]
    u: [seqlen]
    delta: [seqlen]
    A: [dstate]
    B: [n_groups, dstate, seqlen]
    C: [n_groups, dstate, seqlen]

    Returns
    out: [seqlen]
    """
    _, dstate, seqlen = B.shape
    thread_offset = thread_id * nitems # offset relative to chunk start

    # only thread_id 0 writes, thread_id % 32 == 0 reads from smem_running_prefix
    smem_running_prefix = [None] * dstate 

    for chunk in range(n_chunks):
        slice_start = chunk_size * chunk + thread_offset
        slice_end = slice_start + nitems
        seq_slice = slice(slice_start, slice_end)
        u_vals = u[seq_slice] # K140 load_input
        delta_vals_load = delta[seq_slice] # K142 load_input

        for i in range(nitems):
            delta_vals[i] = delta_vals_load[i] + delta_bias
            if delta_vals[i] <= 20.0: # K155 
                delta_vals[i] = log1pf(expf(delta_vals[i])) 
            delta_u_vals[i] = delta_vals[i] * u_vals[i]
            out_vals[i] = D_val * u_vals[i]

        for s in range(dstate):
            thread_data = [None] * nitems
            B_vals = B[0, s, seq_slice] # K182 load_weight...
            C_vals = C[0, s, seq_slice] # K193 load_weight...
            for i in range(nitems):
                if slice_start + i >= seqlen:
                    thread_data[i] = (1.0, 0.0) # K220
                else:
                    thread_data[i] = (exp2f(delta_vals[i] * A[s]), 
                                      B_vals[i] * delta_u_vals[i]) # K217
            if chunk > 0 and thread_id % 32 == 0: # K239
                running_prefix = smem_running_prefix[s] 
            else:
                running_prefix = (1.0, 0.0)
            op = prefix_op(running_prefix)
            InclusiveScan(thread_data, op) # K246

            if thread_id == 0: # K251
                smem_running_prefix[s] = op.running_prefix
                x[chunk, s] = op.running_prefix

            for i in range(nitems):
                out_vals[i] += thread_data[i][1] * C_vals[i] # K261

        out[seq_slice] = out_vals[:] # K277


