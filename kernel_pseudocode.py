"""
This is Python pseudo-code to represent the structure of
selective_scan_fwd_kernel from 
csrc/selective_scan/selective_scan_fwd_kernel.cuh, line 69

References to code there are commit 6dbfc455
given as K + line number
in csrc/selective_scan/selective_scan_fwd_kernel.cuh

The overall structure uses a 2D grid (batch, dim) and a 1D thread block of size
nthreads such that nthreads * nitems = chunk.  A chunk is a fixed-size contiguous
range of the input sequence.

`dim` refers to D, the number of input channels.  Note that in CUDA, there must be no
assumption about what order each thread block runs.  This implies that the channels
are supposed to be independent of each other.

Within one block, each thread works in a nested loop.  The outer loop goes over
chunks of size 2048 timesteps of the sequence.  The inner loop goes over the elements
of the hidden state.

Within a given chunk, a given thread processes only nitems of consecutive timesteps
of that chunk.

At K333 one can see that the values for nthreads, nitems are:

nthreads    nitems    seqlen
32          4         [0, 128] 
32          8         (128, 256]
32          16        (256, 512]
64          16        (512, 1024]
128         16        (1024, inf]

Thus thread thread_id will process items 
[thread_id * nitems, (thread_id + 1) * nitems)

In the python code below, the block / threads hierarchy is represented at the top
level

Ordinarily, one thread block handles one (batch, dim) element of the input.  However,
there is another constant kNRows, ordinarily set to 1, but which serves as a
companion dimension to dim.   See K307.  If kNRows were set larger than 1, then each
input would need to expand the [dim] sub-shape to [dim, kNRows].  There doesn't seem
to be any plan to use kNRows other than 1, and if I'm not mistaken, it would actually
break the existing code because in selective_scan.cpp, the CHECK_SHAPE calls do not
allow for that extra dimension. 

Then, there is the n_groups variable.  It is assumed to be 1 if you provide B and C
with shapes [batch, dstate, seqlen].  You can also provide B and C with shape [batch,
n_groups, dstate, seqlen] however.  This has the effect that it groups the dim
dimension into n_groups groups.

How does this work?  In the default case, we have:

Bbar_{bdln} = Delta_{bld} B_{bnl}
y_{bdl} = h_{bdnl} C_{bnl}

In the case where n_groups > 1, we have instead

Bbar_{bdln} = Delta_{bld} B'_{bdnl}
y_{bdl} = h_{bdnl} C'_{bdnl}

where:
B'_{bdnl} = B_{bgnl} 
C'_{bdnl} = C_{bgnl} 

for g = d // dims_per_group


Use of shared memory within a thread block

There seems to be very minimal shared memory use within a thread block.  The
InclusiveScan uses smem_running_prefix to communicate the partial results between
threads, since each thread processes a different sub-range of the input sequence.
Besides that, all shared memory use is wrapped in the load* statements, which use
cub::BlockLoad.  It's not clear how this actually shares memory between threads - it
doesn't appear that threads actually read data that another thread writes.
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


