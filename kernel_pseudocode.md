# Python Pseudo-code for the Mamba Kernel

This describes the file kernel_pseudocode.py, which is python code meant to
illustrate the [CUDA kernel](./csrc/selective_scan/selective_scan_fwd_kernel.cuh).
While the actual CUDA kernel handles both real and complex case, this pseudocode only
handles the real case.  This means the Selective Scan binary operator has a signature
`float2, float2 -> float2`.  The complex case has signature `float4, float4 ->
float4` and can be found [here](./csrc/selective_scan/selective_scan_common.h#L118).
`float2` is a pair of floats, `float4` is a 4-tuple of floats.

References to code there are commit 6dbfc455 given as K + line number

## Intro to CUDA thread blocks and threads

As a brief intro, note that CUDA code organizes the work to be done into independent
chunks called thread blocks.  Each thread block is a group of threads that can
communicate with each other through small, fast memory called 'shared memory'.  The
individual threads inside one thread block must be synchronized manually by the
programmer.

In contrast, the programmer has no control over the order in which different thread
blocks get executed.  This is controlled by a scheduler.  Because of this, it is a
logical error if the CUDA code is written so that the overall output of the
computation would depend on this order.

What this means practically is that you should feed non-overlapping pieces of the
input data (data 'shards') to different thread blocks, and the output of each thread
block should be to non-overlapping pieces of the overall output.

## Mamba uses one thread block for each (batch, dim) combo

In this work, the overall structure uses a 2D grid `(batch, dim)` of thread blocks,
and a 1D thread block of size `nthreads`, such that `nthreads * nitems = chunk_size`.
A chunk is a fixed-size contiguous range of the input sequence.

`dim` refers to D in the paper, the number of input channels.  Note that this implies
that the computation done by this kernel is done completely independently on each
input channel.  Channel mixing does happen, but it happens in the step that prepares
the input for this kernel.

## Within a thread block, work is sharded by sequence position

Here is a diagram showing a thread block with `chunk_size=40, nthreads=8, nitems=5`.
Note that any given thread processes the same nitems-long subrange in each chunk.

```
|              chunk 0                  |               chunk 1                 |     
| T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
^^^^^
```

Each thread works in a nested loop.  The outer loop goes over chunks of `chunk_size`
timesteps of the sequence.  Within a given chunk, a given thread processes only
`nitems` of consecutive timesteps of that chunk.  The inner loop goes over the
elements of the hidden state.  

At K333 one can see that the values for nthreads, nitems are:

    nthreads    nitems    seqlen
    32          4         [0, 128] 
    32          8         (128, 256]
    32          16        (256, 512]
    64          16        (512, 1024]
    128         16        (1024, inf]

Thus thread thread_id will process items 
`[thread_id * nitems, (thread_id + 1) * nitems)`

## Undocumented details:  kNRows and n_groups

Ordinarily, one thread block handles one `(batch, dim)` element of the input.
However, there is another constant `kNRows`, ordinarily set to 1, but which serves as
a companion dimension to dim.   See K307.  If kNRows were set larger than 1, then
each input would need to expand the `[dim]` sub-shape to `[dim, kNRows]`.  There
doesn't seem to be any plan to use kNRows other than 1, and if I'm not mistaken, it
would actually break the existing code because in `selective_scan.cpp`, the
`CHECK_SHAPE` calls do not allow for that extra dimension. 

Then, there is the n_groups variable.  It is assumed to be 1 if you provide B and C
with shapes `[batch, dstate, seqlen]`.  You can also provide B and C with shape
`[batch, n_groups, dstate, seqlen]` however.  This has the effect that it groups the
dim dimension into n_groups groups.

How does this work?  In the default case, we have (I'm using raw latex since inline
math doesn't display correctly):

```
\bar{B}_{bdln} = \Delta_{bld} B_{bnl}
y_{bdl} = h_{bdnl} C_{bnl}
```

In the case where n_groups > 1, we have instead

```
\bar{B}_{bdln} = \Delta_{bld} B'_{bdnl}
y_{bdl} = h_{bdnl} C'_{bdnl}
```

where:

```
B'_{bdnl} = B_{bgnl}
C'_{bdnl} = C_{bgnl}
```

for `g = d // dims_per_group`


## Use of shared memory within a thread block

There seems to be very minimal shared memory use within a thread block.  The
InclusiveScan uses smem_running_prefix to communicate the partial results between
threads, since each thread processes a different sub-range of the input sequence.
Besides that, all shared memory use is wrapped in the load* statements, which use
`cub::BlockLoad`.  It's not clear how this actually shares memory between threads - it
doesn't appear that threads actually read data that another thread writes.

