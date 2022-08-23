import torch
import triton
import triton.language as tl

from .autotune import conv_heuristics


@conv_heuristics()
@triton.jit
def _kernel(
    x,
    w,
    bias,
    y,
    # stride of tensor
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wn,
    stride_wc,
    stride_wh,
    stride_ww,
    stride_yn,
    stride_yc,
    stride_yh,
    stride_yw,
    stride_biasn,
    # Tensor dimensions
    BATCH,
    IN_C,
    IN_H,
    IN_W,
    KERNEL_N,
    KERNEL_H,
    KERNEL_W,
    OUT_H,
    OUT_W,
    # parameters of conv
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_padding_h,
    output_padding_w,
    groups,
    # Metaparameters
    ACC_TYPE: tl.constexpr,
    # blocks in different dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # reduction tiling parameter for matmul
    BLOCK_K: tl.constexpr,
    # Super-blocking for better L2 peformance
    GROUP_H: tl.constexpr,
):
    """
    each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of y it should compute.
    pid_nhw = tl.program_id(0)
    pid_k = tl.program_id(1)

    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W
    off_y_w = off_y_hw % OUT_W

    # offset for the initial ptr for x
    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h - padding_h
    off_x_w = off_y_w * stride_w - padding_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_inc = tl.arange(0, BLOCK_K)

    # load inc ptr of x, upade x_ptrs
    # position of sliding window at position h, w = [0,0]
    off_x00_crs_unpacked = off_x_inc * stride_xc
    # the remaining 8 elements are power-of-2
    RANGE_8 = tl.arange(0, 8)
    delta_xh = RANGE_8 // 3
    delta_xw = RANGE_8 % 3
    delta_xc = off_x_inc
    delta_xh_ravel = tl.ravel(
        delta_xh[:, None] + tl.zeros([BLOCK_K], dtype=tl.int32)[None, :]
    )
    delta_xw_ravel = tl.ravel(
        delta_xw[:, None] + tl.zeros([BLOCK_K], dtype=tl.int32)[None, :]
    )
    delta_xc_ravel = tl.ravel(
        delta_xc[None, :] + tl.zeros([8], dtype=tl.int32)[:, None]
    )
    off_x_crs_unpacked = (
        delta_xh_ravel * dilation_h * stride_xh
        + delta_xw_ravel * dilation_w * stride_xw
        + delta_xc_ravel * stride_xc
    )
    x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
    # x_ptrs = x + off_x_nhw[:, None] + tl.zeros([8 * BLOCK_K], dtype=tl.int32)[None, :]
    # x_ptrs = x + tl.zeros([BLOCK_M, 8 * BLOCK_K], dtype=tl.int32) + off_x_crs_unpacked[None, :]
    # x_ptrs = x + tl.zeros([BLOCK_M, 8 * BLOCK_K], dtype=tl.int32)
    x00_ptrs = x + off_x_nhw[:, None] + off_x00_crs_unpacked[None, :]

    mask_x00 = (
        (off_x_n < BATCH)[:, None]  # BLOCK_N
        & (off_x_inc < IN_C)[None, :]  # BLOCK_K
        & (off_x_h >= 0)[:, None]
        & (off_x_h < IN_H)[:, None]
        & (off_x_w >= 0)[:, None]
        & (off_x_w < IN_W)[:, None]
    )
    mask_x = (
        (off_x_n < BATCH)[:, None]  # BLOCK_N
        & (delta_xc_ravel < IN_C)[None, :]  # 8 * BLOCK_K
        & (off_x_h[:, None] + (delta_xh_ravel * dilation_h)[None, :] >= 0)
        & (off_x_h[:, None] + (delta_xh_ravel * dilation_h)[None, :] < IN_H)
        & (off_x_w[:, None] + (delta_xw_ravel * dilation_w)[None, :] >= 0)
        & (off_x_w[:, None] + (delta_xw_ravel * dilation_w)[None, :] < IN_W)
    )

    # offset for the inital ptr for w
    off_w00_crs = off_x_inc * stride_wc
    off_w_k = off_y_k
    w00_ptrs = w + off_w00_crs[:, None] + off_w_k[None, :] * stride_wn
    mask_w00 = (off_x_inc < IN_C)[:, None] & (off_w_k < KERNEL_N)[None, :]

    off_w_crs = (
        delta_xh[:, None] * stride_wh
        + delta_xw[:, None] * stride_ww
        + delta_xc[None, :] * stride_wc
    )
    off_w_crs_ravel = tl.ravel(off_w_crs)
    w_ptrs = w + off_w_crs_ravel[:, None] + off_w_k[None, :] * stride_wn
    mask_w = (delta_xc_ravel < IN_C)[:, None] & (off_w_k < KERNEL_N)[None, :]

    # ------ load x ------
    matrix_x00 = tl.load(x00_ptrs, mask=mask_x00)  # BLOCK * (1 * BLOCK_K)
    matrix_x = tl.load(x_ptrs, mask=mask_x)  # BLOCK_M * (8 * BLOCK_K)
    # ------ load w ------
    matrix_w00 = tl.load(w00_ptrs, mask=mask_w00)  # (1 * BLOCK_K) * BLOCK_N
    matrix_w = tl.load(w_ptrs, mask=mask_w)  # (8 * BLOCK_K) * BLOCK_N

    # -----------------------------------------------------------
    # allocate accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for inc in range(0, IN_C, BLOCK_K):

        # ------ matrix multiplication ------
        acc += tl.dot(matrix_x, matrix_w) + tl.dot(matrix_x00, matrix_w00)
        # ------ update ptrs ------
        off_x_inc = inc + BLOCK_K + tl.arange(0, BLOCK_K)
        delta_xc_ravel = tl.ravel(
            off_x_inc[None, :] + tl.zeros([8, BLOCK_K], dtype=tl.int16)
        )
        x00_ptrs += BLOCK_K * stride_xc
        w00_ptrs += BLOCK_K * stride_wc
        x_ptrs += BLOCK_K * stride_xc
        w_ptrs += BLOCK_K * stride_wc

        mask_x00 = (
            (off_x_n < BATCH)[:, None]
            & (off_x_inc < IN_C)[None, :]
            & (off_x_h >= 0)[:, None]
            & (off_x_h < IN_H)[:, None]
            & (off_x_w >= 0)[:, None]
            & (off_x_w < IN_W)[:, None]
        )
        mask_x = (
            (off_x_n < BATCH)[:, None]  # BLOCK_N
            & (delta_xc_ravel < IN_C)[None, :]  # 8 * BLOCK_K
            & (off_x_h[:, None] + (delta_xh_ravel * dilation_h)[None, :] >= 0)
            & (off_x_h[:, None] + (delta_xh_ravel * dilation_h)[None, :] < IN_H)
            & (off_x_w[:, None] + (delta_xw_ravel * dilation_w)[None, :] >= 0)
            & (off_x_w[:, None] + (delta_xw_ravel * dilation_w)[None, :] < IN_W)
        )
        mask_w00 = (off_x_inc < IN_C)[:, None] & (off_w_k < KERNEL_N)[None, :]
        mask_w = (delta_xc_ravel < IN_C)[:, None] & (off_w_k < KERNEL_N)[None, :]
        # ------ prefetch ------
        # ------ load x ------
        matrix_x00 = tl.load(x00_ptrs, mask=mask_x00)
        matrix_x = tl.load(x_ptrs, mask=mask_x)
        # ------ load w ------
        matrix_w00 = tl.load(w00_ptrs, mask=mask_w00)
        matrix_w = tl.load(w_ptrs, mask=mask_w)

    # add bias if is not None
    if bias is not None:
        off_bias_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_ptrs = bias + off_bias_k * stride_biasn
        mask_bias = off_bias_k < KERNEL_N
        _bias = tl.load(bias_ptrs, mask=mask_bias)
        acc += _bias[None, :]

    acc = acc.to(y.dtype.element_ty)

    # rematerialize -- this saves some registers
    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    # consider output padding
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    # y ptrs in the block of [BLOCK_M, BLOCK_N]
    y_ptrs = (
        y
        + off_y_n[:, None] * stride_yn
        + off_y_h[:, None] * stride_yh
        + off_y_w[:, None] * stride_yw
        + off_y_k[None, :] * stride_yc
    )

    # out-of-bounds check
    mask_y = (
        (off_y_n < BATCH)[:, None]
        & (off_y_h < OUT_H + output_padding_h)[:, None]
        & (off_y_w < OUT_W + output_padding_w)[:, None]
        & (off_y_k < KERNEL_N)[None, :]
    )

    tl.store(y_ptrs, acc, mask=mask_y)

    return


class _conv3x3_unroll:
    kernel = _kernel

    @staticmethod
    def _call(
        x,
        w,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    ):
        # Q: should we check x, w, bias dtypes?
        device = x.device
        # input shapes
        shape_x = x.shape
        shape_w = w.shape
        shape_bias = bias.shape if bias is not None else None

        # indicies for the layeout
        xn, xc, xh, xw = 0, 1, 2, 3
        yn, yc, yh, yw = 0, 1, 2, 3
        wn, wc, wh, ww = 0, 1, 2, 3

        # out_channel, in_channel, kernel_height, kernel_width
        kernel_size = [shape_w[wh], shape_w[ww]]
        input_size = [shape_x[xh], shape_x[xw]]
        assert (
            not shape_bias or shape_bias[0] == shape_w[wn]
        ), f"bias shape did not match{shape_bias} != {shape_w[wn]}"
        in_channel = shape_w[wc] * groups

        assert shape_x[xc] % groups == 0, "in_channels must be divisible by groups"
        assert shape_w[wn] % groups == 0, "out_channels must be divisible by groups"
        assert (
            shape_x[xc] == in_channel
        ), f"in_channel did not match {shape_x[xc]} != {in_channel}"
        # assert kernel_size == [3, 3], "should be _split kernel"

        assert (
            len(stride)
            == len(padding)
            == len(dilation)
            == len(output_padding)
            == len(kernel_size)
            == len(input_size)
        )

        # output shape
        shape_y = [0] * 4
        shape_y[yn] = shape_x[xn]
        shape_y[yc] = shape_w[wn]
        shape_y[yh] = (
            input_size[0]
            + 2 * padding[0]
            - dilation[0] * (kernel_size[0] - 1)
            - 1
            + stride[0]
        ) // stride[0] + 2 * output_padding[0]
        shape_y[yw] = (
            input_size[1]
            + 2 * padding[1]
            - dilation[1] * (kernel_size[1] - 1)
            - 1
            + stride[1]
        ) // stride[1] + 2 * output_padding[1]

        BATCH = shape_x[xn]
        IN_C = shape_x[xc]
        IN_H = shape_x[xh]
        IN_W = shape_x[xw]
        KERNEL_N = shape_w[wn]
        KERNEL_H = shape_w[wh]
        KERNEL_W = shape_w[ww]
        OUT_H = shape_y[yh]
        OUT_W = shape_y[yw]

        # allocate output
        y = torch.zeros(shape_y, device=device, dtype=x.dtype)

        # get strides for tensors
        stride_x = x.stride()
        stride_w = w.stride()
        stride_bias = bias.stride() if shape_bias else None
        stride_biasn = stride_bias[0] if stride_bias else None

        # output layout should be the same as x
        if stride_x[xc] < stride_x[xh] and stride_x[xc] < stride_x[xw]:
            y = y.to(memory_format=torch.channels_last)
        stride_y = y.stride()

        # accumulator types
        ACC_TYPE = (
            tl.float32
            if x.dtype in [torch.float16, torch.bfloat16, torch.float32]
            else tl.int32
        )

        # launch kernel, 2-dim, batch*h*w, kernel
        def grid(META):
            return (
                triton.cdiv(BATCH * OUT_H * OUT_W, META["BLOCK_M"]),
                triton.cdiv(KERNEL_N, META["BLOCK_N"]),
            )

        _kernel[grid](
            x,
            w,
            bias,
            y,
            # stride nchw for x,w,y tensor
            stride_x[xn],
            stride_x[xc],
            stride_x[xh],
            stride_x[xw],
            stride_w[wn],
            stride_w[wc],
            stride_w[wh],
            stride_w[ww],
            stride_y[yn],
            stride_y[yc],
            stride_y[yh],
            stride_y[yw],
            stride_biasn,
            # Tensor dimensions
            BATCH,
            IN_C,
            IN_H,
            IN_W,
            KERNEL_N,
            KERNEL_H,
            KERNEL_W,
            OUT_H,
            OUT_W,
            # conv parameters
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            output_padding[0],
            output_padding[1],
            groups,
            # Metaparameters
            ACC_TYPE=ACC_TYPE,
            # BLOCK_M=128,
            # BLOCK_N=32,
            # BLOCK_K=BLOCK_K,
            GROUP_H=1,
        )
        return y

    @staticmethod
    def forward(
        x,
        w,
        bias,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        transposed=False,
        output_padding=(0, 0),
        groups=1,
    ):
        if groups != 1:
            print(f"Do not support groups = {groups}")
            return
        if transposed:
            print("Do not support transposed")
        return _conv3x3_unroll._call(
            x,
            w,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )


conv3x3_unroll = _conv3x3_unroll.forward
