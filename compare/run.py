from asyncore import read
import os
import argparse
import re
import csv
import torch
import triton
import torchinductor
import torchinductor.triton_ops

torch.backends.cuda.matmul.allow_tf32 = True


def parse_csv(filename, func, a_col=1, b_col=2):
    M_K_N_list = []
    # dirname = os.path.dirname(__file__)
    # filename = os.path.join(dirname, filename)
    if func == "bmm":
        len_shape = 3
    elif func == "mm":
        len_shape = 2
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            a_shape = re.findall(r'\b\d+\b', row[a_col])
            b_shape = re.findall(r'\b\d+\b', row[b_col])
            # triton.matmul only supports 2d matrix
            if len(a_shape) != len_shape or len(b_shape) != len_shape:
                continue
            if func == "mm":
                assert a_shape[1] == b_shape[1]
                new_shapes = [int(a_shape[0]), int(a_shape[1]), int(b_shape[0])]
            elif func == "bmm":
                assert a_shape[-1] == b_shape[1]
                new_shapes = [int(a_shape[0]), int(a_shape[1]), int(a_shape[2]), int(b_shape[0])]
            if new_shapes not in M_K_N_list:
                M_K_N_list.append(new_shapes)
    return M_K_N_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--read_file", type=str, help="The csv file to read"
    )
    parser.add_argument(
        "--func", type=str, help="mm or bmm"
    )
    parser.add_argument(
        "--a_col", type=int, default=1, help="a_col idx in read_file"
    )
    parser.add_argument(
        "--b_col", type=int, default=2, help="a_col idx in read_file"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", help="float16 or float32"
    )
    parser.add_argument(
        "--warmup", type=int, default=25, help="warmup iterations"
    )
    parser.add_argument(
        "--rep", type=int, default=100, help="rep iterations"
    )


    args = parser.parse_args()
    read_file = args.read_file
    dtype = getattr(torch, args.dtype)
    warmup = args.warmup
    rep = args.rep

    M_K_N_list = parse_csv(read_file, args.func, args.a_col, args.b_col)
    # M_K_N_list = [[2048, 320, 628, 2048]]
    aten_tflops, triton_tflops = [], []
    write_file = read_file.split("/")[-1]
    write_file = write_file.split(".")[0] + f"_{args.dtype}.csv"
    with open(write_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        # writer.writerow(["M", "K", "N", "fp16 aten", "fp16 triton", "fp32 aten", "fp32 triton"])
        for shapes in M_K_N_list:
            dtype_tflops = []
            # for dtype in [torch.float16, torch.float32]:
            if args.func == "mm":
                M, K, N = shapes
                a =  torch.randn((M, K), dtype=dtype, device="cuda")
                b =  torch.randn((K, N), dtype=dtype, device="cuda")

                tflops = (
                    lambda ms: 2.0 * M * N * K / ms * 1e-9
                )

                def aten_fn():
                    return torch.ops.aten.mm(a, b)
                def triton_fn():
                    return triton.ops.matmul(a, b)
            elif args.func == "bmm":
                BATCH, M, K, N = shapes
                a = torch.randn((BATCH, M, K), dtype=dtype, device="cuda")
                b = torch.randn((BATCH, K, N), dtype=dtype, device="cuda")
                c = torch.empty((BATCH, M, N), dtype=dtype, device="cuda")

                tflops = (
                    lambda ms: 2.0 * BATCH * M * N * K / ms * 1e-9
                )

                def aten_fn():
                    return torch.ops.aten.bmm.out(a, b, out=c)
                def triton_fn():
                    return torchinductor.triton_ops.bmm_out(a, b, c)


            triton_ms, _, _ = triton.testing.do_bench(triton_fn, warmup=warmup, rep=rep)
            aten_ms, _, _ = triton.testing.do_bench(aten_fn, warmup=warmup, rep=rep)
            writer.writerow([*shapes, tflops(aten_ms), tflops(triton_ms)])
            # dtype_tflops.append(tflops(aten_ms), tflops(triton_ms))
            # print([M, K, N, *dtype_tflops])
            # writer.writerow([M, K, N, *dtype_tflops])

    
        
