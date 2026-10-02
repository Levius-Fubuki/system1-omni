"""GPU regression tests: python test_native_vision_cuda.py /path/libqwen3_5_cuda.so."""


def main():
    import ctypes as C
    import sys

    import torch
    import torch.nn.functional as F

    lib = C.CDLL(sys.argv[1])
    P, I, Z = C.c_void_p, C.c_int, C.c_size_t
    signatures = {
        "cs1_vision_linear": [P, P, P, P, P, I, I, I, P],
        "cs1_gemm_f32": [P, P, P, P, I, I, I, P],
        "cs1_vision_bias": [P, P, Z, I, P],
        "cs1_vision_norm": [P, P, P, P, I, I, P],
        "cs1_vision_rope": [P, P, P, P, P, I, P],
        "cs1_vision_attention": [P, P, P, P, I, P],
        "cs1_vision_gelu": [P, Z, I, P],
        "cs1_vision_lora_add": [P, P, Z, C.c_float, P],
        "cs1_gemm_create": [Z],
    }
    for name, args in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = P if name == "cs1_gemm_create" else I

    def run(name, *args):
        rc = getattr(lib, name)(
            *[a.data_ptr() if isinstance(a, torch.Tensor) else a for a in args]
        )
        assert rc == 0, (name, rc)

    def close(got, want, atol=0.02, rtol=0.02):
        torch.cuda.synchronize()
        torch.testing.assert_close(got, want, atol=atol, rtol=rtol)

    torch.manual_seed(93)
    s = torch.cuda.current_stream().cuda_stream
    g = lib.cs1_gemm_create(32 << 20)
    assert g
    x = torch.randn(68, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16) / 32
    b = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    run("cs1_vision_linear", g, x, w, b, y, 68, 1024, 1024, s)
    close(y, F.linear(x, w, b), atol=0.015)
    run("cs1_vision_norm", x, w[0], b, y, 68, 1024, s)
    close(y, F.layer_norm(x, (1024,), w[0], b, 1e-6), atol=0.008)
    for exact in (0, 1):
        y.copy_(x)
        run("cs1_vision_gelu", y, y.numel(), exact, s)
        close(
            y,
            F.gelu(x, approximate="none" if exact else "tanh"),
            atol=0.0001,
            rtol=0.001,
        )
    a = torch.randn(16, 1024, device="cuda") / 32
    xf = x.float()
    r = torch.empty(68, 16, device="cuda")
    run("cs1_gemm_f32", g, xf, a, r, 68, 16, 1024, s)
    close(r, F.linear(xf, a), atol=2e-6, rtol=2e-5)
    lora = torch.randn_like(xf) * 0.01
    y.copy_(x)
    run("cs1_vision_lora_add", y, lora, y.numel(), 2.0, s)
    close(y, (x.float() + 2 * lora).bfloat16(), atol=0, rtol=0)
    for n in (4, 68, 256):
        qkv = torch.randn(n, 3, 16, 64, device="cuda", dtype=torch.bfloat16)
        angles = torch.randn(n, 32, device="cuda")
        co, si = angles.cos(), angles.sin()
        q, k = (
            torch.empty(n, 16, 64, device="cuda", dtype=torch.bfloat16),
            torch.empty(n, 16, 64, device="cuda", dtype=torch.bfloat16),
        )
        run("cs1_vision_rope", qkv, co, si, q, k, n, s)
        co2, si2 = co.repeat(1, 2)[:, None], si.repeat(1, 2)[:, None]

        def rope(z, co2=co2, si2=si2):
            z = z.float()
            return (
                z * co2 + torch.cat((-z[..., 32:], z[..., :32]), -1) * si2
            ).bfloat16()

        close(q, rope(qkv[:, 0]), atol=0, rtol=0)
        close(k, rope(qkv[:, 1]), atol=0, rtol=0)
        out = torch.empty_like(q)
        run("cs1_vision_attention", q, k, qkv[:, 2].data_ptr(), out, n, s)
        want = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), qkv[:, 2].transpose(0, 1)
        ).transpose(0, 1)
        close(out, want, atol=0.008, rtol=0.015)
    # Conv3d rounds the convolution before its separately applied BF16 bias.
    xp = torch.randn(4, 1536, device="cuda", dtype=torch.bfloat16)
    wp = torch.randn(1024, 1536, device="cuda", dtype=torch.bfloat16) / 32
    bp = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    z = torch.zeros_like(bp)
    yp = torch.empty(4, 1024, device="cuda", dtype=torch.bfloat16)
    run("cs1_vision_linear", g, xp, wp, z, yp, 4, 1024, 1536, s)
    run("cs1_vision_bias", yp, bp, yp.numel(), 1024, s)
    want = F.conv3d(
        xp.reshape(4, 3, 2, 16, 16),
        wp.reshape(1024, 3, 2, 16, 16),
        bp,
        stride=(2, 16, 16),
    ).flatten(1)
    close(yp, want, atol=0.008, rtol=0.001)
    assert (yp.float() - want.float()).abs().mean().item() < 1e-5
    print("native vision CUDA primitives: PASS")


if __name__ == "__main__":
    main()
