
def check_gpu():
    import torch
    import ultralytics

    print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
    print("Ultralytics:", ultralytics.__version__)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install CUDA PyTorch wheels; see README.md.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("Compiled architectures:", torch.cuda.get_arch_list())
    free, total = torch.cuda.mem_get_info(0)
    print(f"VRAM free / total: {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB")
    # CUDA availability alone does not verify Blackwell kernel compatibility.
    with torch.enable_grad():
        layer = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
        inputs = torch.randn(2, 3, 64, 64, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = layer(inputs).square().mean()
        loss.backward()
        torch.cuda.synchronize()
    del layer, inputs, loss
    torch.cuda.empty_cache()
    print("CUDA convolution + AMP + backward: OK")


if __name__ == "__main__":
    check_gpu()
