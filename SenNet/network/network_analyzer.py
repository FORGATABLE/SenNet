import torch
from fvcore.nn import FlopCountAnalysis
from torch.functional import F
from SenNet import config

class NetworkAnalyzer:
    def __init__(self, network, patch_size=config.patch_size, batch_size=2, input_channels=1, deep_supervision=True,
                 test_backward=False, print_flops=True, print_network=False, print_memory=True):
        self.network = network
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.deep_supervision = deep_supervision
        self.test_backward = test_backward
        self.print_flops = print_flops
        self.print_network = print_network
        self.print_memory = print_memory

    def analyze(self):
        if self.print_network:
            print(self.network)

        if self.print_memory:
            torch.cuda.reset_peak_memory_stats()  # 重置显存统计，避免之前的影响

        x = torch.zeros((self.batch_size, self.input_channels, *self.patch_size), requires_grad=self.test_backward).cuda()
        with torch.autocast(device_type='cuda', enabled=True):
            if self.print_flops:
                with torch.no_grad():
                    flop_analyzer = FlopCountAnalysis(self.network, x)
                    flops = flop_analyzer.total() / 1e9  # GFLOPS
                    params = self.count_parameters() / 1e6  # 1e6 = 10⁶
                    # 输出 FLOPS & 参数量
                    print(f"GFLOPS: {flops:.2f}, 参数量: {params:.2f}M")

            if self.test_backward:
                loss = 0.
                pred = self.network(x)
                if isinstance(pred, list):
                    for p in pred:
                        y = torch.rand(p.size(), requires_grad=False).cuda()
                        loss += F.cross_entropy(p, y.argmax(1))
                else:
                    y = torch.rand(pred.size(), requires_grad=False).cuda()
                    loss += F.cross_entropy(pred, y.argmax(1))

                print(f"Loss: {loss.item()}")
                loss.backward()

        if True:
            # 获取显存占用情况
            mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # MB
            max_mem_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
            mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            max_mem_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2  # MB

            print(f"当前 GPU 显存占用: {mem_allocated:.2f} MB")
            print(f"最大 GPU 显存占用: {max_mem_allocated:.2f} MB")
            print(f"当前 GPU 预留显存: {mem_reserved:.2f} MB")
            print(f"最大 GPU 预留显存: {max_mem_reserved:.2f} MB")

    def print_gflops(self):
        # if self.print_flops:
        #     flops, params = get_model_complexity_info(self.network, (self.input_channels, *self.patch_size), as_strings=True)
        #     print(f"Flops: {flops}, Parameters: {params}")
        # elif self.print_params:
        #     print(f"Number of parameters: {self.count_parameters()}")
        pass


    def count_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)