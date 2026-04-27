import torch
import matmul1

torch.set_printoptions(threshold=float('inf'), linewidth=2048)
M,N,K = 1024,8192,128

a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
b = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
c = torch.ones(M, N, dtype=torch.bfloat16, device="cuda")

with open('logs.txt', 'w') as f:
    matmul1.kernel_matmul(a, b, c)
    # print(c, file=f)
    c_true = torch.matmul(a,b.T)
    # print(c, file=f)
    torch.testing.assert_close(c, c_true)

torch.cuda.synchronize()

def shiftr(m, s):
    return m >> s

class Swizzle:
    def __init__(self, BBits, MBase, SShift):
        num_bits = BBits
        num_base = MBase
        num_shift = SShift
        assert(num_base >= 0), "MBase must be positive."
        assert(num_bits >= 0), "BBits must be positive."
        assert(abs(num_shift) >= num_bits), "abs(SShift) must be more than BBits."
        
        bit_msk = (1 << num_bits) - 1
        self.yyy_msk = bit_msk << (num_base + max(0, num_shift))
        zzz_msk = bit_msk << (num_base - min(0, num_shift))
        self.msk_sft = num_shift
        
        swizzle_code = self.yyy_msk | zzz_msk
    
    def __call__(self, offset):
        return offset ^ shiftr(offset & self.yyy_msk, self.msk_sft)

swizzle = Swizzle(3,4,3) # 128B swizzleing. 2,4,3   1,4,3 - None
for i in range(0, 1024, 128):
    for j in range(0, 128, 16):
        print(swizzle(i+j + 128) // 16, end='   ') # siwzzling done by TMA. Rows get swizzled as if they are shifted by 128 bytes
    print()