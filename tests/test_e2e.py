import numpy as np
import os
import tvm
from tvm.autotvm.measure.measure_methods import request_remote
from t_mac.ops import QGeMMLUTBitsCodegen, QGeMMLUTBitsPreprocessorCodegen
from t_mac.weights import preprocess_weights
from t_mac.utils import get_default_device_kwargs
import logging


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

def transform_to_i2(x : np.array):
    # print(x.dtype)
    x_num = np.prod(x.shape)
    x = np.reshape(x, x_num)
    scale = 1
    for i in range(x_num):
        if x[i] != 0:
            scale = x[i]
            break
    x = np.divide(x, scale)
    x = x.astype(np.uint8)
    x = np.reshape(x, [x.shape[0] // 4, 4])
    keep_bit = {0:192, 1:48, 2:12, 3:3}
    ans = np.zeros([x_num // 4], dtype=np.uint8)
    for i in range(4):
        x_bit_col = x[:, i]
        x_bit_shift = np.left_shift(x_bit_col, 6 - i * 2)
        x_bit_shift = np.bitwise_and(x_bit_shift, keep_bit[i])
        ans = np.bitwise_or(ans, x_bit_shift)
    # print(scale.dtype)
    return ans, scale

def i2_to_uint8(w : np.array, m, k):
    num = w.shape[0]
    tmp = []
    for i in range(num):
        x = w[i]
        for j in range(4):
            check = np.right_shift(x, 6 - 2 * j)
            check = np.bitwise_and(check, 3)
            if check == 3:
                tmp.append(-1)
            else:
                tmp.append(check)
    return np.array(tmp).reshape(m, k)


dtype = "int8"
bits = 2
g = 4
group_size = 128
act_group_size = 64

device_kwargs = get_default_device_kwargs("intel_win")

out_dtype = device_kwargs["out_dtype"]

# if act_group_size == -1:
#     act_group_size = K
    
remote_kwargs = None
codegen_kwargs = {
    "save_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out"),
    "dtype": dtype,
    "target": device_kwargs["target"],
    "verify": True,
    "tune": False,
    "remote_kwargs": device_kwargs["remote_kwargs"],
    "bits": bits,
    "out_dtype": out_dtype,
    "act_group_size": act_group_size,
    "cc_opts": device_kwargs["cc_opts"],
}

preprocessor = QGeMMLUTBitsPreprocessorCodegen(name="preprocessor", fast_aggregation_k=0, **codegen_kwargs)
qgemm = QGeMMLUTBitsCodegen(name="qgemm_lut", group_size=group_size, **codegen_kwargs)

# qgemm.m_groups = m_groups
# preprocessor.M = M
# qgemm.act_group_size = K
# preprocessor.act_group_size = K
# qgemm.num_threads = 1
# act_group_size = K


M = 4096 * bits
N = 1
K = 4096

check_scale = np.random.rand()

x = np.random.randn(K, N)
w = []
for i in range(M // bits * K):
     w.append(np.random.randint(-1, 2))
w = np.array(w).reshape(M // bits, K)*check_scale

w_trans, scale = transform_to_i2(w)

pf, _ = preprocessor.compile(N, K)
qf, _ = qgemm.compile(M, N, K)

int8_w = i2_to_uint8(w_trans, M // bits, K) + 2
print(int8_w)

bm = qgemm.bm
kfactor = qgemm.kfactor
weight_dtype = qgemm.weight_dtype

# # Inputs
# Aref -> i2
# Aref = np.random.randint(0, 4, size=(M // bits, K)).astype(weight_dtype)
Aref = int8_w.astype(weight_dtype)
# sref -> whole 1
Sref = np.ones([M // bits, K // group_size]).astype(out_dtype)*check_scale
# Sref = np.ones([M // bits, K // group_size].astype())
Bref = np.random.randn(N, K).astype(out_dtype)

# Outputs
# Aref -> 0 1 2 3
# Adq  -> -2 -1 0 1
Adq = Aref.T.reshape(K // group_size, group_size, M // bits).astype(out_dtype) - 2
Adq = (Adq.transpose(1, 0, 2) * Sref.T).transpose(1, 0, 2).reshape(K, M // bits)
Cref = Bref.dot(Adq)
print(Cref)

dev = tvm.device("llvm")
# TVM Inputs
A_t, Scales_t = preprocess_weights(Aref, Sref, bits=bits, g=g, bm=bm, kfactor=kfactor)
A_t = tvm.nd.array(A_t, dev)
B_t = tvm.nd.array(Bref, dev)
Scales_t = tvm.nd.array(Scales_t, dev)

# TVM Outputs
C_t = tvm.nd.array(Cref, dev)

# print("lut_scale")
# lut_scale = np.zeros((N, K // act_group_size))
# print(lut_scale.shape)

# print("lut_biases")
# lut_bias = np.zeros((N, K // act_group_size))
# print(lut_bias.shape)

# print("q_lut")
# q_lut = np.zeros((N, K // g, 1 << g))
# print(q_lut.shape)

# TVM Intermediates
LUT_Scales = tvm.nd.array(np.zeros((N, K // act_group_size), dtype=out_dtype), dev)
LUT_Biases = tvm.nd.array(np.zeros((N, K // act_group_size), dtype=out_dtype), dev)
QLUT = tvm.nd.array(np.zeros((N, K // g, 1 << g), dtype=dtype), dev)

pf(B_t, LUT_Scales, LUT_Biases, QLUT)
qf(A_t, QLUT, Scales_t, LUT_Scales, LUT_Biases, C_t)

print(C_t)
