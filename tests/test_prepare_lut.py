import numpy as np
N = 4096
K = 4096
g = 4
act_group_size = 64
_gamma = 1
maxv = (1 << 7) - 1
dtype = "int8"

_states = [-1, 1]

out_dtype = "float32"

b = np.random.randn(N, K).astype("int8")
b_t = b

b = b.reshape(N, K // g, g)

codes = np.array([[i] for i in range(1 << g)], dtype=np.uint8)
codes = np.unpackbits(codes, axis=1, bitorder="little", count=g).T

def map_states(c):
    return _states[c]

m = np.vectorize(map_states)(codes).astype(out_dtype)

lut = b.dot(m)

lut_biases = lut.reshape(N, K // act_group_size, act_group_size // g, 1 << g)[:, :, :, 0]
lut_biases = np.sum(lut_biases, axis=-1) * _gamma

qlut = lut.reshape(N, K // act_group_size, act_group_size // g * (1 << g))
absmax = np.max(np.abs(qlut), axis=-1)
lut_scales = absmax / maxv

def recp(s):
    return 1.0 / s if s != 0 else 0

ils = np.vectorize(recp)(lut_scales).astype(out_dtype)
qlut = np.rint((qlut.transpose(0, 2, 1) * ils).transpose(0, 2, 1).reshape(N, K // g, 1 << g)).astype(dtype)