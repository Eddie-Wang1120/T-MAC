import numpy as np

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
    for i in range(len(num)):
        x_tmp = tmp[i]

x = np.array(177).astype(np.uint8)
tmp = []
for i in range(4):
    check = np.right_shift(x, 6 - 2 * i)
    check = np.bitwise_and(check, 3)
    if check == 3:
        tmp.append(-1)
    else:
        tmp.append(check)


# bits = 2

# M = 4096 * bits
# N = 1
# K = 4096

# check_scale = np.random.rand()

# x = np.random.randn(K, N)
# w = []
# for i in range(M // bits * K):
#      w.append(np.random.randint(-1, 2))
# w = np.array(w).reshape(M // bits, K)*check_scale

# w_trans, scale = transform_to_i2(w)

# print(w_trans)

# i2_to_uint8(w_trans, M // bits, K)