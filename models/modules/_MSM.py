import torch
from tqdm import tqdm

"""
We provide two implementations of the MSM module.
The above commented out function provides faster speeds, but because more tensors are loaded onto the GPU at once, the memory consumption is higher.
By default, our program uses the following function, which is slower but consumes less GPU memory.
"""

def compute_scores_fast(Z, i, device, topmin_min=0, topmin_max=0.3):
    #q:思路，是不是可以考虑使用余弦相似度？
    #a:可以尝试，但效果可能会有所不同，需进行实验验证。
    # speed fast but space large
    # compute anomaly scores
    image_num, patch_num, c = Z.shape
    patch2image = torch.tensor([]).to(device)
    Z_ref = torch.cat((Z[:i], Z[i+1:]), dim=0)
    patch2image = torch.cdist(Z[i:i+1], Z_ref.reshape(-1, c)).reshape(patch_num, image_num-1, patch_num)
    patch2image = torch.min(patch2image, -1)[0]
    # interval average
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max#选出指定个数个照片
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)#这里对最后一维求均值，shape为[1369，]

def compute_scores_slow(Z, i, device, topmin_min=0, topmin_max=0.3):
    # space small but speed slow
    # compute anomaly scores
    patch2image = torch.tensor([]).to(device)
    for j in range(Z.shape[0]):
        if j != i:
            patch2image = torch.cat((patch2image, torch.min(torch.cdist(Z[i], Z[j]), 1)[0].unsqueeze(1)), dim=1)
    # interval average
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)

def MSM(Z, device, topmin_min=0, topmin_max=0.3):
    anomaly_scores_matrix = torch.tensor([]).double().to(device)
    for i in tqdm(range(Z.shape[0])):
    # for i in range(Z.shape[0]):
        anomaly_scores_i = compute_scores_fast(Z, i, device, topmin_min, topmin_max).unsqueeze(0)#照片1的异常分数，必须为其他照片的同层同r的所有patch的距离最小值   
        anomaly_scores_matrix = torch.cat((anomaly_scores_matrix, anomaly_scores_i.double()), dim=0)    # (N, B)
    return anomaly_scores_matrix#(照片数，patch数)对应每个照片每个patch的异常分数

if __name__ == "__main__":
    device = 'cuda:0'
    import time
    s_time = time.time()
    Z = torch.rand(200, 1369, 1024).to(device)
    MSM(Z, device)
    e_time = time.time()
    print((e_time-s_time)*1000)
    