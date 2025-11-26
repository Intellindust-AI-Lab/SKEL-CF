import torch


m2mm = 1000.0


def L2_error(x:torch.Tensor, y:torch.Tensor):
    '''
    Calculate the L2 error across the last dim of the input tensors.

    ### Args
    - `x`: torch.Tensor, shape (..., D)
    - `y`: torch.Tensor, shape (..., D)

    ### Returns
    - torch.Tensor, shape (...)
    '''
    return (x - y).norm(dim=-1)


def similarity_align_to(
    S1 : torch.Tensor,
    S2 : torch.Tensor,
):
    '''
    Computes a similarity transform (sR, t) that takes a set of 3D points S1 (3 x N)
    closest to a set of 3D points S2, where R is an 3x3 rotation matrix,
    t 3x1 translation, s scales. That is to solves the orthogonal Procrutes problem.

    The code was modified from [WHAM](https://github.com/yohanshin/WHAM/blob/d1ade93ae83a91855902fdb8246c129c4b3b8a40/lib/eval/eval_utils.py#L201-L252).

    ### Args
    - `S1`: torch.Tensor, shape (...B, N, 3)
    - `S2`: torch.Tensor, shape (...B, N, 3)

    ### Returns
    - torch.Tensor, shape (...B, N, 3)
    '''
    assert (S1.shape[-1] == 3 and S2.shape[-1] == 3), 'The last dimension of `S1` and `S2` must be 3.'
    assert (S1.shape[:-2] == S2.shape[:-2]), 'The batch size of `S1` and `S2` must be the same.'
    original_BN3 = S1.shape
    N = original_BN3[-2]
    S1 = S1.reshape(-1, N, 3) # (B', N, 3) <- (...B, N, 3)
    S2 = S2.reshape(-1, N, 3) # (B', N, 3) <- (...B, N, 3)
    B = S1.shape[0]

    S1 = S1.transpose(-1, -2) # (B', 3, N) <- (B', N, 3)
    S2 = S2.transpose(-1, -2) # (B', 3, N) <- (B', N, 3)
    _device = S2.device
    S1 = S1.to(_device)

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True) # (B', 3, 1)
    mu2 = S2.mean(axis=-1, keepdims=True) # (B', 3, 1)
    X1 = S1 - mu1 # (B', 3, N)
    X2 = S2 - mu2 # (B', 3, N)

    # 2. Compute variance of X1 used for scales.
    var1 = torch.einsum('...BDN->...B', X1**2) # (B',)

    # 3. The outer product of X1 and X2.
    K = X1 @ X2.transpose(-1, -2) # (B', 3, 3)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K) # (B', 3, 3), (B', 3), (B', 3, 3)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(3, device=_device)[None].repeat(B, 1, 1) # (B', 3, 3)
    Z[:, -1, -1] *= (U @ V.transpose(-1, -2)).det().sign()

    # Construct R.
    R = V @ (Z @ U.transpose(-1, -2)) # (B', 3, 3)

    # 5. Recover scales.
    traces = [torch.trace(x)[None] for x in (R @ K)]
    scales = torch.cat(traces) / var1 # (B',)
    scales = scales[..., None, None] # (B', 1, 1)

    # 6. Recover translation.
    t = mu2 - (scales * (R @ mu1)) # (B', 3, 1)

    # 7. Error:
    S1_aligned = scales * (R @ S1) + t # (B', 3, N)

    S1_aligned = S1_aligned.transpose(-1, -2) # (B', N, 3) <- (B', 3, N)
    S1_aligned = S1_aligned.reshape(original_BN3) # (...B, N, 3)
    return S1_aligned # (...B, N, 3)


def align_pcl(Y: torch.Tensor, X: torch.Tensor, weight=None, fixed_scale=False):
    '''
    Align similarity transform to align X with Y using umeyama method. X' = s * R * X + t is aligned with Y.

    The code was copied from [SLAHMR](https://github.com/vye16/slahmr/blob/58518fec991877bc4911e260776589185b828fe9/slahmr/geometry/pcl.py#L10-L60).

    ### Args
    - `Y`: torch.Tensor, shape (*, N, 3) first trajectory
    - `X`: torch.Tensor, shape (*, N, 3) second trajectory
    - `weight`: torch.Tensor, shape (*, N, 1) optional weight of valid correspondences
    - `fixed_scale`: bool, default = False

    ### Returns
    - `s` (*, 1)
    - `R` (*, 3, 3)
    - `t` (*, 3)
    '''
    *dims, N, _ = Y.shape
    N = torch.ones(*dims, 1, 1) * N

    if weight is not None:
        Y = Y * weight
        X = X * weight
        N = weight.sum(dim=-2, keepdim=True)  # (*, 1, 1)

    # subtract mean
    my = Y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = X.sum(dim=-2) / N[..., 0]
    y0 = Y - my[..., None, :]  # (*, N, 3)
    x0 = X - mx[..., None, :]

    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = torch.eye(3).reshape(*(1,) * (len(dims)), 3, 3).repeat(*dims, 1, 1)
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S[neg, 2, 2] = -1

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fixed_scale:
        s = torch.ones(*dims, 1, device=Y.device, dtype=torch.float32)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N  # (*, 1, 1)
        s = (
            torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(
                dim=-1, keepdim=True
            )
            / var[..., 0]
        )  # (*, 1)

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    return s, R, t


def first_k_frames_align_to(
    S1  : torch.Tensor,
    S2  : torch.Tensor,
    k_f : int,
):
    '''
    Compute the transformation between the first trajectory segment of S1 and S2, and use
    the transformation to align S1 to S2.

    The code was modified from [SLAHMR](https://github.com/vye16/slahmr/blob/58518fec991877bc4911e260776589185b828fe9/slahmr/eval/tools.py#L68-L81).

    ### Args
    - `S1`: torch.Tensor, shape (..., L, N, 3)
    - `S2`: torch.Tensor, shape (..., L, N, 3)
    - `k_f`: int
        - The number of frames to use for alignment.

    ### Returns
    - `S1_aligned`: torch.Tensor, shape (..., L, N, 3)
        - The aligned S1.
    '''
    assert (len(S1.shape) >= 3 and len(S2.shape) >= 3), 'The input tensors must have at least 3 dimensions.'
    original_shape = S1.shape # (..., L, N, 3)
    L, N, _ = original_shape[-3:]
    S1 = S1.reshape(-1, L, N, 3) # (B, L, N, 3)
    S2 = S2.reshape(-1, L, N, 3) # (B, L, N, 3)
    B = S1.shape[0]

    # 1. Prepare the clouds to be aligned.
    S1_first = S1[:, :k_f, :, :].reshape(B, -1, 3) # (B, 1, k_f * N, 3)
    S2_first = S2[:, :k_f, :, :].reshape(B, -1, 3) # (B, 1, k_f * N, 3)

    # 2. Get the transformation to perform the alignment.
    s_first, R_first, t_first = align_pcl(
        X = S1_first,
        Y = S2_first,
    ) # (B, 1), (B, 3, 3), (B, 3)
    s_first = s_first.reshape(B, 1, 1, 1) # (B, 1, 1, 1)
    t_first = t_first.reshape(B, 1, 1, 3) # (B, 1, 1, 3)

    # 3. Perform the alignment on the whole sequence.
    S1_aligned = s_first * torch.einsum('Bij,BLNj->BLNi', R_first, S1) + t_first # (B, L, N, 3)
    S1_aligned = S1_aligned.reshape(original_shape) # (..., L, N, 3)
    return S1_aligned # (..., L, N, 3)