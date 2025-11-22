import torch
from torch.optim import AdamW

def svd_topk(x, k=None):
    U, S, Vt = torch.linalg.svd(x, full_matrices=False)
    if k is not None:
        k = min(k, S.size(0))
        U = U[:, :k]
        S = S[:k]
        Vt = Vt[:k, :]
    return U, S, Vt


class Q3RModule:
    def __init__(self, lambda_reg=1e-3, eps=1e-3, recompute_every=100, target_rank=None, use_truncated=False):

        self.lambda_reg = float(lambda_reg)
        self.eps = float(eps)
        self.recompute_every = int(recompute_every)
        self.target_rank = target_rank
        self.use_truncated = use_truncated

    @staticmethod
    def _build_projection_from_svd(U, S, Vt, eps):
        device = U.device
        dtype = U.dtype

        S_clamped = S.clamp(min=1e-12)
        denom = torch.maximum(S_clamped / eps, torch.ones_like(S_clamped))
        inv = (1.0 / denom).to(device=device, dtype=dtype)

        Sigma_inv = torch.diag(inv)

        left_proj = (U @ Sigma_inv) @ U.T
        V = Vt.T
        right_proj = (V @ Sigma_inv) @ V.T

        return left_proj, right_proj

    def compute_and_store_reweight(self, param, state, step):

        W = param.detach()
        
        if W.ndim != 2:
            W_mat = W.view(W.shape[0], -1).detach()
        else:
            W_mat = W

        target_k = None
        if self.target_rank is not None:
            target_k = int(self.target_rank)

        U, S, Vt = svd_topk(W_mat, k=target_k)

        left_proj, right_proj = self._build_projection_from_svd(U, S, Vt, self.eps)

        state['q3r_left'] = left_proj.to(W.device)
        state['q3r_right'] = right_proj.to(W.device)
        state['q3r_ref'] = W_mat.clone()
        state['q3r_singulars'] = S.to(W.device)
        state['q3r_step'] = step

    @staticmethod
    def apply_reweight_operator_to_matrix(W, left_proj, right_proj):
        return left_proj @ W @ right_proj

    def q3r_value(self, W, left_proj, right_proj):
        Rw = self.apply_reweight_operator_to_matrix(W, left_proj, right_proj)
        return 0.5 * torch.sum(W * Rw)

    def add_q3r_grad_to_param(self, param, state, step):
        if param.grad is None:
            param.grad = torch.zeros_like(param.data)

        last = state.get('q3r_step', -999999)
        if ('q3r_left' not in state) or (self.recompute_every > 0 and (step - last) >= self.recompute_every):
            self.compute_and_store_reweight(param, state, step)

        left = state['q3r_left']
        right = state['q3r_right']

        W = param.data
        if W.ndim != 2:
            W_mat = W.view(W.shape[0], -1)
        else:
            W_mat = W

        Rw = self.apply_reweight_operator_to_matrix(W_mat, left, right)
        grad_reg = self.lambda_reg * Rw

        if param.data.ndim != 2:
            grad_reg = grad_reg.view_as(param.data)

        param.grad = param.grad + grad_reg

        reg_val = self.q3r_value(W_mat, left, right) * self.lambda_reg
        return reg_val


class AdamQ3R(AdamW):
    def __init__(self, params, q3r_module = None, *,
                 lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 amsgrad=False, foreach=None):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, foreach=foreach)
        self.q3r_module = q3r_module

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if not hasattr(self, "_adam_initialized"):
            super().step(closure=closure)
            self._adam_initialized = True
            return loss

        approx_step = 0
        for group in self.param_groups:
            for p in group["params"]:
                s = self.state.get(p, {}).get("step", 0)
                try:
                    approx_step = max(approx_step, int(s))
                except Exception:
                    pass

        if self.q3r_module is not None:
            for group in self.param_groups:
                if not group.get("q3r", False):
                    continue
                for p in group["params"]:
                    if p.requires_grad:
                        st = self.state.setdefault(p, {})
                        self.q3r_module.add_q3r_grad_to_param(p, st, approx_step)

        super().step(closure=closure)
        return loss
