
import numpy as np
import torch
from scipy.optimize import minimize
import time

def run_single_restart_gan(restart_idx, z0, target_vec, scale, config):
    """
    Worker function for GAN inverse solver restarts.
    Each worker loads its own Generator to avoid pickling/concurrency issues.
    """
    from eddy_current_workflow.forward.edc_solver import edc_forward
    from dataclasses import replace as dataclass_replace
    import sys
    from pathlib import Path
    
    # Setup paths for the worker
    root = Path(config['root_dir'])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    
    # Load model inside worker
    from model import ConditionalConv1DGenerator
    device = torch.device("cuda" if torch.cuda.is_available() and config['use_cuda'] else "cpu")
    
    netG = ConditionalConv1DGenerator(nz=config['nz'], K=config['K'], conditional=False)
    state = torch.load(Path(config['model_path']), map_location=device)
    netG.load_state_dict(state)
    netG.to(device)
    netG.eval()
    
    def decode_z(z_np):
        z_t = torch.tensor(z_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, sigma_norm, mu_norm = netG(z_t)
        sigma_norm = sigma_norm.squeeze().cpu().numpy()
        mu_norm = mu_norm.squeeze().cpu().numpy()
        
        # Denormalize
        n = config['norm']
        sigma = (sigma_norm + 1) / 2 * (n["sigma_max"] - n["sigma_min"]) + n["sigma_min"]
        mu = (mu_norm + 1) / 2 * (n["mu_max"] - n["mu_min"]) + n["mu_min"]
        return sigma, mu

    def objective(z_np):
        sigma, mu = decode_z(z_np)
        try:
            # Forward multi-freq
            freqs = config['frequencies']
            out = []
            for f in freqs:
                probe = dataclass_replace(config['probe_base'], frequency=float(f))
                resp = edc_forward(sigma, mu, probe, config['layer_thickness'], n_quad=config['n_quad_opt'])
                out.extend([resp.impedance_real, resp.impedance_imag])
            pred_vec = np.array(out, dtype=np.float64)
            return float(np.sum((pred_vec - target_vec) ** 2) / scale)
        except Exception:
            return 1e10

    t_start = time.perf_counter()
    opt = minimize(
        objective,
        z0,
        method="L-BFGS-B",
        options={
            "maxiter": config['n_iter'],
            "ftol": 1e-15,
            "gtol": 1e-10,
            "eps": config['fd_epsilon'],
        },
    )
    
    z_opt = opt.x.astype(np.float32)
    sigma, mu = decode_z(z_opt)
    elapsed = time.perf_counter() - t_start
    
    return {
        'restart_idx': restart_idx,
        'z_opt': z_opt,
        'sigma': sigma,
        'mu': mu,
        'nit': opt.nit,
        'fun': opt.fun,
        'elapsed': elapsed
    }

def run_single_restart_dodd(restart_idx, z0, target, scale, config):
    """
    Worker function for Dodd-GAN inverse solver restarts.
    """
    from dodd_forward import dodd_forward
    import sys
    from pathlib import Path
    
    root = Path(config['root_dir'])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        
    from model import ConditionalConv1DGenerator
    device = torch.device("cuda" if torch.cuda.is_available() and config['use_cuda'] else "cpu")
    
    netG = ConditionalConv1DGenerator(nz=config['nz'], K=config['K'], conditional=False)
    state = torch.load(Path(config['model_path']), map_location=device)
    netG.load_state_dict(state)
    netG.to(device)
    netG.eval()

    def decode_z(z_np):
        z_t = torch.tensor(z_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, sigma_norm, mu_norm = netG(z_t)
        sigma_norm = sigma_norm.squeeze().cpu().numpy()
        mu_norm = mu_norm.squeeze().cpu().numpy()
        
        n = config['norm']
        sigma = (sigma_norm + 1) / 2 * (n["sigma_max"] - n["sigma_min"]) + n["sigma_min"]
        mu = (mu_norm + 1) / 2 * (n["mu_max"] - n["mu_min"]) + n["mu_min"]
        return sigma, mu

    def objective(z_np):
        sigma, mu = decode_z(z_np)
        try:
            resp = dodd_forward(sigma, mu, config['probe'], config['layer_thickness'], 
                                integ_top_range=config['integ_range_opt'])
            pred = resp.impedance_complex
            diff = pred - target
            return float((diff.real ** 2 + diff.imag ** 2) / scale)
        except Exception:
            return 1e10

    t_start = time.perf_counter()
    opt = minimize(
        objective,
        z0,
        method="L-BFGS-B",
        options={
            "maxiter": config['n_iter'],
            "ftol": 1e-15,
            "gtol": 1e-10,
            "eps": config['fd_epsilon'],
        },
    )
    
    z_opt = opt.x.astype(np.float32)
    sigma, mu = decode_z(z_opt)
    elapsed = time.perf_counter() - t_start
    
    return {
        'restart_idx': restart_idx,
        'z_opt': z_opt,
        'sigma': sigma,
        'mu': mu,
        'nit': opt.nit,
        'fun': opt.fun,
        'elapsed': elapsed
    }
