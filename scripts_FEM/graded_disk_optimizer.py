"""
Optimization of radial grading for pressurized annulus.

Optimizes vfrac, r_combi, and (optionally) mu at N radial control points
to minimize peak von Mises stress under internal pressure.

The grading is 1D: parameters are defined at N control radii and linearly
interpolated to all quadrature points. Same profile for all angles.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
from material_params import FUNGI_MU, FUNGI_LAMBDA
from graded_disk_setup import (
    GradedDiskSimulation, plot_grading_overview, plot_grading_comparison,
    plot_stress_comparison, plot_radial_slice,
)
import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use(['science', 'bright'])
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
rc('text', usetex=True)

class GradedDiskOptimizer:
    """Gradient-based optimization of radial grading for a pressurized disk.

    Parameters are vfrac and r_combi (and optionally mu) at N_ctrl radial
    control points, linearly interpolated to quadrature points.
    """

    def __init__(self, output_folder, prnn_model_loc,
                 macro_meshsize=0.02, R_inner=0.2, R_outer=0.5,
                 n_ctrl=2, optimize_mu=False, fd_step=1e-3,
                 max_displacement=None):
        self.output_folder = output_folder
        self.fd_step = fd_step
        self.n_ctrl = n_ctrl
        self.optimize_mu = optimize_mu
        self.max_displacement = max_displacement
        os.makedirs(output_folder, exist_ok=True)

        self.sim = GradedDiskSimulation(
            macro_meshsize=macro_meshsize,
            R_inner=R_inner,
            R_outer=R_outer,
            output_folder=output_folder,
            prnn_model_loc=prnn_model_loc,
        )

        # Radial control points and QP radii for interpolation
        self.r_ctrl = np.linspace(R_inner, R_outer, n_ctrl)
        self.r_qp = np.sqrt(self.sim.qp_coords[:, 0]**2 + self.sim.qp_coords[:, 1]**2)

        # Parameter layout: [vfrac (n_ctrl) | r_combi (n_ctrl) | mu (n_ctrl, optional)]
        self.n_params = n_ctrl * (3 if optimize_mu else 2)

        # Bounds: vfrac in [0.1, 0.5], r_combi in [0.4, 2.5], mu in [0.1, 3.0]
        self.bounds = (
            [(0.1, 0.5)] * n_ctrl
            + [(0.4, 2.5)] * n_ctrl
            + ([(0.1, FUNGI_MU * 5)] * n_ctrl if optimize_mu else [])
        )

        self.eval_count = 0
        self.history = []
        self.worst_loss = 0.0

    def _unpack(self, x):
        vfrac = x[:self.n_ctrl]
        r_combi = x[self.n_ctrl:2 * self.n_ctrl]
        mu = x[2 * self.n_ctrl:] if self.optimize_mu else np.full(self.n_ctrl, 1.30)
        return vfrac, r_combi, mu

    def _to_qp(self, ctrl_values):
        """Linearly interpolate control-point values to QP radii."""
        return np.interp(self.r_qp, self.r_ctrl, ctrl_values)

    def _apply(self, x):
        vfrac, r_combi, mu = self._unpack(x)
        self.sim.apply_radial_grading(
            self._to_qp(vfrac),
            self._to_qp(r_combi),
            self._to_qp(mu),
        )

    def _objective(self, x):
        try:
            self.sim.reset()
        except Exception:
            pass

        self._apply(x)
        out = self.sim.run(write_output=False)

        if isinstance(out, bool):
            loss = self.worst_loss * 1.5 if self.worst_loss > 0 else 2.0
            print(f"  eval {self.eval_count + 1}: FAILED, penalty = {loss:.6f}")
        else:
            vm = self.sim.get_von_mises_at_qp()
            loss = float(vm.max())

            if self.max_displacement is not None:
                max_disp = self.sim.get_max_inner_displacement()
                if max_disp > self.max_displacement:
                    violation = (max_disp - self.max_displacement) / self.max_displacement
                    penalty = 100.0 * violation
                    loss += penalty
                    print(f"    disp penalty: max_disp={max_disp:.4f} > {self.max_displacement:.4f}, penalty={penalty:.2f} for total loss of {loss:.2f}")

            self.worst_loss = max(self.worst_loss, loss)

        self.eval_count += 1
        self.history.append(loss)
        print(f"  eval {self.eval_count}: loss = {loss:.6f}")
        return loss

    def initial_guess(self, strategy='uniform'):
        if strategy == 'uniform':
            vfrac0 = np.full(self.n_ctrl, 0.3)
            r_combi0 = np.ones(self.n_ctrl)
            mu0 = np.full(self.n_ctrl, 1.30) if self.optimize_mu else np.array([])
        elif strategy == 'graded':
            t = np.linspace(0, 1, self.n_ctrl)
            vfrac0 = 0.1 + t * (0.5 - 0.1)
            r_combi0 = 0.4 + t * (2.5 - 0.4)
            mu0 = np.full(self.n_ctrl, 1.30) if self.optimize_mu else np.array([])
        elif strategy == 'random':
            rng = np.random.default_rng(42)
            vfrac0 = rng.uniform(0.1, 0.5, self.n_ctrl)
            r_combi0 = rng.uniform(0.4, 2.5, self.n_ctrl)
            mu0 = rng.uniform(0.1, 3.0, self.n_ctrl) if self.optimize_mu else np.array([])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return np.concatenate([vfrac0, r_combi0, mu0])

    def run_lbfgsb(self, x0=None, maxiter=50):
        if x0 is None:
            x0 = self.initial_guess()

        print(f"\n{'='*60}")
        print(f"L-BFGS-B: {self.n_params} params, FD step {self.fd_step}")
        print(f"{'='*60}\n")

        self.eval_count = 0
        self.history = []

        result = minimize(
            self._objective,
            x0,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={
                'maxiter': maxiter,
                'ftol': 1e-8,
                'gtol': 1e-6,
                'eps': self.fd_step,
                'disp': True,
                'maxfun': maxiter * (self.n_params + 1) * 2,
            },
        )

        print(f"\nL-BFGS-B done: {result.message}")
        print(f"Final loss: {result.fun:.6f}, evals: {self.eval_count}")
        return result

    def run_cmaes(self, x0=None, sigma0=0.3, maxiter=200):  #, popsize=None):
        try:
            import cma
        except ImportError:
            raise ImportError("Install cma: pip install cma")

        if x0 is None:
            x0 = self.initial_guess()
        # if popsize is None:     # default
        #     popsize = 4 + int(3 * np.log(self.n_params))

        lower = np.array([b[0] for b in self.bounds])
        upper = np.array([b[1] for b in self.bounds])

        print(f"\n{'='*60}")
        print(f"CMA-ES: {self.n_params} params, sigma0={sigma0}") #, popsize={popsize}")
        print(f"{'='*60}\n")

        self.eval_count = 0
        self.history = []

        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            'maxiter': maxiter,
            # 'popsize': popsize,
            'bounds': [list(lower), list(upper)],
            'seed': 42,
            'verb_disp': 1,
            'verb_log': 0,
            'tolfun': 1e-8,
        })

        while not es.stop():
            solutions = es.ask()
            fitnesses = [self._objective(s) for s in solutions]
            es.tell(solutions, fitnesses)
            es.disp()

        result = es.result
        print(f"\nCMA-ES done: {self.eval_count} evals, best loss: {result.fbest:.6f}")
        return result

    def run_hybrid(self, maxiter_cma=100, maxiter_lbfgsb=30, sigma0=0.3):
        """CMA-ES global search followed by L-BFGS-B local refinement."""
        print("Phase 1: CMA-ES global search")
        cma_result = self.run_cmaes(sigma0=sigma0, maxiter=maxiter_cma)
        history_cma = self.history.copy()

        if maxiter_lbfgsb <= 0:
            print("\nPhase 2: L-BFGS-B skipped (maxiter_lbfgsb=0)")
            result = SimpleNamespace(x=cma_result.xbest, fun=cma_result.fbest)
            return result, history_cma, []

        print("\nPhase 2: L-BFGS-B local refinement")
        lbfgsb_result = self.run_lbfgsb(x0=cma_result.xbest, maxiter=maxiter_lbfgsb)
        history_lbfgsb = self.history.copy()

        return lbfgsb_result, history_cma, history_lbfgsb

    def save_results(self, x_best, loss_best, history, fname='optimization_results.npz'):
        vfrac, r_combi, mu = self._unpack(x_best)
        path = os.path.join(self.output_folder, fname)
        np.savez(path,
                 x_best=x_best,
                 vfrac_best=vfrac,
                 r_combi_best=r_combi,
                 mu_best=mu,
                 loss_best=loss_best,
                 history=np.array(history),
                 r_ctrl=self.r_ctrl)
        print(f"Saved: {path}")

    def plot_convergence(self, history, n_cma=None, fname='convergence.pdf'):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(history, '-', lw=1, alpha=0.4, color=colours[0], label='All evaluations')
        best_so_far = np.minimum.accumulate(history)
        ax.plot(best_so_far, '-', lw=2, color=colours[1], label='Best so far')
        ax.set_ylim(ymax=max(best_so_far)*1.5)
        if n_cma is not None and 0 < n_cma < len(history):
            ax.axvline(n_cma, ls='--', color='black', lw=0.8, alpha=0.6)
            ax.text(n_cma, ax.get_ylim()[1], '  L-BFGS-B', va='top', fontsize=8, color='black')
            ax.text(n_cma, ax.get_ylim()[1], 'CMA-ES  ', va='top', ha='right', fontsize=8, color='black')
        ax.set_xlabel('Function evaluation')
        ax.set_ylabel('Loss')
        # ax.set_yscale('log')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        path = os.path.join(self.output_folder, fname)
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def plot_radial_profiles(self, x_best, fname='radial_profiles.pdf'):
        """Plot vfrac, r_combi (and mu if optimized) vs radius at control points."""
        vfrac, r_combi, mu = self._unpack(x_best)
        r = self.r_ctrl

        # Bounds per field: (lo, hi)
        bound_vfrac = (0.1, 0.5)
        bound_rc = (0.4, 2.5)
        bound_mu = (0.1, FUNGI_MU * 5)
        margin = 0.12  # relative padding beyond bounds

        panels = [
            (vfrac, r'$V_f$', bound_vfrac, colours[0]),
            (r_combi, r'$r_{\mathrm{combi}}$', bound_rc, colours[1]),
        ]
        if self.optimize_mu:
            panels.append((mu, r'$\mu$', bound_mu, colours[2]))

        fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.5))

        for ax, (vals, ylabel, (lo, hi), col) in zip(axes, panels):
            ax.plot(r, vals, 'o-', color=col)
            ax.axhline(lo, ls=':', color='black', lw=0.8)
            ax.axhline(hi, ls=':', color='black', lw=0.8)
            span = hi - lo
            ax.set_ylim(lo - margin * span, hi + margin * span)
            ax.set_xlabel('distance')
            ax.set_ylabel(ylabel)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # r_combi: add isotropic reference line and labels
        ax_rc = axes[1]
        ax_rc.axhline(1.0, ls='--', color='gray', lw=0.8)
        ax_rc.text(r[-1], 1.05, 'radial', ha='right', fontsize=8, color='gray')
        ax_rc.text(r[-1], 0.85, 'circ.', ha='right', fontsize=8, color='gray')

        plt.tight_layout()
        path = os.path.join(self.output_folder, fname)
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def _apply_random(self, vfrac, mu=FUNGI_MU, seed=42):
        """Per-element random fiber angle and shape; homogeneous vfrac and mu."""
        rng = np.random.default_rng(seed)
        n = self.sim.num_ips
        theta = rng.uniform(0.0, np.pi, n)  # orientation is pi-periodic
        ratio = rng.uniform(1.0, 2.5, n)
        self.sim.micro_variables = {
            'vfrac': jnp.full(n, float(vfrac)),
            'theta': jnp.array(theta),
            'ratio': jnp.array(ratio),
            'mu': jnp.full(n, float(mu)),
            'lambda': jnp.full(n, FUNGI_LAMBDA),
        }
        self.sim.output_name = "random"
        # For grading-comparison plots: store ratio magnitude as r_combi and
        # mark random mode so the plot uses theta directly (not phi-encoded).
        self.sim._r_combi = np.where(ratio >= 1.0, ratio, 1.0 / ratio)
        self.sim._random_mode = True

    def run_random_baseline(self, mu=FUNGI_MU, seed=42, tol=1e-3,
                            vfrac_lo=0.1, vfrac_hi=0.5):
        """Bisection over homogeneous vfrac to just satisfy the displacement cap.

        Returns (vfrac_best, vm, du). Higher vfrac → stiffer → less displacement,
        so we seek the smallest vfrac with max_disp ≤ max_displacement.
        """
        assert self.max_displacement is not None, "need max_displacement for random baseline"
        print(f"\n{'#'*60}\n# Random baseline: bisection on vfrac\n{'#'*60}")

        def eval_disp(vfrac):
            try:
                self.sim.reset()
            except Exception:
                pass
            self._apply_random(vfrac, mu=mu, seed=seed)
            out = self.sim.run(write_output=False)
            if isinstance(out, bool):
                return np.inf, None, None
            vm = self.sim.get_von_mises_at_qp()
            disp = self.sim.get_max_inner_displacement()
            print(f"  vfrac={vfrac:.4f} → max_disp={disp:.4f}, peak VM={vm.max():.4f}")
            return disp, vm, None

        # Check the bounds first
        d_hi, _, _ = eval_disp(vfrac_hi)
        if d_hi > self.max_displacement:
            print(f"  Even vfrac={vfrac_hi} exceeds displacement cap; using upper bound.")
            lo = hi = vfrac_hi
        else:
            d_lo, _, _ = eval_disp(vfrac_lo)
            if d_lo <= self.max_displacement:
                print(f"  Even vfrac={vfrac_lo} satisfies cap; using lower bound.")
                lo = hi = vfrac_lo
            else:
                lo, hi = vfrac_lo, vfrac_hi
                while hi - lo > tol:
                    mid = 0.5 * (lo + hi)
                    d_mid, _, _ = eval_disp(mid)
                    if d_mid <= self.max_displacement:
                        hi = mid
                    else:
                        lo = mid

        vfrac_best = hi
        # Final evaluation + write VTX output
        try:
            self.sim.reset()
        except Exception:
            pass
        self._apply_random(vfrac_best, mu=mu, seed=seed)
        self.sim.run(write_output=True, write_name='disk_random')
        vm = self.sim.get_von_mises_at_qp()
        du = self.sim.get_nodal_displacements()
        print(f"Random — vfrac={vfrac_best:.4f}, peak VM={vm.max():.4f}, "
              f"max_disp={self.sim.get_max_inner_displacement():.4f}")
        return vfrac_best, vm, du

    def run_uniform_baseline(self, vfrac=0.3, r_combi=1.75, mu=1.30):
        """Run uniform grading (constant params) and return (vm, displacements)."""
        try:
            self.sim.reset()
        except Exception:
            pass
        self.sim.apply_radial_grading(
            np.full(self.sim.num_ips, vfrac),
            np.full(self.sim.num_ips, r_combi),
            np.full(self.sim.num_ips, mu),
        )
        self.sim.run(write_output=True, write_name='disk_uniform')
        vm = self.sim.get_von_mises_at_qp()
        du = self.sim.get_nodal_displacements()
        print(f"Uniform — peak VM: {vm.max():.4f}, mean: {vm.mean():.4f}")
        return vm, du

    def visualize_best(self, x_best):
        """Re-run best solution with VTX output, grading overview, and profile plots."""
        try:
            self.sim.reset()
        except Exception:
            pass
        self._apply(x_best)
        self.sim.run(write_output=True, write_name='disk_optimized')

        vm = self.sim.get_von_mises_at_qp()
        du = self.sim.get_nodal_displacements()
        print(f"Optimized — peak VM: {vm.max():.4f}, mean: {vm.mean():.4f}")
        plot_grading_overview(self.sim, self.output_folder, radial_1d=True)
        # plot_radial_slice(self.sim, self.output_folder)
        self.plot_radial_profiles(x_best)
        return vm, du

    def visualize_grading_only(self, x_best):
        """Apply grading and plot overview + radial profiles without running the sim."""
        self._apply(x_best)
        plot_grading_overview(self.sim, self.output_folder, radial_1d=True)
        self.plot_radial_profiles(x_best)


def visualize_saved(output_folder, prnn_model_loc, n_ctrl, label, fname='optimization_results.npz', **kwargs):
    """Load saved x_best from a previous run and plot grading only (no FEM solve)."""
    opt = GradedDiskOptimizer(
        output_folder=output_folder,
        prnn_model_loc=prnn_model_loc,
        n_ctrl=n_ctrl,
        **kwargs,
    )
    path = os.path.join(output_folder, fname)
    data = np.load(path)
    x_best = data['x_best']
    loss_best = float(data['loss_best'])
    print(f"\n{'#'*60}")
    print(f"# {label}: loaded {path}")
    print(f"#   loss_best = {loss_best:.6f}, n_params = {len(x_best)}")
    print(f"{'#'*60}")

    opt.visualize_grading_only(x_best)

    vfrac_best, r_combi_best, mu_best = opt._unpack(x_best)
    print(f"\n{label} loaded result:")
    print(f"  vfrac:   {np.round(vfrac_best, 3)}")
    print(f"  r_combi: {np.round(r_combi_best, 3)}")
    if opt.optimize_mu:
        print(f"  mu:      {np.round(mu_best, 3)}")

    return opt, x_best


def run_optimization(output_folder, prnn_model_loc, n_ctrl, label,
                     maxiter_cma=100, maxiter_lbfgsb=5, **kwargs):
    """Run hybrid optimization and return (optimizer, x_best, vm, du)."""
    opt = GradedDiskOptimizer(
        output_folder=output_folder,
        prnn_model_loc=prnn_model_loc,
        n_ctrl=n_ctrl,
        **kwargs,
    )
    print(f"\n{'#'*60}")
    print(f"# {label}: {opt.n_params} params  "
          f"({opt.n_ctrl} ctrl pts × {'3' if opt.optimize_mu else '2'} fields)")
    print(f"{'#'*60}")

    result, hist_cma, hist_lbfgsb = opt.run_hybrid(
        maxiter_cma=maxiter_cma, maxiter_lbfgsb=maxiter_lbfgsb, sigma0=0.3,
    )
    x_best = result.x
    full_history = hist_cma + hist_lbfgsb

    if full_history:
        opt.save_results(x_best, float(min(full_history)), full_history)
        opt.plot_convergence(full_history, n_cma=len(hist_cma))

    # Re-run best with VTX output + plots
    vm, du = opt.visualize_best(x_best)

    vfrac_best, r_combi_best, mu_best = opt._unpack(x_best)
    print(f"\n{label} result:")
    print(f"  vfrac:   {np.round(vfrac_best, 3)}")
    print(f"  r_combi: {np.round(r_combi_best, 3)}")
    if opt.optimize_mu:
        print(f"  mu:      {np.round(mu_best, 3)}")
    print(f"  peak VM: {vm.max():.4f}, mean: {vm.mean():.4f}")

    return opt, x_best, vm, du


def main_visualize_saved():
    """Load previously saved optimization results and re-plot grading only."""
    base_folder = "../results/graded_disk/optimized/v3"
    prnn_model_loc = "../trained_models/train_vary_all_v2/prnn_nonlin_1L8_12m_sigmoid/samples512_run1"

    common_kwargs = dict(
        macro_meshsize=0.03,
        optimize_mu=True,
        fd_step=1e-3,
        max_displacement=0.05,
    )

    opt_hom, _ = visualize_saved(
        output_folder=f"{base_folder}/homogeneous",
        prnn_model_loc=prnn_model_loc,
        n_ctrl=1,
        label="Uniform",
        **common_kwargs,
    )
    opt_grad, _ = visualize_saved(
        output_folder=f"{base_folder}/graded",
        prnn_model_loc=prnn_model_loc,
        n_ctrl=4,
        label="Graded",
        **common_kwargs,
    )

    plot_grading_comparison(opt_hom.sim, opt_grad.sim, output_folder=base_folder)


if __name__ == "__main__":
    # To re-plot previous results without re-running the optimization,
    # uncomment this line:
    # main_visualize_saved(); import sys; sys.exit(0)

    base_folder = "../results/graded_disk/optimized/v5"
    prnn_model_loc = "../trained_models/train_vary_all_v2/prnn_nonlin_1L8_12m_sigmoid/samples512_run1"

    common_kwargs = dict(
        macro_meshsize=0.03,
        optimize_mu=True,
        fd_step=1e-3,
        max_displacement=0.05,
        maxiter_cma = 100,
        maxiter_lbfgsb = 0,
    )

    # --- Case 0: random control (homogeneous vfrac/mu, per-element random theta/ratio) ---
    init_kwargs = {k: v for k, v in common_kwargs.items()
                   if k not in ('maxiter_cma', 'maxiter_lbfgsb')}
    opt_rand = GradedDiskOptimizer(
        output_folder=f"{base_folder}/random",
        prnn_model_loc=prnn_model_loc,
        n_ctrl=1,
        **init_kwargs,
    )
    vfrac_rand, vm_rand, du_rand = opt_rand.run_random_baseline(mu=1.5, seed=42)

    # --- Optimization 1: homogeneous (n_ctrl=1) ---
    opt_hom, x_hom, vm_hom, du_hom = run_optimization(
        output_folder=f"{base_folder}/homogeneous",
        prnn_model_loc=prnn_model_loc,
        n_ctrl=1,
        label="Uniform",
        **common_kwargs,
    )

    # --- Optimization 2: graded (n_ctrl=4) ---
    opt_grad, x_grad, vm_grad, du_grad = run_optimization(
        output_folder=f"{base_folder}/graded",
        prnn_model_loc=prnn_model_loc,
        n_ctrl=4,
        label="Graded",
        **common_kwargs,
    )

    # --- Comparison: all three cases ---
    plot_stress_comparison(
        opt_grad.sim,
        output_folder=f"{base_folder}/", disp_scale=1,
        cases=[
            (vm_rand, du_rand, 'Random'),
            (vm_hom, du_hom, 'Uniform'),
            (vm_grad, du_grad, 'Graded'),
        ],
    )
    plot_grading_comparison(
        None, None, output_folder=base_folder, disp_scale=1.0,
        cases=[
            ('Random',      opt_rand.sim, vm_rand, du_rand),
            ('Uniform', opt_hom.sim,  vm_hom,  du_hom),
            ('Graded',      opt_grad.sim, vm_grad, du_grad),
        ],
    )

    print(f"\n{'='*60}")
    print(f"Random      — peak VM: {vm_rand.max():.4f}, mean: {vm_rand.mean():.4f}  (vfrac={vfrac_rand:.4f})")
    print(f"Homogeneous — peak VM: {vm_hom.max():.4f}, mean: {vm_hom.mean():.4f}")
    print(f"Graded      — peak VM: {vm_grad.max():.4f}, mean: {vm_grad.mean():.4f}")
    if vm_rand.max() > 0:
        print(f"Peak reduction vs random: "
              f"homogeneous {100*(1 - vm_hom.max()/vm_rand.max()):.1f}%, "
              f"graded {100*(1 - vm_grad.max()/vm_rand.max()):.1f}%")
    if vm_hom.max() > 0:
        print(f"Graded vs homogeneous peak reduction: "
              f"{100*(1 - vm_grad.max()/vm_hom.max()):.1f}%")
    print(f"{'='*60}")
