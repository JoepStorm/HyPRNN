import os
import numpy as np
import jax.numpy as jnp
from graded_hole_compression import GradedHoleSimulation
from plotting_utils import plot_interpolation_field
import optuna


class PolarGradedHole(GradedHoleSimulation):
    """Grading based on polar coordinates"""

    def __init__(self, direction, **kwargs):
        super().__init__(**kwargs)
        self._precompute_polar_coords()
        self.direction = direction #kwargs.get('direction', 'maximize')

    def _precompute_polar_coords(self):
        """Precompute polar coordinate quantities used in grading"""
        x = self.qp_coords[:, 0]
        y = self.qp_coords[:, 1]

        # Polar coordinates relative to hole center
        self._dx = x - self.hole_center[0]
        self._dy = y - self.hole_center[1]
        r = np.sqrt(self._dx**2 + self._dy**2)
        self._theta = np.arctan2(self._dy, self._dx)

        # Normalized radial distance: 0 at hole edge, 1 at plate boundary
        r_max = np.sqrt(2) * self.plate_size / 2
        self._r_norm = np.clip((r - self.hole_radius) / (r_max - self.hole_radius), 0, 1)

        # Fourier basis functions that preserve left-right symmetry:
        # sin(θ): symmetric under θ → π-θ (left-right mirror), allows top-bottom asymmetry
        # cos(2θ): symmetric under θ → π-θ, allows diagonal vs cardinal variation
        self._sin1 = np.sin(self._theta)
        self._cos2 = np.cos(2 * self._theta)

    def _fourier_angular_symmetric(self, b, a):
        """Angular modulation preserving left-right symmetry: b1*sin(θ) + a2*cos(2θ)."""
        return b * self._sin1 + a * self._cos2

    def apply_polar_grading(self, design_vars, plot_grading=False):
        """Apply grading with left-right symmetry, top-bottom asymmetry."""
        # Vfrac: radial gradient + symmetric angular modulation
        vfrac_radial = (design_vars['vfrac_inner'] +
                        self._r_norm * (design_vars['vfrac_outer'] - design_vars['vfrac_inner']))
        vfrac_angular = self._fourier_angular_symmetric(
            design_vars['vfrac_b'], design_vars['vfrac_a']
        )
        vfrac = vfrac_radial + vfrac_angular
        vfrac = np.clip(vfrac, 0.1, 0.55)

        # Fiber angle: radial direction + symmetric angular offset
        angle_angular = self._fourier_angular_symmetric(
            design_vars['angle_b'], design_vars['angle_a']
        )
        fiber_angle = self._theta * design_vars['angle_radial_weight'] + angle_angular

        # Ratio: radial gradient + symmetric angular modulation
        ratio_radial = (design_vars['ratio_inner'] +
                        self._r_norm * (design_vars['ratio_outer'] - design_vars['ratio_inner']))
        ratio_angular = self._fourier_angular_symmetric(
            design_vars['ratio_b'], design_vars['ratio_a']
        )
        ratio = ratio_radial + ratio_angular
        ratio = np.clip(ratio, 1.0, 3.0)

        self.micro_variables = {
            'vfrac': jnp.array(vfrac),
            'theta': jnp.array(fiber_angle),
            'ratio': jnp.array(ratio),
            'mu': jnp.ones(self.num_ips) * 100.0,
            'lambda': jnp.ones(self.num_ips) * 150.0,
        }
        self.output_name = "polar_graded"

        if plot_grading:
            plot_interpolation_field(
                domain=self.domain,
                control_coords=None,
                field_dict={
                    'vfrac': self.micro_variables['vfrac'],
                    'theta': self.micro_variables['theta'],
                    'ratio': self.micro_variables['ratio']
                },
                output_folder=f'{self.output_folder}interpolate_field'
            )

    def run_optimization_trial(self, design_vars, plot_grading=False):
        """Run a single optimization trial: reset, apply grading, solve, return squeeze."""
        try:
            self.reset()
            print(f"Problem reset for next trial..")
        except:
            print(f"Running first trial..")

        self.apply_polar_grading(design_vars, plot_grading=plot_grading)
        out = self.run(write_output=False)  # True

        if isinstance(out, bool):
            print("Simulation failed to converge.")
            if self.direction == 'maximize':
                return self.get_hole_squeeze()  # Failed simulations still return their squeeze when maximizing
            else:
                # When minimizing, not converging can be advantageous. So we can't use the actual squeeze, and instead give a fixed penalty.
                return .15

        return self.get_hole_squeeze()


class OptimizationProblem:
    """Wrapper for Optuna optimization of hole squeeze."""

    PARAM_NAMES = [
        # Radial parameters (5)
        'vfrac_inner', 'vfrac_outer',
        'ratio_inner', 'ratio_outer',
        'angle_radial_weight',
        # Symmetric Fourier coefficients (6)
        'vfrac_b', 'vfrac_a',
        'angle_b', 'angle_a',
        'ratio_b', 'ratio_a',
    ]

    def __init__(self, output_folder, prnn_model_loc, macro_meshsize=0.03, direction='maximize'):
        self.output_folder = output_folder
        self.sim = PolarGradedHole(
            macro_meshsize=macro_meshsize,
            use_surrogate=True,
            output_folder=output_folder,
            prnn_model_loc=prnn_model_loc,
            direction=direction
        )

    def objective(self, trial):
        """Optuna objective function."""
        design_vars = {
            # Radial gradients
            'vfrac_inner': trial.suggest_float('vfrac_inner', 0.1, 0.5),
            'vfrac_outer': trial.suggest_float('vfrac_outer', 0.1, 0.5),
            'ratio_inner': trial.suggest_float('ratio_inner', 1.0, 2.5),
            'ratio_outer': trial.suggest_float('ratio_outer', 1.0, 2.5),
            'angle_radial_weight': trial.suggest_float('angle_radial_weight', 0.0, 1.0),
            # Vfrac: b1=top-bottom asymmetry, a2=diagonal variation
            'vfrac_b': trial.suggest_float('vfrac_b', -0.15, 0.15),
            'vfrac_a': trial.suggest_float('vfrac_a', -0.15, 0.15),
            # Angle: b1=top-bottom asymmetry, a2=diagonal variation
            'angle_b': trial.suggest_float('angle_b', -0.5, 0.5),
            'angle_a': trial.suggest_float('angle_a', -0.5, 0.5),
            # Ratio: b1=top-bottom asymmetry, a2=diagonal variation
            'ratio_b': trial.suggest_float('ratio_b', -0.5, 0.5),
            'ratio_a': trial.suggest_float('ratio_a', -0.5, 0.5),
        }

        squeeze = self.sim.run_optimization_trial(design_vars, plot_grading=False)
        print(f"Trial {trial.number}: squeeze = {squeeze:.6f}")
        return squeeze

    def save_results(self, study, save_loc):
        """Save optimization results to CSV."""
        with open(f"{save_loc}/all_trials.csv", "w") as f:
            header = "squeeze," + ",".join(self.PARAM_NAMES) + "\n"
            f.write(header)
            for trial in study.trials:
                if trial.value is not None:
                    params = [str(trial.params[p]) for p in self.PARAM_NAMES]
                    f.write(f"{trial.value}," + ",".join(params) + "\n")


if __name__ == "__main__":
    direction = 'maximize'  # 'minimize' or 'maximize'

    save_loc = f"../results/optimization/polar_grading/{direction}2"
    os.makedirs(save_loc, exist_ok=True)

    output_folder = "../results/optimization/"
    prnn_model_loc = "../trained_models/train_vary_all_v2/prnn_nonlin_1L8_12m_sigmoid/samples512_run1"


    problem = OptimizationProblem(output_folder, prnn_model_loc, direction=direction)

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler, direction=direction)  # maximize
    study.optimize(problem.objective, n_trials=500)

    problem.save_results(study, save_loc)

    print(f"\nBest squeeze: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")


# best: {'vfrac_inner': 0.3195254015709299, 'vfrac_outer': 0.3860757465489678, 'vfrac_diagonal': 0.04110535042865754, 'angle_radial_weight': 0.5448831829968969, 'ratio_inner': 1.635482199008357, 'ratio_outer': 1.9688411695999841, 'ratio_diagonal': -0.06241278873730749}
# [I 2026-01-20 17:38:44,432] Trial 327 finished with value: 0.2026874774606468 and parameters: {'vfrac_inner': 0.47869642810960783, 'vfrac_outer': 0.44070429158154234, 'vfrac_diagonal': 0.11620184447338162, 'angle_radial_weight': 0.5566363482684432, 'ratio_inner': 1.9588880098960788, 'ratio_outer': 2.2920103536429886, 'ratio_diagonal': 0.25402068969827435}. Best is trial 327 with value: 0.2026874774606468.
