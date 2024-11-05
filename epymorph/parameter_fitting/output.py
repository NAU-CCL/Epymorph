from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class ParticleFilterOutput:
    num_particles: int
    duration: str
    param_quantiles: Dict[str, List[float]]
    param_values: Dict[str, List[float]]
    true_data: np.ndarray
    model_data: np.ndarray

    def __str__(self) -> str:
        output = []
        output.append("Particle Filter Output")
        output.append("-----------------------")
        output.append(f"Number of Particles: {self.num_particles}")
        output.append(f"Duration: {self.duration}\n")

        output.append("Parameter Quantiles (Showcasing a few values):")
        # Display only a few values for the param_quantiles
        for key, value in list(self.param_quantiles.items())[
            :3
        ]:  # Limit to first 3 key-value pairs
            if isinstance(value[0], list):  # If the value is a list of lists
                output.append(
                    f"{key}: "
                    + "\n".join(
                        f"  {', '.join(f'{v:.3f}' for v in sublist)}"  # type: ignore
                        for sublist in value[:2]  # Limit to the first 2 sublists
                    )
                )
            else:
                output.append(
                    f"{key}: {', '.join(f'{v:.3f}' for v in value[:3])}"
                )  # Limit to first 3 values

        output.append("\nParameter Values (Showcasing a few values):")
        # Display only a few values for the param_values
        for key, value in list(self.param_values.items())[
            :3
        ]:  # Limit to first 3 key-value pairs
            output.append(
                f"{key}: {', '.join(f'{v:.3f}' for v in value[:3])}"
            )  # Limit to first 3 values

        output.append("\nTrue Data:")
        output.extend(
            f"  {val:.3f}" for val in self.true_data[:5]
        )  # Show first 5 values for true data

        output.append("\nModel Data:")
        output.extend(
            f"  {val:.3f}" for val in self.model_data[:5]
        )  # Show first 5 values for model data

        return "\n".join(output)
