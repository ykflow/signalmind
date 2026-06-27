from enum import Enum, auto


class ModelType(Enum):
    LOCATION = auto()
    SCALE = auto()
    LOCATION_SCALE = auto()


class ProcessName(Enum):
    WHITE_NOISE = "White-Noise"
    RANDOM_WALK = "Random Walk"
    RANDOM_WALK_DRIFT= "Random Walk + Drift"
    RANDOM_WALK_NOISE = "Random Walk + Noise"
    AR1 = "AR(1)"
    AR1_NOISE = "AR(1) + Noise"
    MA1 = "MA(1)"
    MA1_NOISE = "MA(1) + Noise"
    GARCH11 = "GARCH(1,1)"
    LOG_VAR_AR1 = "Log-Var-AR(1)"
    LOG_VAR_2FAR1 = "Log-Var-2FAR(1)"

    @classmethod
    def from_string(cls, string_value):
        """Performs a reverse lookup to find an Enum member by its string value.

        Args:
            string_value: The exact string value defined in the enum (e.g., "AR(1) + Noise").
        """
        # Build the mapping dictionary locally to bypass Enum member assignment rules
        value_map = {member.value: member for member in cls}
        if string_value in value_map:
            return value_map[string_value]

        raise ValueError(f"'{string_value}' is not a recognized ProcessName. "
                         f"Valid options are: {list(value_map.keys())}")
