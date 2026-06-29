import os

from app.common.exceptions import MissingEnvVariableException


def load_env(var: str) -> str:
    env = os.getenv(var)

    if env is None:
        raise MissingEnvVariableException(var)

    return env


def load_env_or(var: str, or_value: str) -> str:
    return os.environ.get(var, or_value)


def unwrap[T](val: T | None) -> T:
    assert val is not None
    return val
