class MissingEnvVariableException(Exception):
    def __init__(self, var: str):
        super().__init__(f"{var} environment variable is not set")
