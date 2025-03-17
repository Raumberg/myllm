from tabulate import tabulate

def tabula(data: dict) -> str:
    table = []
    for key, value in data.items():
        if isinstance(value, list):
            value = ', '.join(map(str, value))
        elif isinstance(value, dict):
            value = str(value)
        table.append([key, value])
    return tabulate(
        table,
        headers=["Config key", "Config value"],
        tablefmt="grid"
    )