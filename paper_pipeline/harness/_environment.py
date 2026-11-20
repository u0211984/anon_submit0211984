import os

def reject_model_environment():
    forbidden = sorted(name for name in os.environ
                       if name.startswith(("RAPID_", "OPMODEL_", "COUNTS_", "JOIN_")))
    if forbidden:
        raise SystemExit("ambient model variables are forbidden; use CLI/YAML inputs: "
                         + ", ".join(forbidden))
