from pydantic import BaseModel

def docstring_from_model(model: type[BaseModel]) -> str:
    """Generate a readable string of fields, types, and descriptions from a Pydantic model."""
    lines = []
    for name, field in model.model_fields.items():
        desc = field.description or ""
        # Get type name if possible, else fallback to string repr
        if hasattr(field.annotation, "__name__"):
            typ = field.annotation.__name__
        else:
            typ = str(field.annotation)
        lines.append(f"- {name}: {typ} — {desc}")
    return "\n".join(lines)
