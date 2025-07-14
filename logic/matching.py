import math

def match_with_reason(solution, sensitive, collab, volume):
    reasons = []


    # --- Sensitive data matching ---
    if sensitive and sensitive.lower() == "yes":
        if solution.get("SupportsSensitive") is False:
            reasons.append("Data not suitable for sensitive information")

    # --- Collaboration matching ---
    collab_field = solution.get("Sharing and Collaboration", "")
    if not isinstance(collab_field, str):
        collab_field = ""
    collab_text = collab_field.lower()

    if collab:
        collab = collab.lower()
        if collab == "internal":
            if not any(term in collab_text for term in ["internal", "binnen", "uu", "institutional", "within"]):
                reasons.append("No support for internal collaboration")
        elif collab == "external":
            if not any(term in collab_text for term in ["external", "partners", "extern", "outside", "third parties", "external institutions"]):
                reasons.append("No support for external collaboration")

    # --- Volume matching ---
    if volume:
        volume = volume.lower()
        if volume == "large":
            if solution.get("SupportsLarge") is False:
                reasons.append("Not suitable for large data volumes")
        elif volume == "medium":
            if solution.get("SupportsMedium") is False:
                reasons.append("Not suitable for medium data volumes")
        elif volume == "small":
            if solution.get("SupportsSmall") is False:
                reasons.append("Not suitable for small data volumes")

    ok = len(reasons) == 0
    return ok, reasons


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    else:
        return obj
