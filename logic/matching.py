import math

def match_with_reason(solution, phase=None, sensitive=None, collab=None, volume=None):
    """
    Determine suitability of a storage solution.
    Returns: (ok: bool, reasons: list[str])
    """
    reasons = []

    # --- 1️⃣ Onderzoeksfase ---
    lifecycle = solution.get("positioning_datalyfecycle", [])
    if isinstance(lifecycle, str):
        lifecycle = [lifecycle]
    lifecycle_text = " ".join(lifecycle).lower()

    if phase:
        if phase == "active":
            if not any(term in lifecycle_text for term in [
                "collect", "create", "collaborate", "process", "analyse"
            ]):
                reasons.append("Not suitable for active research phase")
        elif phase == "preservation":
            if not any(term in lifecycle_text for term in [
                "archive", "publish", "share", "evaluate", "reuse"
            ]):
                reasons.append("Not suitable for archiving or preservation phase")

    # --- 2️⃣ Gevoelige data ---
    if sensitive and sensitive.lower() == "yes":
        sensitive_ok = False
        for key in [
            "sensitivity_classification",
            "availability_classification",
            "GDPR_compliant",
            "Data CIA",
            "file_encryption (by default)"
        ]:
            val = str(solution.get(key, "")).lower()
            if any(term in val for term in [
                "yes", "true", "sensitive", "high", "gdpr", "protected", "confidential"
            ]):
                sensitive_ok = True
                break
        if not sensitive_ok:
            reasons.append("Not suitable for sensitive or personal data")

    # --- 3️⃣ Samenwerking ---
    if collab:
        collab_field = " ".join([
            str(solution.get("collaboration", "")),
            str(solution.get("collaboration_notes", "")),
            str(solution.get("ownershiptransfer", "")),
            str(solution.get("sync_equipment", "")),
            str(solution.get("sync_HPC", "")),
            str(solution.get("sync_R/JupyterNsync", "")),
        ]).lower()

        if collab == "internal":
            if not any(term in collab_field for term in [
                "internal", "within", "uu", "institution", "organization", "department"
            ]):
                reasons.append("No internal collaboration support")
        elif collab == "external":
            if not any(term in collab_field for term in [
                "external", "partner", "outside", "third", "other institution", "cross"
            ]):
                reasons.append("No external collaboration support")

    # --- 4️⃣ Datavolume ---
    # --- 4️⃣ Datavolume ---
    # --- 4️⃣ Datavolume (Small <1TB vs Large ≥1TB) ---
    if volume:
        capacity = str(solution.get("capacity", "")).lower()

        # Heuristiek:
        # - Large: als capacity iets zegt als "tb", "pb", "petabyte", "100 tb", "> 1 pb", "unlimited", "large"
        # - Anders: aannemen dat het in elk geval small aankan
        supports_large = False

        large_markers = ["tb", "pb", "petabyte", "unlimited", "> 1", "100", "500", "1 pb", "large"]
        if any(m in capacity for m in large_markers):
            supports_large = True

        # Small is eigenlijk altijd ok (conservatief), tenzij je ooit expliciet kleine limieten hebt.
        supports_small = True

        if volume == "large" and not supports_large:
            reasons.append("Not suitable for large data volumes (≥ 1 TB)")
        elif volume == "small" and not supports_small:
            reasons.append("Not suitable for small data volumes (< 1 TB)")

    ok = len(reasons) == 0
    return ok, reasons


def sanitize_for_json(obj):
    """Remove NaN and Inf from objects before returning JSON."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    else:
        return obj
