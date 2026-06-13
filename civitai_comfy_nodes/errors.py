import json


class CivitaiNodeError(Exception):
    """Raised for any Civitai orchestration failure; the message is shown in the ComfyUI error popup."""


def http_error_message(status_code: int, body_text: str) -> str:
    detail = body_text
    field_errors = ""
    try:
        problem = json.loads(body_text)
        if isinstance(problem, dict):
            detail = problem.get("detail") or problem.get("title") or body_text
            errors = problem.get("errors")
            if isinstance(errors, dict):
                lines = []
                for field, messages in errors.items():
                    if isinstance(messages, list):
                        lines.extend(f"{field}: {m}" for m in messages)
                    else:
                        lines.append(f"{field}: {messages}")
                field_errors = "\n" + "\n".join(lines)
    except (json.JSONDecodeError, TypeError):
        pass

    if status_code == 401:
        hint = " (token missing, expired, or invalid — check your Civitai API token or re-login)"
    elif status_code == 402:
        hint = " (insufficient Buzz)"
    elif status_code == 429:
        hint = " (rate limited — slow down)"
    else:
        hint = ""
    return f"Civitai API error {status_code}{hint}: {detail}{field_errors}"


def workflow_failure_message(workflow: dict) -> str:
    status = workflow.get("status", "unknown")
    workflow_id = workflow.get("id", "?")
    reasons: list[str] = []
    for step in workflow.get("steps") or []:
        output = step.get("output") or {}
        for err in output.get("errors") or []:
            reasons.append(str(err))
        for job in step.get("jobs") or []:
            if job.get("reason"):
                reasons.append(str(job["reason"]))
            if job.get("blockedReason"):
                reasons.append(f"blocked: {job['blockedReason']}")
    detail = "; ".join(dict.fromkeys(reasons)) or "no failure details reported"
    return f"Civitai workflow {workflow_id} {status}: {detail}"
