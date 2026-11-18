"""Compare scientific task context while preserving producer provenance."""


def matches_task_context(actual, expected):
    if not isinstance(actual, dict):
        return False
    operation = actual.get('operation')
    expected_operation = expected.get('operation')
    if (not isinstance(operation, dict) or not isinstance(expected_operation, dict)
        or set(operation) != set(expected_operation)
        or not isinstance(operation['profile_id'], str) or not operation['profile_id']
        or operation['role'] != expected_operation.get('role')
        or {key:value for key,value in operation.items() if key != 'profile_id'}
           != {key:value for key,value in expected_operation.items() if key != 'profile_id'}):
        return False
    # A later screen may have no execution binding for its completed sample.
    # Its dormant/current profile must not redefine the producer's provenance.
    return actual == {**expected, 'operation': operation}
