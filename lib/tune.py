from ray import tune


def _tune_config(params, cli_report_columns=[]):
    result = {}
    for key, val in params.items():
        if isinstance(val, dict):       # not an end-point
            val, _ = _tune_config(val, cli_report_columns)
            if not val:                 #
                continue
        if isinstance(val, dict):       #
            result[key] = val
        else:
            if str(val).startswith('tune'):
                result[key] = eval(val)
                cli_report_columns.append(key)
            else:
                result[key] = tune.choice([val])
    return result, cli_report_columns


def search(d, k, path=None):
    if path is None:
        path = []

    # Reached bottom of dict - no good
    if not isinstance(d, dict):
        return False

    # Found it!
    if k in d.keys():
        path.append(k)
        return path

    else:
        check = list(d.keys())
        # Look in each key of dictionary
        while check:
            first = check[0]
            # Note which we just looked in
            path.append(first)
            if search(d[first], k, path) is not False:
                break
            else:
                # Move on
                check.pop(0)
                path.pop(-1)
        else:
            return False
        return path


def convert_to_tune_config(hparams):
    config, tune_param_names = _tune_config(hparams)
    report_columns = {'/'.join(search(config, tune_param)): search(config, tune_param)[-1]
                      for tune_param in tune_param_names}
    return config, report_columns
