import os.path as osp
import pandas as pd

from dro import keys
from dro.utils.training_utils import make_model_uid


def _dict_to_row(d):
    return pd.DataFrame.from_dict(d, orient="index").T


class Report:
    def __init__(self, flags):
        self.results_list = list()
        # Set is_adversarial=True when generating the model_uid so that the adversarial
        # parameters (attack type, epsilon, etc) will be recorded in the uid.
        self.uid = make_model_uid(flags, is_adversarial=True)
        self.metric = keys.ACC  # the name of the metric being recorded
        self.results = None

    def add_result(self, result: dict):
        """Record the results of an experiment."""
        # Check for duplicates; while this is not strictly a problem, it is almost
        # definitely a mistake if duplicate results are being added.
        result_row = _dict_to_row(result)
        if self.results is None:  # case: this is first entry; initialize the dataframe
            self.results = result_row
        else:
            self.results = pd.concat([self.results, result_row])
        return

    def to_csv(self, metrics_dir, attr_name=None):
        """Write the current results to a CSV file in metrics_dir."""
        if attr_name:
            csvname = "{}-{}.csv".format(self.uid, attr_name)
        else:
            csvname = "{}.csv".format(self.uid)
        fp = osp.join(metrics_dir, csvname)
        print("[INFO] writing results to {}".format(fp))
        print(self.results)
        self.results.to_csv(fp, index=False)
        return
