import pandas as pd

from dro import keys
from dro.utils.training_utils import make_model_uid
from scripts.train_vggface2_cleverhans import FLAGS


class Report:
    def __init__(self):
        self.results_list = list()
        # Set is_adversarial=True when generating the model_uid so that the adversarial
        # parameters (attack type, epsilon, etc) will be recorded in the uid.
        self.uid = make_model_uid(FLAGS, is_adversarial=True)
        self.metric = keys.ACC  # the name of the metric being recorded

    def add_result(self, val, model, data, phase):
        """Record the results of an experiment."""
        results_entry = (self.uid, self.metric, val, model, data, phase)
        # Check for duplicates; while this is not strictly a problem, it is almost
        # definitely a mistake if duplicate results are being added.
        assert results_entry not in self.results_list, "duplicate results added to report"
        self.results_list.append(results_entry)
        return

    def to_csv(self):
        df = pd.DataFrame(self.results_list, columns=["uid", "metric", "value",
                                                      "model", "data", "phase"])
        fp = "./metrics/{}.csv".format(self.uid)
        print("[INFO] writing results to {}".format(fp))
        print(df)
        df.to_csv(fp, index=False)
        return