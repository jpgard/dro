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
        self.results = None

    def add_result(self, result: dict):
        """Record the results of an experiment."""
        # Check for duplicates; while this is not strictly a problem, it is almost
        # definitely a mistake if duplicate results are being added.
        if not self.results:  # case: this is first entry; initialize the dataframe
            self.results = pd.DataFrame(result)
        else:
            self.results = self.results.append(result)
        return

    def to_csv(self):
        fp = "./metrics/{}.csv".format(self.uid)
        print("[INFO] writing results to {}".format(fp))
        print(self.results)
        self.results.to_csv(fp, index=False)
        return