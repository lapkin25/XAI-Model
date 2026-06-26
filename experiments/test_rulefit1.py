import numpy as np
import pandas as pd
import pysubgroup as ps

import sys
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data

data = Data("DataSet.xlsx")

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]

data.prepare(predictors, "Dead", [], scale_data=False)

data_np = np.column_stack([data.x, data.y])

#print(data_np)

columns = predictors + ['Dead']
dataset = pd.DataFrame(data_np, columns=columns)


import numpy as np
import pandas as pd
from numpy import median
from rulefit import RuleFit
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor


def get_rules(rf, exclude_zero_coef=False, subregion=None):
        """Return the estimated rules
        Parameters
        ----------
        exclude_zero_coef: If True (default), returns only the rules with an estimated
                           coefficient not equalt to  zero.
        subregion: If None (default) returns global importances (FP 2004 eq. 28/29), else returns importance over
                           subregion of inputs (FP 2004 eq. 30/31/32).
        Returns
        -------
        rules: pandas.DataFrame with the rules. Column 'rule' describes the rule, 'coef' holds
               the coefficients and 'support' the support of the rule in the training
               data set (X)
        """
        n_features= len(rf.coef_) - len(rf.rule_ensemble.rules)
        rule_ensemble = list(rf.rule_ensemble.rules)
        output_rules = []
        ## Add coefficients for linear effects
        for i in range(0, n_features):
            if rf.lin_standardise:
                coef=rf.coef_[i]*rf.friedscale.scale_multipliers[i]
            else:
                coef=rf.coef_[i]
            if subregion is None:
                importance = abs(coef)*rf.stddev[i]
            else:
                subregion = np.array(subregion)
                importance = sum(abs(coef)* abs([ x[i] for x in rf.winsorizer.trim(subregion) ] - rf.mean[i]))/len(subregion)
            output_rules += [(rf.feature_names[i], 'linear',coef, 1, importance)]
        ## Add rules
        for i in range(0, len(rf.rule_ensemble.rules)):
            rule = rule_ensemble[i]
            coef=rf.coef_[i + n_features]
            if subregion is None:
                importance = abs(coef)*(rule.support * (1-rule.support))**(1/2)
            else:
                rkx = rule.transform(subregion)
                importance = sum(abs(coef) * abs(rkx - rule.support))/len(subregion)

            output_rules += [(rule.__str__(), 'rule', coef,  rule.support, importance)]
        rules = pd.DataFrame(output_rules, columns=["rule", "type","coef", "support", "importance"])
        if exclude_zero_coef:
            rules = rules.ix[rules.coef != 0]
        return rules

#dataset = pd.read_excel("Dataset.xlsx")
x = predictors
y = 'Dead'


gb = GradientBoostingRegressor(n_estimators=500, max_depth=10, learning_rate=0.01)
rf = RuleFit( tree_generator=gb,max_rules=30) #,  max_iter = 100)

rf.fit(dataset[x].values, dataset[y].values, feature_names=x)
proba = rf.predict( dataset[x].values )
auc_score = roc_auc_score( dataset[y].values, proba )
print(auc_score)
pd.set_option('display.max_columns', None)
rules = get_rules( rf )
print(rules)




