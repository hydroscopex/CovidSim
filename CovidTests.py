class CovidTest:
    def __init__(self, sensitivity, specificity, fpr, fnr, name='GenericCovidTest'):
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.fpr = fpr
        self.fnr = fnr
        self.name = name

        self.positive_bf = self.sensitivity / self.fpr
        self.negative_bf = self.fnr / self.specificity

    def positive_proba(self, prevalence):
        odds = prevalence / (1 - prevalence)
        updated_odds = odds * self.positive_bf
        return updated_odds / (1 + updated_odds)

    def negative_proba(self, prevalence):
        odds = prevalence / (1 - prevalence)
        updated_odds = odds * self.negative_bf
        return updated_odds / (1 + updated_odds)

    def get_positive_bf(self, quantile=0.5):
        return self.positive_bf

    def get_negative_bf(self, quantile=0.5):
        return self.negative_bf


new_gene = CovidTest(0.98, 0.991, 0.0087, 0.0202, 'New Gene')
aaz = CovidTest(0.966, 100, 1e-9, 0.034, 'AAZ')

#prevalence = 0.00633

#print("probabilité d'avoir le covid suit à un test New Gene +", new_gene.positive_proba(prevalence) * 100, "%")
#print("probabilité d'avoir le covid suit à un test New Gene -", new_gene.negative_proba(prevalence) * 100, "%")

#print("probabilité d'avoir le covid suit à un test AAZ +", aaz.positive_proba(prevalence) * 100, "%")
#print("probabilité d'avoir le covid suit à un test AAZ -", aaz.negative_proba(prevalence) * 100, "%")
