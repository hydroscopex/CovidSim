import numpy as np
import pandas as pd
import seaborn as sns
from skopt.sampler import *
import matplotlib.pyplot as plt
import scipy.stats as sci

from CovidTests import new_gene, aaz

tests = [aaz, new_gene]

seed = 1234


class CovidSim:
    def __init__(self):
        # Infection Simulation
        self.guest_list = None
        self.incidence = None
        self.prevalence = None
        self.bayesian_factor = None
        self.infected_odds = None
        self.infected_probas = None
        self.n_people = 0
        self.n_mc_infections = 0
        self.infection_sampler = None
        self.infection_bounds = None
        self.infection_realisations = None
        self.infections = None

        # Contamination Simulation
        self.n_infected = 0
        self.f = 0
        self.q = 0
        self.log_q_mean = 0
        self.log_q_std_dev = 0
        self.log_q = 0
        self.r = 0
        self.tilpf_mean = 0
        self.tilpf_sdev = 0
        self.tilpf = 0
        self.exposition = 0
        self.p_contamination = 0
        self.n_contaminated = 0

        self.contamination_computation_type = None
        self.n_mc_contaminations = 0
        self.contamination_sampler = 0
        self.contamination_bounds = 0
        self.contamination_realisations = None
        self.contamination_data = None
        self.contamination_data_analysis = None
        self.full_contamination_data = None
        self.full_contamination_data_analysis = None

        # Outcome Simulation

    # Infections

    def load_guest(self, filename):
        self.guest_list = pd.read_excel(filename)
        self.incidence = self.guest_list['Incidence [-]'].values
        self.prevalence = self.incidence
        self.bayesian_factor = np.ones_like(self.incidence)
        for test in tests:
            if test.name in self.guest_list.columns:
                # https://covid-19-diagnostics.jrc.ec.europa.eu/devices/detail/1833
                for i, value in enumerate(self.guest_list[test.name]):
                    if value > 0.5:
                        self.bayesian_factor[i] *= value * test.get_positive_bf()
                    if value < 0.5:
                        self.bayesian_factor[i] *= value * test.get_negative_bf()

        if 'Contact Case' in self.guest_list.columns:
            # https://doi.org/10.1001/jamainternmed.2021.4686
            for i, value in enumerate(self.guest_list['New Gene']):
                if value > 0.5:
                    self.bayesian_factor[i] *= 1.3

        self.infected_odds = self.prevalence * self.bayesian_factor / (1 - self.prevalence)
        self.infected_probas = self.infected_odds / (1 + self.infected_odds)
        self.n_people = len(self.incidence)

    def compute_infections(self, n_mc_points):
        self.n_mc_infections = n_mc_points
        self.infection_sampler = Lhs()
        self.infection_bounds = np.zeros((2, len(self.incidence)))
        self.infection_bounds[0] += 0
        self.infection_bounds[1] += 1
        self.infection_realisations = np.array(self.infection_sampler.generate(self.infection_bounds.T, self.n_mc_infections, random_state=seed))
        self.infections = np.sum((self.infection_realisations < self.infected_probas), axis=1)

    def plot_infections(self, figsize=(12, 12)):
        plt.figure(figsize=figsize)
        ax = sns.histplot(self.infections, bins=np.arange(0, np.amax(self.infections)+2), stat='density')
        ax.set(xlabel="Nombre d'infectés")
        plt.show()

    # Contaminations

    def compute_f(self, c, co=410e-6, ca=0.038, c_sdev=30e-6, quantile=0.5):
        # Hypotheses of normaly distributed CO2 concentrations
        self.co2 = sci.norm(loc=c, scale=c_sdev).ppf(quantile)
        self.f = (self.co2 - co) / ca

    def compute_quanta_generation_rate(self, type='default', quantile=0.5, variant='alpha'):
        if variant == 'alpha':
            return self.compute_quanta_generation_rate_alpha(type=type, quantile=quantile)

    def compute_quanta_generation_rate_alpha(self, type='default', quantile=0.5):
        # from 10.1016/j.envint.2020.106112
        if type == 'default':
            self.q = 142 / 3600
        else:
            if type == 'resting/breathing':
                self.log_q_mean = -0.429
                self.log_q_std_dev = 0.72
            elif type == 'activity/breathing':
                self.log_q_mean = 0.399
                self.log_q_std_dev = 0.72
            elif type == 'activity/speaking':
                self.log_q_mean = 0.698
                self.log_q_std_dev = 0.72
            elif type == 'activity/singing':
                self.log_q_mean = 1.5
                self.log_q_std_dev = 0.72
            self.log_q = sci.norm(loc=self.log_q_mean, scale=self.log_q_std_dev).ppf(quantile)
            self.q = 10 ** self.log_q / 3600

    def compute_respirator(self, type='none', quantile=0.5):
        # from 10.1371/journal.pone.0258191
        if type == 'none':
            self.r = 1
        else:
            if type == '2 layer':
                self.tilpf_mean = 0.71511
                self.tilpf_sdev = 0.12129
            elif type == 'multi layer':
                self.tilpf_mean = 0.61419
                self.tilpf_sdev = 0.22357
            elif type == 'surgical':
                self.tilpf_mean = 0.49108
                self.tilpf_sdev = 0.19704
            elif type == 'N95':
                self.tilpf_mean = 0.00651
                self.tilpf_sdev = 0.00261
            self.tilpf = sci.norm(loc=self.tilpf_mean, scale=self.tilpf_sdev).ppf(quantile)
            if self.tilpf < 0:
                self.tilpf = 0
            self.r = self.tilpf  # todo : check validity of this relation

    def compute_exposition(self, exposition, exposition_sdev=600, quantile=0.5):
        self.exposition = sci.norm(loc=exposition, scale=exposition_sdev).ppf(quantile)

    def compute_p_con(self):
        self.p_contamination = 1 - np.exp(-self.f * self.n_infected * self.q * self.exposition * self.r / self.n_people)
        self.n_contaminated = (self.n_people - self.n_infected) * self.p_contamination

    def compute_contaminations(self, n_mc_points, co2_levels=np.array([600, 800, 1000, 1200]) * 1e-6, activity_levels=['activity/speaking', 'activity/singing'], masks=['none', '2 layer', 'surgical', 'N95'], exposition_levels=np.array([2, 4, 6]) * 3600, type='single', full_data=True):
        self.contamination_computation_type = type
        self.n_mc_contaminations = n_mc_points
        if self.contamination_computation_type == 'single':
            self.contamination_sampler = Lhs()
            self.contamination_bounds = np.zeros((2, 4))
            self.contamination_bounds[0] += 0.111
            self.contamination_bounds[1] += 0.999  # to avoid getting weird samples from infinite support distributions
            self.contamination_realisations = np.array(self.contamination_sampler.generate(self.contamination_bounds.T, self.n_mc_contaminations, random_state=seed))

            self.contamination_data = pd.DataFrame()

            counter = 0
            for n_infected in range(1, np.amax(self.infections) + 1):
                self.n_infected = n_infected
                print('N infected [-] = ', self.n_infected)
                for co2_level in co2_levels:
                    print('CO2 level [ppm e-6]= ', co2_level)
                    for activity_level in activity_levels:
                        print('Activity level = ', activity_level)
                        for mask in masks:
                            print('Mask = ', mask)
                            for exposition_level in exposition_levels:
                                print('Exposition level [s]= ', exposition_level)
                                for realisation in self.contamination_realisations:
                                    self.compute_f(co2_level, quantile=realisation[0])
                                    self.compute_quanta_generation_rate(type=activity_level, quantile=realisation[1])
                                    self.compute_respirator(type=mask, quantile=realisation[2])
                                    self.compute_exposition(exposition_level, quantile=realisation[3])
                                    self.compute_p_con()
                                    # n_infected, co2_level, co2, f, activity_level, q, mask, r, exposition_level, exposition, P-contamination, n_contamination
                                    self.contamination_data = self.contamination_data.append({'Number of infected [-]': n_infected,
                                                                                              'CO2 level [ppm e-6]': co2_level,
                                                                                              'Mean CO2 [ppm e-6]': self.co2,
                                                                                              'Fraction of exhaled breath [-]': self.f,
                                                                                              'Activity level': activity_level,
                                                                                              'Quanta emission rate [quanta/s]': self.q,
                                                                                              'Respirator type': mask,
                                                                                              'Fraction of particle penetration [-]': self.r,
                                                                                              'Exposition level [s]': exposition_level,
                                                                                              'Exposition [s]': self.exposition,
                                                                                              'Individual probability of Contamination [-]': self.p_contamination,
                                                                                              'Number of contaminations [-]': self.n_contaminated},
                                                                                             ignore_index=True)
                                    counter += 1

            if full_data:
                no_contamination_data = self.contamination_data[self.contamination_data['Number of infected [-]'] == 1]
                no_contamination_data['Number of infected [-]'] = 0
                no_contamination_data['Individual probability of Contamination [-]'] = 0
                no_contamination_data['Number of contaminations [-]'] = 0
                self.full_contamination_data = pd.DataFrame()
                for infection in self.infections:
                    if infection == 0:
                        self.full_contamination_data = self.full_contamination_data.append(no_contamination_data)
                    else:
                        self.full_contamination_data = self.full_contamination_data.append(self.contamination_data[self.contamination_data['Number of infected [-]'] == infection])

    def plot_contaminations(self, bins=200, figsize=(12, 12)):
        plt.figure(figsize=figsize)
        ax = sns.histplot(self.contamination_data['Number of contaminations [-]'], x='Number of contaminations [-]', kde=True, bins=bins)
        plt.show()

    def plot_contaminations_marginal(self, parameter, quantile=0.95, bins=200, figsize=(12, 12)):
        plt.figure(figsize=figsize)
        ax = sns.histplot(self.contamination_data[['Number of contaminations [-]', parameter]], x='Number of contaminations [-]', hue=parameter, multiple='dodge', kde=True, bins=bins)
        print(self.contamination_data[['Number of contaminations [-]', parameter]].groupby(parameter).quantile(quantile))
        plt.show()

    def analyse_contamination_data(self, quantile=0.95):
        self.contamination_data_analysis = self.contamination_data[['Number of infected [-]', 'CO2 level [ppm e-6]', 'Activity level', 'Respirator type', 'Exposition level [s]', 'Number of contaminations [-]']].groupby(['Number of infected [-]', 'CO2 level [ppm e-6]', 'Activity level', 'Respirator type', 'Exposition level [s]'], as_index=False).quantile(quantile)

    def analyse_full_contamination_data(self, quantile=0.95):
        self.full_contamination_data_analysis = self.full_contamination_data[['CO2 level [ppm e-6]', 'Activity level', 'Respirator type', 'Exposition level [s]', 'Number of contaminations [-]']].groupby(['CO2 level [ppm e-6]', 'Activity level', 'Respirator type', 'Exposition level [s]'], as_index=False).quantile(quantile)

    # Outcomes

    def p_hosp_c_contaminated(self, age, sex, quantile=0.5, variant='alpha'):
        if variant == 'alpha':
            return self.p_hosp_c_contaminated(age, sex, quantile=quantile)

    def p_hosp_c_contaminated_alpha(self, age, sex, quantile=0.5):
        # from https://hal-pasteur.archives-ouvertes.fr/pasteur-02548181
        if sex == 'M':
            if age < 20:
                mean = 0.001
                lci = 0.0006
                uci = 0.002
            if 20 <= age < 30:
                mean = 0.006
                lci = 0.003
                uci = 0.01
            if 30 <= age < 40:
                mean = 0.012
                lci = 0.006
                uci = 0.02
            if 40 <= age < 50:
                mean = 0.016
                lci = 0.009
                uci = 0.027
            if 50 <= age < 60:
                mean = 0.032
                lci = 0.017
                uci = 0.053
            if 60 <= age < 70:
                mean = 0.07
                lci = 0.037
                uci = 0.117
            if 70 <= age < 80:
                mean = 0.114
                lci = 0.061
                uci = 0.19
            if 80 <= age:
                mean = 0.314
                lci = 0.167
                uci = 0.526
        if sex == 'F':
            if age < 20:
                mean = 0.0009
                lci = 0.0005
                uci = 0.002
            if 20 <= age < 30:
                mean = 0.005
                lci = 0.003
                uci = 0.008
            if 30 <= age < 40:
                mean = 0.009
                lci = 0.005
                uci = 0.015
            if 40 <= age < 50:
                mean = 0.013
                lci = 0.007
                uci = 0.022
            if 50 <= age < 60:
                mean = 0.025
                lci = 0.014
                uci = 0.042
            if 60 <= age < 70:
                mean = 0.053
                lci = 0.028
                uci = 0.088
            if 70 <= age < 80:
                mean = 0.08
                lci = 0.043
                uci = 0.134
            if 80 <= age:
                mean = 0.159
                lci = 0.085
                uci = 0.265

        return mean

    def p_icu_c_hosp(self, age, sex, quantile=0.5, variant='alpha'):
        if variant == 'alpha':
            return self.p_icu_c_hosp_alpha(age, sex, quantile=quantile)

    def p_icu_c_hosp_alpha(self, age, sex, quantile=0.5):
        # from https://hal-pasteur.archives-ouvertes.fr/pasteur-02548181
        if sex == 'M':
            if age < 20:
                mean = 0.175
                lci = 0.138
                uci = 0.22
            if 20 <= age < 30:
                mean = 0.122
                lci = 0.10
                uci = 0.148
            if 30 <= age < 40:
                mean = 0.172
                lci = 0.152
                uci = 0.193
            if 40 <= age < 50:
                mean = 0.243
                lci = 0.225
                uci = 0.263
            if 50 <= age < 60:
                mean = 0.317
                lci = 0.30
                uci = 0.334
            if 60 <= age < 70:
                mean = 0.364
                lci = 0.348
                uci = 0.381
            if 70 <= age < 80:
                mean = 0.29
                lci = 0.277
                uci = 0.303
            if 80 <= age:
                mean = 0.057
                lci = 0.052
                uci = 0.061
        if sex == 'F':
            if age < 20:
                mean = 0.085
                lci = 0.058
                uci = 0.121
            if 20 <= age < 30:
                mean = 0.068
                lci = 0.051
                uci = 0.089
            if 30 <= age < 40:
                mean = 0.104
                lci = 0.088
                uci = 0.122
            if 40 <= age < 50:
                mean = 0.143
                lci = 0.128
                uci = 0.159
            if 50 <= age < 60:
                mean = 0.19
                lci = 0.177
                uci = 0.204
            if 60 <= age < 70:
                mean = 0.216
                lci = 0.203
                uci = 0.229
            if 70 <= age < 80:
                mean = 0.17
                lci = 0.16
                uci = 0.181
            if 80 <= age:
                mean = 0.034
                lci = 0.03
                uci = 0.038
        return mean

    def p_death_c_hosp(self, age, sex, quantile=0.5, variant='alpha'):
        if variant == 'alpha':
            return self.p_death_c_hosp_alpha(age, sex, quantile=quantile)

    def p_death_c_hosp_alpha(self, age, sex, quantile=0.5):
        # from https://hal-pasteur.archives-ouvertes.fr/pasteur-02548181
        if sex == 'M':
            if age < 20:
                mean = 0.012
                lci = 0.004
                uci = 0.028
            if 20 <= age < 30:
                mean = 0.013
                lci = 0.006
                uci = 0.024
            if 30 <= age < 40:
                mean = 0.025
                lci = 0.018
                uci = 0.034
            if 40 <= age < 50:
                mean = 0.039
                lci = 0.031
                uci = 0.047
            if 50 <= age < 60:
                mean = 0.075
                lci = 0.066
                uci = 0.083
            if 60 <= age < 70:
                mean = 0.142
                lci = 0.132
                uci = 0.153
            if 70 <= age < 80:
                mean = 0.253
                lci = 0.241
                uci = 0.266
            if 80 <= age:
                mean = 0.42
                lci = 0.407
                uci = 0.434
        if sex == 'F':
            if age < 20:
                mean = 0.001
                lci = 0.00005
                uci = 0.0015
            if 20 <= age < 30:
                mean = 0.014
                lci = 0.006
                uci = 0.027
            if 30 <= age < 40:
                mean = 0.016
                lci = 0.009
                uci = 0.014
            if 40 <= age < 50:
                mean = 0.032
                lci = 0.025
                uci = 0.041
            if 50 <= age < 60:
                mean = 0.064
                lci = 0.056
                uci = 0.072
            if 60 <= age < 70:
                mean = 0.12
                lci = 0.11
                uci = 0.131
            if 70 <= age < 80:
                mean = 0.207
                lci = 0.195
                uci = 0.22
            if 80 <= age:
                mean = 0.34
                lci = 0.327
                uci = 0.354
        return mean

    def compute_outcome(self, age, sex, variant='alpha', organ_transplant=False, obesity=False, quantiles=np.zeros(6) + 0.5):
        p_hosp = self.p_hosp_c_contaminated(age, sex, quantile=quantiles[0], variant=variant)
        odds_hosp = p_hosp / (1 - p_hosp)
        if organ_transplant:
            # https://doi.org/10.1186/s12916-021-02058-6
            odds_hosp *= 3.4  # ci (1.7 - 6.6)
        if obesity:
            odds_hosp *= 1.59  # ci (1.52 - 1.66)
        updated_p_hosp = odds_hosp / (1 + odds_hosp)

        p_icu = updated_p_hosp * self.p_icu_c_hosp(age, sex, quantile=quantiles[1], variant=variant)
        p_death = updated_p_hosp * self.p_death_c_hosp(age, sex, quantile=quantiles[2], variant=variant)
        hosp = False
        icu = False
        death = False
        if updated_p_hosp > quantiles[3]:
            hosp = True
            if p_icu > quantiles[4]:
                icu = True
            if p_death > quantiles[5]:
                death = True
        return hosp, icu, death

'''n_mc_points = 100

# 1 - Estimate apriori number of infected throught MC

guest_list = pd.read_excel('./apriori_incidence_P0.xlsx')
incidence = guest_list['Incidence [-]'].values

n_people = len(incidence)

sampler = Lhs()
bounds = np.zeros((2, len(incidence)))
bounds[0] += 0
bounds[1] += 1 # to avoid getting weird samples from infinite support distributions
realisations = np.array(sampler.generate(bounds.T, n_mc_points, random_state=seed))

infections = np.sum((realisations < incidence), axis=1)

#ax = sns.histplot(infections, bins=np.arange(0, np.amax(infections)+2), stat='density')
#ax.set(xlabel="Nombre d'infectés")
#plt.show()

# 2 - estimate quanta Production Rate

"""# taken from equation 3 in 10.1016/j.envint.2020.105794

volume = 40
co2 =

er = 142  # quanta / h
aer =
k = 0.24  # h-1 deposition rate due to gravity
l = 0.63  # h-1 virus inactivation
ivvr = aer + k + l

# steady state assumption"""

# based on https://doi.org/10.1034/j.1600-0668.2003.00189.x

c = 1200e-6 # average
co = 410e-6
ca = 0.038

f = (c - co) / ca

q = 142 / 3600  # quanta / s from 10.1016/j.envint.2020.105794

exposition = 2*3600  # s

P_inf = 1 - np.exp(-f * infections.mean() * q * exposition / n_people)

print(P_inf)'''