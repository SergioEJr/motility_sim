import numpy as np

def calculate_pop_stats(pop):
    gen_size = len(list(pop))
    lists = np.zeros((9,gen_size))
    n_errors = 0
    for j, cell in enumerate(pop):
        if not cell.ERROR:
            dist = cell.distance
            pol = np.max((cell.K1/cell.K2, cell.K2/cell.K1))
            avg_k = cell.K2 + cell.K2
            motors = cell.motors
            n_mots = len(motors)
            avg_mot_E = np.mean([motor.ENERGY for motor in motors])
            avg_mot_cov = np.mean([motor.COV for motor in motors])
            avg_mot_dirp = np.mean([motor.DIR_PROB for motor in motors])
            avg_mot_sens = np.mean([motor.IMP_SENS for motor in motors])
            avg_mot_cent = np.mean([motor.IMP_CENTER for motor in motors])
            x = [dist, pol, avg_k, n_mots, avg_mot_E, avg_mot_cov, avg_mot_cov,
                avg_mot_dirp, avg_mot_sens, avg_mot_cent]
            for i in range(9):
                lists[i,j] = x[i]
        else:
            n_errors += 1
    lists[2] = lists[2]/2
    lists[3] = lists[3]/cell.LENGTH
    means = np.zeros(10)
    stds = np.zeros(10)
    for i in range(9):
        means[i] = np.mean(lists[i])
        stds[i] = np.std(lists[i])
    means[9] = n_errors/gen_size
    stds[9] = 0
    return means, stds

def calculate_pgen_stats(genalg):
    data = genalg.full_data
    properties = genalg.properties
    pgen_mean = {prop : [] for prop in properties}
    pgen_std = {prop : [] for prop in properties}
    for gen in data:
        means, stds = calculate_pop_stats(gen)
        for i, prop in enumerate(properties):
            pgen_mean[prop].append(means[i])
            pgen_std[prop].append(stds[i])
    return pgen_mean, pgen_std