import new_sim as sim
import par_gen_alg as alg
from multiprocessing import Pool

# this just makes sure the classes run without any exceptions

def test_cell_class(settings, initial):
    a = sim.Cell(settings)
    b = sim.Cell(settings)
    a.initialize(initial)
    b.inherit(a)
    a.run()
    b.run()
    print("Cell class okay")

def test_gen_alg_class(gen_size, s, settings, initial, pool):
    genalg = alg.Genetic(gen_size, s, settings, initial)
    genalg.parallel_gen_alg(2, pool)
    print("Genetic class okay")

def main():
    # cell settings
    settings = {'duration' : 5, 'beta' : 0.4, 'FPS' : 100, 'budget' : 100_000}
    # (K1, K2, K, m_dens, m_cov, dir_prob, sens)
    initial = {'k1' : 1000, 'k2' : 1000, 'K' : 10000, 'm_dens' : 0.01, 'm_cov' : 0.1, 'dir_prob' : 0.01, 'imp_sens' : 1}
    test_cell_class(settings, initial)
    with Pool(2) as pool:
        test_gen_alg_class(2, 1, settings, initial, pool)

if __name__ == '__main__':
    main()