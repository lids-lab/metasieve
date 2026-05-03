from torch_geometric.sampler.base import NumNeighbors

num_neighbors = NumNeighbors({
    ('constructor_results', 'f2p_constructorId', 'constructors'): [0, 0, 64, 0],
    ('constructor_standings', 'f2p_constructorId', 'constructors'): [0, 0, 64, 0],
    ('constructor_standings', 'f2p_raceId', 'races'): [0, 0, 64, 0],
    ('constructors', 'rev_f2p_constructorId', 'constructor_results'): [0, 0, 0, 64],
    ('constructors', 'rev_f2p_constructorId', 'constructor_standings'): [0, 0, 0, 64],
    ('constructors', 'rev_f2p_constructorId', 'qualifying'): [0, 64, 0, 64],
    ('constructors', 'rev_f2p_constructorId', 'results'): [0, 0, 0, 64],
    ('drivers', 'rev_f2p_driverId', 'qualifying'): [0, 64, 0, 0],
    ('drivers', 'rev_f2p_driverId', 'results'): [0, 64, 0, 64],
    ('drivers', 'rev_f2p_driverId', 'standings'): [0, 64, 0, 64],
    ('qualifying', 'f2p_driverId', 'drivers'): [64, 0, 64, 0],
    ('races', 'rev_f2p_raceId', 'constructor_results'): [0, 0, 0, 64],
    ('races', 'rev_f2p_raceId', 'constructor_standings'): [0, 0, 0, 64],
    ('races', 'rev_f2p_raceId', 'qualifying'): [0, 64, 0, 0],
    ('races', 'rev_f2p_raceId', 'standings'): [0, 64, 0, 0],
    ('results', 'f2p_constructorId', 'constructors'): [0, 0, 64, 0],
    ('results', 'f2p_driverId', 'drivers'): [64, 0, 0, 0],
    ('results', 'f2p_raceId', 'races'): [0, 0, 64, 0],
    ('standings', 'f2p_driverId', 'drivers'): [64, 0, 64, 0],
}, default=[0]*4)
