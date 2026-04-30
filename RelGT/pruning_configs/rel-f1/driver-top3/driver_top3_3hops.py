from torch_geometric.sampler.base import NumNeighbors

num_neighbors = NumNeighbors({
    ('circuits', 'rev_f2p_circuitId', 'races'): [0, 0, 64],
    ('constructor_standings', 'f2p_constructorId', 'constructors'): [0, 0, 64],
    ('constructor_standings', 'f2p_raceId', 'races'): [0, 0, 64],
    ('constructors', 'rev_f2p_constructorId', 'qualifying'): [0, 64, 0],
    ('constructors', 'rev_f2p_constructorId', 'results'): [0, 64, 0],
    ('drivers', 'rev_f2p_driverId', 'results'): [0, 64, 0],
    ('qualifying', 'f2p_constructorId', 'constructors'): [0, 0, 64],
    ('qualifying', 'f2p_driverId', 'drivers'): [64, 0, 64],
    ('qualifying', 'f2p_raceId', 'races'): [0, 0, 64],
    ('races', 'rev_f2p_raceId', 'qualifying'): [0, 64, 0],
    ('races', 'rev_f2p_raceId', 'results'): [0, 64, 0],
    ('races', 'rev_f2p_raceId', 'standings'): [0, 64, 0],
    ('results', 'f2p_constructorId', 'constructors'): [0, 0, 64],
    ('results', 'f2p_driverId', 'drivers'): [64, 0, 64],
    ('results', 'f2p_raceId', 'races'): [0, 0, 64],
    ('standings', 'f2p_driverId', 'drivers'): [64, 0, 64],
    ('standings', 'f2p_raceId', 'races'): [0, 0, 64],
}, default=[0]*3)
