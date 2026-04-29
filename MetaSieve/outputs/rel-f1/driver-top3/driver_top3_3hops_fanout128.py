from torch_geometric.sampler.base import NumNeighbors

num_neighbors = NumNeighbors({
    ('circuits', 'rev_f2p_circuitId', 'races'): [0, 0, 128],
    ('constructor_standings', 'f2p_constructorId', 'constructors'): [0, 0, 128],
    ('constructor_standings', 'f2p_raceId', 'races'): [0, 0, 128],
    ('constructors', 'rev_f2p_constructorId', 'qualifying'): [0, 128, 0],
    ('constructors', 'rev_f2p_constructorId', 'results'): [0, 128, 0],
    ('drivers', 'rev_f2p_driverId', 'results'): [0, 128, 0],
    ('qualifying', 'f2p_constructorId', 'constructors'): [0, 0, 128],
    ('qualifying', 'f2p_driverId', 'drivers'): [128, 0, 128],
    ('qualifying', 'f2p_raceId', 'races'): [0, 0, 128],
    ('races', 'rev_f2p_raceId', 'qualifying'): [0, 128, 0],
    ('races', 'rev_f2p_raceId', 'results'): [0, 128, 0],
    ('races', 'rev_f2p_raceId', 'standings'): [0, 128, 0],
    ('results', 'f2p_constructorId', 'constructors'): [0, 0, 128],
    ('results', 'f2p_driverId', 'drivers'): [128, 0, 128],
    ('results', 'f2p_raceId', 'races'): [0, 0, 128],
    ('standings', 'f2p_driverId', 'drivers'): [128, 0, 128],
    ('standings', 'f2p_raceId', 'races'): [0, 0, 128],
}, default=[0]*3)
