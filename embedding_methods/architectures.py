""" Target Graph Architectures """

import dwave_networkx as dnx

__all__ = ["ARCHS"]

""" C4 """
C4_GEN = dnx.generators.chimera_graph
C4_DRAW = dnx.draw_chimera_embedding
C4_SPECS = [4,4,4]
C4_PROFILE = {'bipartite':32, 'complete':17, 'grid2d':49}
""" D-Wave 2X """
DW2X_GEN = dnx.generators.chimera_graph
DW2X_DRAW = dnx.draw_chimera_embedding
DW2X_SPECS = [12,12,4]
DW2X_PROFILE = {'bipartite':96, 'complete':49, 'grid2d':256}
""" D-Wave 2000Q """
DW2000Q_GEN = dnx.generators.chimera_graph
DW2000Q_DRAW = dnx.draw_chimera_embedding
DW2000Q_SPECS = [16,16,4]
DW2000Q_PROFILE = {'bipartite':128, 'complete':65, 'grid2d':484}
""" D-Wave P6 """
P6_GEN = dnx.generators.pegasus_graph
P6_DRAW = dnx.draw_pegasus_embedding
P6_SPECS = [6]
P6_PROFILE = {'bipartite':100, 'complete':59, 'grid2d':361}
""" D-Wave P16 """
P16_GEN = dnx.generators.pegasus_graph
P16_DRAW = dnx.draw_pegasus_embedding
P16_SPECS = [16]
P16_PROFILE = {'bipartite':266, 'complete':172, 'grid2d':1939}

ARCHS = {'C4':      ( C4_GEN, C4_DRAW, C4_SPECS, C4_PROFILE ),
        'DW2X':     ( DW2X_GEN, DW2X_DRAW, DW2X_SPECS, DW2X_PROFILE ),
        'DW2000Q':  ( DW2000Q_GEN, DW2000Q_DRAW, DW2000Q_SPECS, DW2000Q_PROFILE ),
        'P6':       ( P6_GEN, P6_DRAW, P6_SPECS, P6_PROFILE ),
        'P16':      ( P16_GEN, P16_DRAW, P16_SPECS, P16_PROFILE )}
