""" Target Graph Architectures """

import dwave_networkx as dnx

__all__ = ["ARCHS"]

""" C4 """
C4_GEN = dnx.generators.chimera_graph
C4_DRAW = dnx.draw_chimera_embedding
C4_SPECS = [4,4,4]
""" D-Wave 2X """
DW2X_GEN = dnx.generators.chimera_graph
DW2X_DRAW = dnx.draw_chimera_embedding
DW2X_SPECS = [12,12,4]
""" D-Wave 2000Q """
DW2000Q_GEN = dnx.generators.chimera_graph
DW2000Q_DRAW = dnx.draw_chimera_embedding
DW2000Q_SPECS = [16,16,4]
""" D-Wave P6 """
P6_GEN = dnx.generators.pegasus_graph
P6_DRAW = dnx.draw_pegasus_embedding
P6_SPECS = [6]
""" D-Wave P16 """
P16_GEN = dnx.generators.pegasus_graph
P16_DRAW = dnx.draw_pegasus_embedding
P16_SPECS = [16]

ARCHS = {'C4':      ( C4_GEN, C4_DRAW, C4_SPECS ),
        'DW2X':     ( DW2X_GEN, DW2X_DRAW, DW2X_SPECS ),
        'DW2000Q':  ( DW2000Q_GEN, DW2000Q_DRAW, DW2000Q_SPECS ),
        'P6':       ( P6_GEN, P6_DRAW, P6_SPECS ),
        'P16':      ( P16_GEN, P16_DRAW, P16_SPECS )}
