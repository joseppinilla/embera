import embera
import networkx as nx
import matplotlib.pyplot as plt

palette = plt.get_cmap('Pastel2')

def plot_topologies(topologies, nrows=1, ncols=None, savefig=True):
    if ncols is None:
        ncols = len(topologies)//nrows + bool(len(topologies)%nrows)

    fig, axs = plt.subplots(nrows, ncols, subplot_kw={'aspect':'equal'})
    fig.set_size_inches(ncols , nrows)

    for i, (ax,G) in enumerate(zip(axs.flat,topologies)):
        if i>=len(topologies): fig.delaxes(ax); continue
        pos = G.graph.setdefault('pos', nx.spring_layout(G))
        draw_params = {"node_size":10, "width":0.2,
                       "edge_color":'grey', "node_color":palette(i//ncols)}
        nx.draw(G, pos=pos, ax=ax, **draw_params)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    if savefig:
        path = savefig if isinstance(savefig,str) else "./topologies.eps"
        plt.savefig(path)

def plot_embeddings(db, S, T, tags=[""], savefig=True):

    embeddings = [db.load_embedding(S,T,tag=tag) for tag in tags]

    nplots = len(embeddings)
    fig, axs = plt.subplots(1, nplots, subplot_kw={'aspect':'equal'}, squeeze=False)
    fig.set_size_inches(2*nplots, 2)
    for ax,embedding in zip(axs.flat,embeddings):
        embera.draw_architecture_embedding(T,embedding,node_size=0.2,ax=ax)
        method = embedding.properties["embedding_method"]
        runtime = embedding.properties["embedding_runtime"]
        ax.set_title(f"{method}\nruntime: {runtime:.2f}s\n{embedding.total_qubits} qubits")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=.85, wspace=0, hspace=0)

    if savefig:
        path = savefig if isinstance(savefig,str) else "./embeddings.eps"
        plt.savefig(path)
