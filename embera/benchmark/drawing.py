import embera
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

palette = plt.get_cmap('Pastel2')

def plot(plot_method, args, plot_kw={}, subplot_kw={}, savefig=True):
    nplots = len(args)
    fig, axs = plt.subplots(1, nplots, squeeze=False, subplot_kw=subplot_kw)

    for ax, arg in zip(axs.flat, args):
        plot_method(arg, ax=ax, **plot_kw)

    if savefig:
        path = savefig if isinstance(savefig,str) else f"./{plot_method.__name}.eps"
        plt.savefig(path)

def plot_parameters(bqms, savefig=True):
    nplots = len(bqms)

    fig, axs = plt.subplots(1, nplots, squeeze=False)
    fig.set_size_inches(2*nplots, 2)

    for ax,bqm in zip(axs.flat,bqms):
        type = bqm.info.get("type","")
        ax.set_title(type)
        h,J = bqm.linear.values(), bqm.quadratic.values()
        _ = ax.hist(J, range=(-2,2),bins=100,color=palette(0),label='J')
        _ = ax.hist(h, range=(-1,1),bins=100,color=palette(1),label='h')
        ax.legend()

    plt.subplots_adjust(left=0.05, right=0.995, bottom=0.1, top=0.9, wspace=0.2, hspace=0)

    if savefig:
        path = savefig if isinstance(savefig,str) else "./parameters.eps"
        plt.savefig(path)

def plot_topologies(topologies, nrows=1, ncols=None, savefig=True):
    if ncols is None:
        ncols = len(topologies)//nrows + bool(len(topologies)%nrows)

    fig, axs = plt.subplots(nrows, ncols, subplot_kw={'aspect':'equal'},squeeze=False)
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

def plot_embeddings(embeddings, T, savefig=True):

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

def plot_joint_sampleset(samplesets, savefig=True):

    nplots = len(samplesets)
    fig, axs = plt.subplots(1, nplots, squeeze=False)

    for ax, sampleset in zip(axs.flat, samplesets):
        x = []
        y = []
        E = []
        c = []
        for datum in sampleset.data():

            value = ''.join(str((1+datum.sample[k])//2) for k in sorted(datum.sample))
            size = len(value)
            x.append(int(value[0:size//2],2)/(2**(size//2)))
            y.append(int(value[size//2: ],2)/(2**(size//2)))
            c.append(datum.num_occurrences)
            E.append(datum.energy)


        grid = sns.JointGrid(x,y,space=0,xlim=(0,1),ylim=(0,1))

        grid = grid.plot_marginals(sns.distplot,kde=False,color='.5',bins=100)


        minE = sampleset.first.energy
        ratE = [250**((energy/minE)**5) for energy in E]
        grid = grid.plot_joint(plt.scatter, s=ratE, c=E, cmap="jet", alpha=0.5)

        xmin = grid.ax_marg_x.get_position().xmin
        xmax = grid.ax_marg_x.get_position().width
        cax = grid.fig.add_axes([xmin, 0, xmax, 0.02]) # [left, bottom, width, height]
        plt.colorbar(orientation='horizontal', cax=cax)
        cax.set_xlabel('Energy')


        for ax in [grid.ax_joint, grid.ax_marg_x, grid.ax_marg_y]:
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
        break # TEST


    if savefig:
        path = savefig if isinstance(savefig,str) else "./samplesets.eps"
        plt.savefig(path)
