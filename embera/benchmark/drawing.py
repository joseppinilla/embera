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

def plot_joint_samplesets(samplesets, savefig=True):
    """
    Based on:
    https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html
    """
    for i,sampleset in enumerate(samplesets):
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

        minE = sampleset.first.energy
        ratE = [250**((energy/minE)**5) for energy in E]

        fig = plt.figure(i, figsize=(6, 6))
        grid = plt.GridSpec(5, 5, hspace=0.0, wspace=0.0)

        # Set up the axes with gridspec
        main_ax = fig.add_subplot(grid[1:5,0:4])
        y_hist = fig.add_subplot(grid[1:5,4], sharey=main_ax)
        x_hist = fig.add_subplot(grid[0,0:4], sharex=main_ax)

        y_hist.spines['right'].set_visible(False)
        y_hist.spines['top'].set_visible(False)
        y_hist.xaxis.set_ticks_position('bottom')
        y_hist.set_yticks([],[])

        x_hist.spines['left'].set_visible(False)
        x_hist.spines['top'].set_visible(False)
        x_hist.yaxis.set_ticks_position('right')
        x_hist.set_xticks([],[])

        # scatter points on the main axes
        sct = main_ax.scatter(x, y, s=ratE, c=E, cmap="jet", alpha=0.5)

        # histogram on the attached axes
        x_hist.hist(x, 100, histtype='stepfilled',
                    orientation='vertical', color='gray')

        y_hist.hist(y, 100, histtype='stepfilled',
                    orientation='horizontal', color='gray')

        xmin = main_ax.get_position().xmin
        xmax = main_ax.get_position().width
        cax = fig.add_axes([xmin, 0, xmax, 0.02]) # [left, bottom, width, height]
        # cax = fig.add_axes([0.09, 0.06, 0.84, 0.02])
        plt.colorbar(sct,orientation='horizontal',cax=cax)
        cax.set_xlabel('Energy')
        break
