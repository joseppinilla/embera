import embera
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors

from mpl_toolkits.mplot3d import Axes3D
palette = plt.get_cmap('Pastel2')

def plot(plot_method, args, plot_kw={}, subplot_kw={}, savefig=True):
    nplots = len(args)
    fig, axs = plt.subplots(1, nplots, squeeze=False, subplot_kw=subplot_kw)

    for ax, arg in zip(axs.flat, args):
        plot_method(arg, ax=ax, **plot_kw)

    if savefig:
        path = savefig if isinstance(savefig,str) else f"./{plot_method.__name}.pdf"
        plt.savefig(path)

def plot_yields(solvers, savefig=True):
    kwargs = {'plot_kw':{'node_size':1},
              'subplot_kw':{'aspect':'equal'},
              'savefig':'yield.pdf'}
    plot(embera.draw_architecture_yield,solvers,**kwargs)

def plot_parameters(bqms, savefig=True):
    nplots = len(bqms)

    fig, axs = plt.subplots(1, nplots, squeeze=False)
    fig.set_size_inches(2*nplots, 2)

    for ax,bqm in zip(axs.flat,bqms):
        tags = bqm.info.get("tags","")
        ax.set_title(" ".join(tags))
        h,J = bqm.linear.values(), bqm.quadratic.values()
        _ = ax.hist(J, range=(-2,2),bins=100,color=palette(0),label='J')
        _ = ax.hist(h, range=(-1,1),bins=100,color=palette(1),label='h')
        ax.legend()

    plt.subplots_adjust(left=0.05, right=0.995, bottom=0.1, top=0.9, wspace=0.2, hspace=0)

    if savefig:
        path = savefig if isinstance(savefig,str) else "./parameters.pdf"
        plt.savefig(path)

def plot_topologies(topologies, nrows=1, ncols=None, spring_seed=None, savefig=True):
    if ncols is None:
        ncols = len(topologies)//nrows + bool(len(topologies)%nrows)

    fig, axs = plt.subplots(nrows,ncols,subplot_kw={'aspect':'equal'},squeeze=False)
    fig.set_size_inches(ncols , nrows)

    for i, (ax,G) in enumerate(zip(axs.flat,topologies)):
        if i>=len(topologies): fig.delaxes(ax); continue
        pos = G.graph.setdefault('pos', nx.spring_layout(G,seed=spring_seed,weight=None))
        draw_params = {"node_size":10, "width":0.2,
                       "edge_color":'grey',
                       "node_color":matplotlib.colors.to_hex(palette(i//ncols))}
        nx.draw(G, pos=pos, ax=ax, **draw_params)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    if savefig:
        path = savefig if isinstance(savefig,str) else "./topologies.pdf"
        plt.savefig(path)

def plot_embeddings(embeddings, T, classes=[], savefig=True):
    nplots = len(embeddings)
    fig, axs = plt.subplots(1, nplots, subplot_kw={'aspect':'equal'}, squeeze=False)
    fig.set_size_inches(2*nplots, 2)
    for ax,embedding in zip(axs.flat,embeddings):
        embera.draw_architecture_embedding(T,embedding,node_size=0.2,ax=ax)
        if not embedding: ax.set_title(f"N/A\nruntime: N/A\nN/A qubits"); continue
        method = embedding.properties["embedding_method"]
        runtime = embedding.properties["embedding_runtime"]
        ax.set_title(f"{method}\nruntime: {runtime:.2f}s\n{embedding.total_qubits} qubits")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=.85, wspace=0, hspace=0)

    if savefig:
        path = savefig if isinstance(savefig,str) else "./embeddings.pdf"
        plt.savefig(path)

def plot_joint_samplesets(samplesets, info_key=None, gray=True, savefig=True):
    nplots = len(samplesets)
    fig = plt.figure(figsize=(nplots*3, 3))
    grid = plt.GridSpec(5, 5*nplots, hspace=0.0, wspace=0.0)

    def gray2bin(n):
        n = int(n, 2)
        mask = n
        while mask != 0:
            mask >>= 1
            n ^= mask
        return bin(n)[2:]

    minE = float('Inf')
    maxE = -float('Inf')
    x = {}; y = {}; E = {}; c = {}
    for i,sampleset in enumerate(samplesets):

        x[i] = []; y[i] = []; E[i] = []; c[i] = []
        if not sampleset: continue

        minE = sampleset.first.energy
        size = len(sampleset.first.sample)
        xmax = (2**(size//2))
        ymax = (2**(size-size//2))

        # Reverse iteration allows plotting lower (important) samples on top.
        for datum in sampleset.data(sorted_by='energy',reverse=True):
            value = ''.join(str((1+datum.sample[k])//2) for k in sorted(datum.sample))
            x_point = gray2bin(value[0:size//2]) if gray else value[0:size//2]
            y_point = gray2bin(value[size//2:]) if gray else  value[size//2:]
            x[i].append(int(x_point,2)/xmax)
            y[i].append(int(y_point,2)/ymax)
            c[i].append(datum.num_occurrences)
            E[i].append(datum.energy)
            if datum.energy < minE: minE = datum.energy
            if datum.energy > maxE: maxE = datum.energy


    ims = []
    xlim=ylim=(0.0,1.0)
    rangeE = maxE - minE
    for i,sampleset in enumerate(samplesets):
        # Set up the axes with gridspec
        main_ax = fig.add_subplot(grid[1:5,i*5:4+(i*5)],xlim=xlim,ylim=ylim)
        y_hist = fig.add_subplot(grid[1:5,4+(i*5)],sharey=main_ax,frameon=False)
        x_hist = fig.add_subplot(grid[0,i*5:4+(i*5)],sharex=main_ax,frameon=False)


        # No ticks of histograms
        y_hist.set_xticks([],[])
        y_hist.set_yticks([],[])
        x_hist.set_xticks([],[])
        x_hist.set_yticks([],[])

        if not sampleset: main_ax.set_xlabel('N/A'); continue

        # Scatter points on the main axes
        ratE = [250*(((energy-minE)/rangeE)**2) for energy in E[i]]
        sct = main_ax.scatter(x[i],y[i],s=ratE,c=E[i],cmap="jet",alpha=0.5)


        # Histograms on the attached axes
        x_hist.hist(x[i], 100, histtype='stepfilled',
                    orientation='vertical', color='gray')

        y_hist.hist(y[i], 100, histtype='stepfilled',
                    orientation='horizontal', color='gray')

        ims.append(sct)
        main_ax.set_xlabel(sampleset.info[info_key])

    # Color Bar
    vmin,vmax = zip(*[im.get_clim() for im in ims])

    for i,im in enumerate(ims):
        im.set_clim(vmin=min(vmin),vmax=max(vmax))

    cax = fig.add_axes([0.25,-0.01,0.5,0.02]) # [left,bottom,width,height]
    plt.colorbar(sct,orientation='horizontal',cax=cax)
    _ = cax.set_xlabel('Energy')

    if savefig:
        path = savefig if isinstance(savefig,str) else "./samplesets_joint.pdf"
        plt.savefig(path)

def plot_chain_metrics(embeddings, S, T, classes=[], savefig=True):
    fig, axs = plt.subplots(1,3,num=S.name,subplot_kw={'projection':'3d'})
    fig.set_size_inches(15, 3)
    chain_ax, inter_ax, conns_ax = axs.flat

    cnt_class = {}
    for embedding in embeddings:
        method = embedding.properties["embedding_method"]
        index = classes.index(method)
        cnt_class[method] = 1+cnt_class.get(method,0)

        zs = index*50 + cnt_class[method]
        plt_args = {'color':palette(index),'edgecolor':'k','width':1,
                    'zorder':zs,'zs':zs,'zdir':'y','align':'edge'}

        chain_hist = embedding.chain_histogram()
        chain_ax.bar(chain_hist.keys(),chain_hist.values(),**plt_args)


        inter_hist = embedding.interactions_histogram(S.edges(),T.edges())
        inter_ax.bar(inter_hist.keys(),inter_hist.values(),**plt_args)


        conns_hist = embedding.connectivity_histogram(S.edges(),T.edges())
        conns_ax.bar(conns_hist.keys(),conns_hist.values(),**plt_args)

    for ax in [chain_ax,inter_ax,conns_ax]:
        ax.set_yticks([i*50 for i in range(len(classes))])
        ax.set_yticklabels(classes)
        ax.tick_params('y',labelrotation=-45)

    chain_ax.set_title('Chain Length')
    inter_ax.set_title('Chain Interactions')
    conns_ax.set_title('Qubit Connectivity')

    if savefig:
        path = savefig if isinstance(savefig,str) else "./chain_metrics.pdf"
        plt.savefig(path)
