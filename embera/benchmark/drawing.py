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
        method = embedding.properties.get("embedding_method","N/A")
        runtime = embedding.properties.get("embedding_runtime",0.0)
        ax.set_title(f"{method}\nruntime: {runtime:.2f}s\n{embedding.total_qubits} qubits")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=.85, wspace=0, hspace=0)

    if savefig:
        path = savefig if isinstance(savefig,str) else "./embeddings.pdf"
        plt.savefig(path)

def plot_joint_samplesets(samplesets, info_key=None, gray=False, savefig=True):
    nplots = len(samplesets)
    fig = plt.figure(figsize=(nplots*3, 4))
    grid = plt.GridSpec(5, 5*nplots, hspace=0.0, wspace=0.0)

    def gray2bin(n):
        w = len(n)
        n = int(n, 2)
        mask = n
        while mask != 0:
            mask >>= 1
            n ^= mask
        return format(n,f'0{w}b')

    minE = float('Inf')
    maxE = -float('Inf')
    x = {}; y = {}; E = {}; c = {}
    for i,sampleset in enumerate(samplesets):

        x[i] = []; y[i] = []; E[i] = []; c[i] = []
        if not sampleset: continue

        size = len(sampleset.variables)
        width = 2**(size//2)-1
        height = 2**(size-size//2)-1

        # Reverse iteration allows plotting lower (important) samples on top.
        for datum in sampleset.data(sorted_by='energy',reverse=True):
            value = ''.join(str((1+datum.sample[k])//2) for k in sorted(datum.sample))
            x_point = gray2bin(value[0:size//2]) if gray else value[0:size//2]
            y_point = gray2bin(value[size//2:]) if gray else value[size//2:]
            x[i].append(int(x_point,2)/width)
            y[i].append(int(y_point,2)/height)
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

        h_params = {'frameon':False,'autoscale_on':False,'xticks':[],'yticks':[]}
        y_hist = fig.add_subplot(grid[1:5,4+(i*5)],sharey=main_ax,**h_params)
        x_hist = fig.add_subplot(grid[0,i*5:4+(i*5)],sharex=main_ax,**h_params)

        if not sampleset: main_ax.set_xlabel('N/A'); continue

        # Scatter points on the main axes
        ratE = [5+250*(((energy-minE)/rangeE)**2) for energy in E[i]]
        sct = main_ax.scatter(x[i],y[i],s=ratE,c=E[i],cmap="jet",alpha=0.5)

        minXY = [(x[i][ie],y[i][ie]) for ie,e in enumerate(E[i]) if e==minE]
        if minXY: main_ax.plot(*zip(*minXY),ms=25,mew=1,c='k',marker='x')

        # Histograms on the attached axes
        x_hist.hist(x[i], 100, histtype='stepfilled',
                    orientation='vertical', color='gray')

        y_hist.hist(y[i], 100, histtype='stepfilled',
                    orientation='horizontal', color='gray')

        ims.append(sct)
        main_ax.set_xlabel(sampleset.info.get(info_key,"N/A"))


    # Color Bar
    vmin,vmax = zip(*[im.get_clim() for im in ims])

    for i,im in enumerate(ims):
        im.set_clim(vmin=min(vmin),vmax=max(vmax))

    plt.subplots_adjust(top=1,bottom=0.25,left=.05,right=.95,hspace=0,wspace=0)

    cax = fig.add_axes([0.25,0.15,0.5,0.02]) # [left,bottom,width,height]
    plt.colorbar(sct,orientation='horizontal',cax=cax)
    _ = cax.set_xlabel('Energy')

    if savefig:
        path = savefig if isinstance(savefig,str) else "./samplesets_joint.pdf"
        plt.savefig(path)

def plot_chain_metrics(embeddings, S, T, key=None, tags=[], savefig=True):

    fig, axs = plt.subplots(1,2,num=S.name,subplot_kw={'projection':'3d'})
    fig.set_size_inches(10, 4)
    chain_ax, inter_ax = axs.flat

    tag_cnt = {}
    for embedding in embeddings:
        emb_tag = embedding.properties.get(key)
        index = tags.index(emb_tag) if emb_tag is not None else 0
        tag_cnt[emb_tag] = 1 + tag_cnt.get(emb_tag,0)

        zs = index*50 + tag_cnt[emb_tag]
        plt_args = {'color':palette(index),'edgecolor':'k','width':1,
                    'zorder':zs,'zs':zs,'zdir':'y','align':'edge'}

        chain_hist = embedding.chain_histogram()
        chain_ax.bar(chain_hist.keys(),chain_hist.values(),**plt_args)


        inter_hist = embedding.interactions_histogram(S.edges(),T.edges())
        inter_ax.bar(inter_hist.keys(),inter_hist.values(),**plt_args)

    for ax in [chain_ax,inter_ax]:
        ax.set_yticks([i*50 for i in range(len(tags))])
        ax.set_yticklabels(tags)
        ax.tick_params('y',labelrotation=-45)

    chain_ax.set_title('Chain Length')
    inter_ax.set_title('Chain Interactions')

    plt.subplots_adjust(top=1,bottom=0.15,left=0,right=1,hspace=0,wspace=0)

    if savefig:
        path = savefig if isinstance(savefig,str) else "./chain_metrics.pdf"
        plt.savefig(path)