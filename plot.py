'''
Plotting module for quick methods of making matplotlib plots in pyscf context
'''

from errors import *
import numpy as np
import matplotlib.pyplot as plt
import pyscf as ps

# format matplotlib globally

###############################################################################
#### plotting from txt file

def PlotTxt2D(fname, show = False, handles=[""], styles = [""], labels=["x","y",""]):
    '''
    Take 2D np array stored in txt file and plot x v y
    '''

    # for debugging
    cfname = "PlotTxt2D";

    # unpack data
    dat = np.loadtxt(fname);
    x, y = dat[0], dat[1];

    # check inputs
    if( type(x) != type(np.zeros(1) ) ): # check that x is an np array
        raise PlotTypeError(cfname+" 1st arg must be np array.\n");
    legend=True; # decide whether to implement legend
    if(handles==[""]): # no handles for legend provided
        legend = False;
    if(styles==[]): # no plot style kwargs provided
        pass;
        
    # construct axes
    fig, ax = plt.subplots();
    
    plt.plot(x, y, styles[0], label = handles[0]);

    # format and show
    ax.set(xlabel = labels[0], ylabel = labels[1], title=labels[2]);
    if(legend):
        ax.legend();
        
    if show: plt.show();

    return x, y; # end plot text 2d



###############################################################################
#### plotting directly

def GenericPlot(x,y, handles=[], styles = [], labels=["x","y",""]):
    '''
    Quick x vs y plot
    y can be > 1d and will plot seperate lines
    '''
    
    # for debugging
    cfname = "GenericPlot";
    
    # screen depth of y
    depth = np.shape(y)[0];
    if( len(np.shape(y)) <= 1): # bypass for 1d array inputs
        depth = 1;
        y = np.array([y]);
    
    # check inputs
    if( type(x) != type(np.zeros(1) ) ): # check that x is an np array
        raise PlotTypeError(cfname+" 1st arg must be np array.\n");
    legend=True;
    if(handles==[]): # no handles for legend provided
        handles = np.full(depth, "");
        legend = False;
    if(styles==[]): # no plot style kwargs provided
        styles = np.full(depth, "");
        
    # construct axes
    fig, ax = plt.subplots();

    #iter over y val sets
    for yi in range(depth):
    
        plt.plot(x, y[yi], styles[yi], label = handles[yi]);

    # format and show
    ax.set(xlabel = labels[0], ylabel = labels[1], title=labels[2]);
    if(legend):
        ax.legend();
    plt.show();

    return; # end generic plot


###############################################################################
#### very specific plot functions


def ESpectrumPlot(Evals, title = ""):

    x = np.array([0,1]);
    y = np.zeros((len(Evals), 2));
    for i in range(len(Evals)):
        E = Evals[i];
        y[i,0] = E;
        y[i,1] = E;

    #GenericPlot(x,y,labels = ["","Energy", title+" Energy Spectrum"]);
    return x,y;


def BasisPlot(basisdict, labels, comparediff=False):
    '''
    Given a dict, with
        -keys being string for the basis used in the calc
        -vals being an array of indep var vs energy
    Make line plots comparing E vs indep var for each basis on same figure
    
    Args:
        basis dict, dict
        labels, list of strings for xlabel, ylabel, title
    '''
    
    # for debugging
    fname = BasisPlot;
    
    # check inputs
    if( type(basisdict) != type(dict()) ): # check that basisdict is a dict
        raise PlotTypeError(fname+" 1st arg must be dictionary.\n");
    if( type(labels) != type([]) ): # check that labels is a list
        raise PlotTypeError(fname+" 2nd arg must be a list.\n");
    while( len(labels) < 3): # add dummy labels until we get to three
        labels.append('');
    
    
    # make figure
    # 2 axes required if doing an energy difference comparison
    if(comparediff):
        fig, axs = plt.subplots(2, 1, sharex=True);
        ax, ax1 = axs
    else:
        fig, ax = plt.subplots();
    
    # plot data for each entry in basis dict
    for b in basisdict: # keys are strings
    
        ax.plot(*basisdict[b], label = b);
        
    # plot E diff if asked
    if(comparediff):
    
        #choose baseline data
        baseline = list(basisdict)[0]; # essentially grabs a random key
        basedata = basisdict[baseline][1]; # get E vals that we subtract
    
        # now go thru and get E diff in each case
        for b in basisdict:
        
            data = basisdict[b][1];
            x, y = basisdict[b][0], data - basedata;
            ax1.plot(x, y, label = b);
            
        # format ax1
        ax1.set(xlabel = labels[0], ylabel = labels[1] + " Difference");
        
    # format and show
    ax.set(xlabel = labels[0], ylabel = labels[1], title=labels[2]);
    ax.legend();
    plt.show();

    return; #### end basis plot
    
    
def PlotdtdE():

    # system inputs
    nleads = (4,4);
    nimp = 1;
    nelecs = (nleads[0] + nleads[1] + 1,0); # half filling
    mu = 0;
    Vg = -1.0;

    # time step is variable
    tf = 1.0;
    dts = [0.2, 0.167, 0.1, 0.02, 0.0167, 0.01]

    # delta E vs dt data
    dEvals = np.zeros(len(dts));
    dtvals = np.array(dts);

    # start the file name string
    folderstring = "dat/DotCurrentData/";

    # unpack each _E.txt file
    for i in range(len(dts)):

        dt = dts[i];

        # get arr from the txt file
        fstring = folderstring+"dt"+str(dt)+"_"+ str(nleads[0])+"_"+str(nimp)+"_"+str(nleads[1])+"_e"+str(nelecs[0])+"_mu"+str(mu)+"_Vg"+str(Vg);
        dtdE_arr = np.loadtxt(fstring+"_E.txt");
        
        # what we eant is Ef-Ei
        dEvals[i] = dtdE_arr[1,-1] - dtdE_arr[1,0];
        
    # fit to quadratic
    quad = np.polyfit(dtvals, dEvals, 2);
    tspace = np.linspace(dtvals[0], dtvals[-1], 100);
    quadvals = tspace*tspace*quad[0] + tspace*quad[1] + quad[2];

    # fit to exp
    def e_x(x,a,b,c):
        return a*np.exp(b*x) + c;
    fit = scipy.optimize.curve_fit(e_x, dtvals, dEvals);
    fitvals = e_x(tspace, *fit[0]);
    
    # plot results
    plt.plot(dtvals, dEvals, label = "data");
    plt.plot(tspace, quadvals, label ="Quadratic fit: $y = ax^2 + bx + c$");
    plt.plot(tspace, fitvals, label ="Exponential fit: $y= ae^{bx} + c$");
    plt.xlabel("time step");
    plt.ylabel("E(t=1.0) - E(t=0.0)");
    plt.title("$\Delta E$ vs dt, 4 leads each side");
    plt.legend();
    plt.show();
        
    return # end plot dt dE


def PlotFiniteSize():

    # return vals
    chainlengths = np.array([1,2,3]);
    TimePeriods = np.zeros(len(chainlengths) );

    # prep plot
    fig, (ax1, ax2) = plt.subplots(2);
    
    # how dominant freq depends on length of chain, for dot identical to lead site
    for chaini in range(len(chainlengths) ):

        chainlength = chainlengths[chaini];
        nleads = chainlength, chainlength;
        nelecs = (2*chainlength+1,0); # half filling
        tf = 12.0
        dt = 0.01
        mu = [0.0]
        Vg = [0.0]

        # plot J data for diff chain lengths
        folder = "dat/DotCurrentData/chain/"
        x, J, dummy, dummy = UnpackDotData(folder, nleads, 1, nelecs, mu, Vg);
        x, J = x[0], J[0] ; # unpack lists
        ax1.plot(x,J, label = str(nelecs[0])+" sites");

        # get time period data
        for xi in range(len(x)): # iter over time
            if( J[xi] < 0 and J[xi+1] > 0): # indicates full period
                TimePeriods[chaini] = x[xi];
                break;

    # format J vs t plot (ax1)
    ax1.legend();
    ax1.axhline(color = "grey", linestyle = "dashed")
    ax1.set_xlabel("time (dt = 0.01 s)");
    ax1.set_ylabel("$J*\pi/|V_{bias}|$");
    ax1.set_title("Finite size effects");

    # second plot: time period vs chain length
    numsites = 2*chainlengths + 1;
    ax2.plot(numsites, TimePeriods, label = "Data", color = "black");
    linear = np.polyfit(numsites, TimePeriods, 1); # plot linear fit
    linearvals = numsites*linear[0] + linear[1];
    ax2.plot(numsites, linearvals, label = "Linear fit, m = "+str(linear[0])[:6], color = "grey", linestyle = "dashed");
    
    ax2.legend();
    ax2.set_xlabel("Number of sites")
    ax2.set_ylabel("Time Period (s)");

    # show
    plt.show();
    return; # end plot finite size

    
    
def CorrelPlot(datadict, correl_key, labels):
    '''
    Plot data of energy vs indep var, with and without correl effects included
    
    Args:
    datadict: dictionary with keys name of calc method, vals tuple of x, energy
    correl_key: which key has correl data
    labels: strings for labeling plot
    '''
    
    # for debugging
    fname = CorrelPlot;
    
    # check inputs
    if( type(datadict) != type(dict()) ): # check that basisdict is a dict
        raise PlotTypeError(fname+" 1st arg must be dictionary.\n");
    if( type(labels) != type([]) ): # check that labels is a list
        raise PlotTypeError(fname+" 2nd arg must be a list.\n");
        
    # make figure
    # 1 ax for energy 1 for correl energy
    fig, axs = plt.subplots(2, 1, sharex = True);
    
    # plot energy data
    for k in datadict:
    
        axs[0].plot(*datadict[k], label = k);
        
    # format energy plot
    axs[0].set(xlabel = labels[0], ylabel = labels[1], title=labels[2]);
    axs[0].legend();
    
    # plot correl effects
    correl_energy = datadict[correl_key][1]; # benchmark
    for k in datadict:
        
        if( k != correl_key): # dont plot 0s for the correl energies
            x, y = datadict[k];
            axs[1].plot(x, correl_energy - y);
        
    #format correl effects
    axs[1].set(xlabel = labels[0], ylabel = "Correlation Energy")
    
    #show
    plt.show();


###################################################################################
#### exec code

if __name__ == "__main__":

    Espec = [-121.99, -121.99, -121.99, -121.99, -81.99, -81.99,-81.99,-81.99,-1.622, 0.00007, 0.00007, 0.0237 ]
    labels = np.full(len(Espec), "");
    labels[8] = "S";
    labels[9] = "T+"
    labels[10] = "T-"
    labels[11] = "T0"
    ESpectrumPlot(Espec);
