#!/usr/bin/python3

# reading/writing files
from pathlib import Path
import csv

# quantification analysis
import math, sys
import numpy as np
import pandas as pd

# statistical analysis
from scipy.stats import ttest_ind
from scipy import stats
import statsmodels.stats.multitest as multitest

# plotting
import matplotlib as mpl
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
import seaborn as sns

# plotting options
mpl.use("qt5agg")
mpl.rcParams['pdf.fonttype'] = 42 # print editable font
mpl.rcParams['ps.fonttype'] = 42 # print editable font

###################
#### FUNCTIONS ####
###################

def RotationMatrix(v1,v2):
    v = np.cross( np.array(v1),np.array(v2)) # cross product v = v1 x v2 
    c = np.dot( np.array(v1),np.array(v2) )  # cos angle 

    # skew-symmetric cross-product matrix; needs to be negative to rotate in correct direction
    cross_matrix = np.array( [[    0., +v[2] , -v[1] ],
                             [ -v[2],    0. , +v[0] ],
                             [ +v[1], -v[0] ,    0. ] ])

    cross_matrix_squared = np.linalg.matrix_power(cross_matrix, 2)
    cross_matrix_squared_extrabits = (1./(1.+c))*cross_matrix_squared

    Rot_mat = np.identity(3, dtype = float)
    Rot_mat = Rot_mat.__add__(cross_matrix)
    Rot_mat = Rot_mat.__add__(cross_matrix_squared_extrabits)
    return Rot_mat

###############################
# Dataframe utility functions #
###############################
def new_col_DV(df):

    df_out = df.copy()

    new_col = [0]*df.shape[0]
    for idx, item in enumerate(df_out['Cat']):
        if item == 'VD' or item == 'AD':
            new_col[idx] = 'dorsal'
        elif item == 'VV' or item == 'AV':
            new_col[idx] = 'ventral'
    df_out['dv'] = new_col

    return(df_out)

def rm_rename_cats(df):

    df_out = df.copy()

    conditions = [
                 df['Cat'] == 'VV',
                 df['Cat'] == 'VD',
                 df['Cat'] == 'AV',
                 df['Cat'] == 'AD',
                 df['Cat'] == 'AVC'
                 ]

    outputs = [
                'V',
                'V', 
                'A',
                'A',
                'AVC'
              ]

    for cond, out in zip(conditions, outputs):
        df_out.loc[(cond),'Cat'] = out

    df_out.drop(df_out[~(df_out['Cat'].isin(outputs))].index, inplace=True)

    return(df_out)

# Get the row-wise vector magnitude
def row_mag_groupby(group):
    colnames = ['magnitude']
    idx = group.index
    v = group.to_numpy()
    # Diagonal of array times its transpose = sum of squared elements of each row
    v_mag = np.sqrt( np.diag( np.dot( v, v.T ) ) )

    res = pd.DataFrame(v_mag, columns = colnames, index = idx)
    return(res)

##########
# LINFIT #
##########

# Do a linear fit on several intervals of the data and return the slope for each interval
def lin_fit_function(group, ydata = 'total_rotation', interval_hours = 1.5, maxh = 12):
    numsteps = math.ceil(maxh/interval_hours)+1 # +1 to include border
    bins = np.linspace(0, maxh, numsteps)
    
    ts = group['time_hr']
    df_twistgroups = group.groupby(pd.cut(ts, bins)) # bins df by delta t (physical time)

    l_regr = [] 
    l_regr2 = []
    for df_twist_name, df_twistg in df_twistgroups:
        if not df_twistg.empty: # Some intervals might be empty for this dataset
            x = df_twistg['time_hr']
            y = df_twistg[ydata]
            model = np.polyfit(x, y, 1)
            l_regr.append(model[0])
            l_regr2.append(model[1])
        else:
            l_regr.append(np.nan)
            l_regr2.append(np.nan)

    ys = [ q+m*x for (x,m,q) in zip(bins[:-1], l_regr, l_regr2) ]

    colnames = ['time_intervals', 'angvel', 'intercepts', 'points']
    
    # Transpose the lists otherwise they become looong rows
    newdf = pd.DataFrame(list(map(list, zip(*[bins, l_regr, l_regr2, ys]))), columns = colnames)
    
    return newdf #bins, l_regr, l_regr2, ys # time, angvel, intercepts, points on regression line

####################################
# Data subsets for stats and plots #
####################################

def createDataSubsets(datalists):

    #########################################################
    # Take the total twist for each replicate after 9 hours #
    #########################################################
    after9 = fit_twist_nr.loc[fit_twist_nr['time_intervals'] > 9].copy()
    after9.dropna(subset=['twisting angle'], inplace=True)

    wt_after9 = after9.loc[after9['ec'] == 'wt']
    oug_after9 = after9.loc[after9['ec'] == 'oug']

    # Get the mean for each replicate at the last bin
    wt_after9_mean = wt_after9.groupby(['ec', 'replicate']).mean().reset_index()
    oug_after9_mean = oug_after9.groupby(['ec','replicate']).mean().reset_index()

    df_after9 = pd.concat([wt_after9_mean, oug_after9_mean])
    df_after9.rename(columns = {'twisting angle':'mean total twist'}, inplace = True)
    df_after9['ecrep'] = df_after9['ec'] + ' ' + df_after9['replicate'] 

    datalists.append(df_after9)

    ##########################
    #  Total twist wt vs oug #
    ##########################
    df_test_twist = datalists[2].copy().dropna(subset=['angvel']) # fit on total rotation difference

    # mean slope for each experiment
    #df_test_twist = df_test_twist.groupby(['ec', 'replicate'])['angvel'].mean().reset_index()

    # mean slope for each experiment and interval
    df_test_twist = df_test_twist[['ec', 'replicate', 'time_intervals', 'angvel']].copy().reset_index()
    df_test_twist['ecrep'] =  df_test_twist['ec'] + ' ' + df_test_twist['replicate']
    df_test_twist['ectime'] =  df_test_twist['ec'] + ' ' + df_test_twist['time_intervals'].astype(str)
    datalists.append(df_test_twist)

    #############################################
    # Total rotation wt vs oug for each chamber #
    #############################################
    df_test_totrot = datalists[3].copy().dropna(subset=['angvel']) # fit on total rotation

    # mean slope for each experiment
    #df_test_totrot = df_test_totrot.groupby(['ec', 'replicate', 'Cat'])['angvel'].mean().reset_index()

    # mean slope for each experiment and interval
    df_test_totrot = df_test_totrot[['ec', 'replicate', 'Cat', 'time_intervals', 'angvel']].copy().reset_index() # df1 = df[['a', 'b']]
    
    df_test_totrot['group'] = df_test_totrot['ec'] + df_test_totrot['Cat']
    df_test_totrot['catrep'] =  df_test_totrot['replicate'] + ' ' + df_test_totrot['Cat']
    df_test_totrot['timerep'] =  df_test_totrot['time_intervals'].astype(str) + ' ' + df_test_totrot['Cat']
    datalists.append(df_test_totrot)

    return(datalists)

#########
#########
# Stats #
#########
#########


def doStats(datalists):

    # datalists = [result_twist, 
    #              result_totrot, 
    #              fit_twist, 
    #              fit_totrot, 
    #              fit_twist_nr,
    #              df_after9,
    #              df_test_twist,
    #              df_test_totrot]

    df_after9 = datalists[5]

    wt_after9_mean = df_after9.loc[df_after9['ec'] == 'wt']
    oug_after9_mean = df_after9.loc[df_after9['ec'] == 'oug']
    after9_ttest = ttest_ind(wt_after9_mean['mean total twist'].tolist(), oug_after9_mean['mean total twist'].tolist(), equal_var=False)
    after9_mww = stats.mannwhitneyu(wt_after9_mean['mean total twist'].tolist(), oug_after9_mean['mean total twist'].tolist())

    df_test_twist = datalists[6]
    wt = df_test_twist.loc[df_test_twist['ec'] == 'wt']
    mt = df_test_twist.loc[df_test_twist['ec'] == 'oug']
    
    # t-test
    result_ttest = ttest_ind(wt['angvel'].tolist(), mt['angvel'].tolist())
    twist_stats = ["twist velocity wt vs oug t-test p-value: ", result_ttest.pvalue]
    
    # mann-whitney/wilcoxon rank-sum test
    result_mww = stats.mannwhitneyu(wt['angvel'].tolist(), mt['angvel'].tolist())
    twist_mww = ["twist velocity wt vs oug mww-rank-sum-test p-value: ", result_mww.pvalue]

    twist_normality_strings = [
                    "wt total twist; Wilk-Shapiro normality (alpha=0.05);",
                    "oug total twist; Wilk-Shapiro normality (alpha=0.05);",
                    "wt twisting velocity; Wilk-Shapiro normality (alpha=0.05);",
                    "oug twisting velocity; Wilk-Shapiro normality (alpha=0.05);"
                 ]

    list_datasets_twist = [
                            wt_after9_mean['mean total twist'].tolist(),
                            oug_after9_mean['mean total twist'].tolist(),
                            wt['angvel'].tolist(),
                            mt['angvel'].tolist()
                          ]

    # Test normality of distribution
    for idx, i in enumerate(list_datasets_twist):
        alpha = 0.05
        k2, p =stats.shapiro(i)
        if p < alpha:  # null hypothesis: x comes from a normal distribution
            twist_normality_strings[idx] += "p-value; " + str(round(p,4)) + "; not normal"
        else:
            twist_normality_strings[idx] += "p-value; " + str(round(p,4)) + "; normal"
    
    df_test_totrot = datalists[7]

    # WILDTYPE
    wt = df_test_totrot.loc[df_test_totrot['ec'] == 'wt']
    wt_VA = wt.loc[wt['Cat'].isin(['V', 'A'])]
    wt_AVC = wt.loc[wt['Cat'].isin(['AVC'])]
    
    # MUTANT
    mt = df_test_totrot.loc[df_test_totrot['ec'] == 'oug']
    mt_VA = mt.loc[mt['Cat'].isin(['V', 'A'])]
    mt_AVC = mt.loc[mt['Cat'].isin(['AVC'])]

    # Get the groups for statistical tests
    wt_V = wt.loc[wt['Cat'] == 'V']['angvel'].tolist()
    wt_A = wt.loc[wt['Cat'] == 'A']['angvel'].tolist()
    wt_C = wt.loc[wt['Cat'] == 'AVC']['angvel'].tolist()

    mt_V = mt.loc[mt['Cat'] == 'V']['angvel'].tolist()
    mt_A = mt.loc[mt['Cat'] == 'A']['angvel'].tolist()
    mt_C = mt.loc[mt['Cat'] == 'AVC']['angvel'].tolist()

    # ANOVA
    anovalist = [wt_V, wt_A, wt_C, mt_V, mt_A, mt_C] # 2K, 4I, supplements 
    
    angvel_normality_strings = [
                    "wt ventricle angular velocity; Wilk-Shapiro normality (alpha=0.05);",
                    "wt atrium angular velocity; Wilk-Shapiro normality (alpha=0.05);",
                    "wt AVC angular velocity; Wilk-Shapiro normality (alpha=0.05);",
                    "oug ventricle angular velocity; Wilk-Shapiro normality (alpha=0.05);",
                    "oug atrium angular velocity; Wilk-Shapiro normality (alpha=0.05);",
                    "oug AVC angular velocity; Wilk-Shapiro normality (alpha=0.05);"
                 ]

    # Test normality of distribution
    for idx, i in enumerate(anovalist):
        alpha = 0.05
        k2, p =stats.shapiro(i)
        if p < alpha:  # null hypothesis: x comes from a normal distribution
            angvel_normality_strings[idx] += "p-value; " + str(round(p,4)) + "; not normal"
        else:
            angvel_normality_strings[idx] += "p-value; " + str(round(p,4)) + "; normal"

    anova_F, anova_p = stats.f_oneway(*anovalist)
    anova = ["ANOVA F-value: ", anova_F, "ANOVA p-value: ", anova_p]

    pairings = [(wt_V, wt_A), (wt_A, mt_A), (wt_V, mt_V), (mt_V, mt_A)]
    names = ['wt V vs wt A', 'wt A vs oug A', 'wt V vs oug V', 'oug V vs oug A']

    pairings += [(wt_V, wt_C), (wt_A, wt_C), (wt_C, mt_C), (mt_V, mt_C), (mt_A, mt_C)]
    names += ['wt V vs wt AVC', 'wt A vs wt AVC', 'wt AVC vs oug AVC', 'oug V vs oug AVC', 'oug A vs oug AVC']
    ttest_res = []
    mww_res = []

    for pair in pairings:
        # t-test
        pairs_ttest = ttest_ind(pair[0], pair[1], equal_var=False)
        ttest_res.append(pairs_ttest.pvalue)

        # mann-whitney/wilcoxon rank-sum test
        result_mww = stats.mannwhitneyu(pair[0], pair[1])
        mww_res.append(result_mww.pvalue)

    corr = multitest.multipletests(ttest_res, alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)
    corr2 = multitest.multipletests(mww_res, alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)
    # returns: reject _array, pvals_corrected _array, alphacSidak _float, alphacBonf _float

    # Write statistics to CSV
    res_file = "corrected_pvalues.csv"
    dl = ';'
    with open(res_file, 'w', newline='') as csvfile:
        f = csv.writer(csvfile, delimiter=dl)
        f.writerow(twist_stats)
        f.writerow("\n")
        f.writerow(anova)
        f.writerow("\n")
        for substring in twist_normality_strings:
            f.writerow([substring])
        f.writerow("\n")
        stringtemp = "Total twist wt vs oug: " + dl + str(after9_ttest)
        f.writerow([stringtemp])
        f.writerow("\n")
        stringtemp = "Total twist wt vs oug: " + dl + str(after9_mww)
        f.writerow([stringtemp])
        f.writerow("\n")
        f.writerow("\n")
        for substring in angvel_normality_strings:
            f.writerow([substring])
        f.writerow("\n")
        f.writerow(["Angular velocity wt vs oug for each chamber: "])
        f.writerow(["pair" + dl + "t-test" + dl + "p-value" + dl + "samples from same distribution?" + dl + "mww-rank-sum-test" + dl + "p-value" + dl + "samples from same distribution?"])
        for n, i, j, k, l in zip(names, corr[1], corr[0], corr2[1], corr2[0]):
            string = n + dl + "t-test" + dl + str(i) + dl + str(j) + dl + "mww-test" + dl + str(k) + dl + str(l)
            f.writerow([string])
        f.writerow("\n")

    #Output source data
    df_after9[['ec', 'replicate', 'mean total twist']].to_csv('total_twist_statistics.csv')
    df_test_twist[['ec', 'replicate', 'angvel']].to_csv('twist_angvel_statistics.csv')
    df_test_totrot.to_csv('total_rotation_statistics.csv')

#########
#########
# Plots #
#########
#########

################
# Movie Frames #
################

# vector addition/subtraction for grouped dataframes
def vec_add_groupby(group, v, subtract = False):
    colnames = ['x+v', 'y+v', 'z+v']
    if subtract:
        v = -v
        colnames = ['x-v', 'y-v', 'z-v']
    idx = group.index
    A = group.to_numpy()
    B = np.tile(v, (len(group),1))
    C = A + B
    res = pd.DataFrame(C, columns = colnames, index = idx)
    return(res)

# subtract first element on per-track basis
def df_sub_init_xyz_df(group, cols, subtract = True):
    t_min = group['Time'].min()
    xyz_min = group.loc[ group['Time'] == t_min ][cols]
    res = vec_add_groupby(group[cols], xyz_min, subtract = subtract)
    return(res)

def plot_frames(df_all_data):
    print("Doing 3D scatterplots.")

    xyz = ['Position X','Position Y','Position Z'] # original coordinates
    xyzrot = ['x_rot', 'y_rot', 'z_rot'] # rotated coordinates

    for expcond, df_data in df_all_data.items():

        replicates = df_data['replicate'].unique()
        for rep in replicates:
            df = df_data.loc[df_data['replicate'] == rep]

            # Displacement trajectories 
            # t+1 - t
            dxyz_ori = ['dx_ori', 'dy_ori', 'dz_ori']
            dxyz_rot = ['dx_rot', 'dy_rot', 'dz_rot']

            df = df.sort_values(by=['Time']) # Sort dataframe by time     
            df[dxyz_ori] = df.groupby(['TrackID'])[xyz].diff()
            df[dxyz_rot] = df.groupby(['TrackID'])[xyzrot].diff()

            # t - t0
            dxyz0_ori = ['dx0_ori', 'dy0_ori', 'dz0_ori']
            dxyz0_rot = ['dx0_rot', 'dy0_rot', 'dz0_rot']
            
            df[dxyz0_ori] = df.groupby(['TrackID']).apply(df_sub_init_xyz_df, xyz, subtract=True)
            df[dxyz0_rot] = df.groupby(['TrackID']).apply(df_sub_init_xyz_df, xyzrot, subtract=True)

            # Get reference centroid for original coordinates to get axis limits
            mint = min(df['Time'].unique()) # first timepoint
            avc = df.loc[df['Cat'] == 'AVC']
            avc_mint = avc.loc[df['Time'] == mint]
            reference = avc_mint[xyz].mean().to_numpy()
            
            xlim1 = [reference[0]-100, reference[0]+100]
            ylim1 = [reference[1]-100, reference[1]+100]
            zlim1 = [reference[2]-100, reference[2]+100]
            
            # Plot: scatter point and plot axes
            plt.style.use('dark_background')
            plt.rcParams['grid.color'] = "dimgray"
            fig = plt.figure()
            fig.set_size_inches(32, 18) # set figure's size manually

            ax1 = fig.add_subplot(1, 2, 1, projection='3d') # original data
            ax2 = fig.add_subplot(1, 2, 2, projection='3d') # data translated and rotated
            axs = [ax1, ax2]

            # Construct a color palette with unique colors for each TrackID
            chambers = ['V', 'A', 'AVC']
            palettes = ['flare', 'crest', 'Greys'] # matplotlib/seaborn palettes
            cdict = {'TrackID' : [], 'color' : []}
            for i in range(len(chambers)):
                tracks_nunique = df.loc[df['Cat'] == chambers[i]]['TrackID'].nunique()
                tracks_unique = df.loc[df['Cat'] == chambers[i]]['TrackID'].unique()
                cp = sns.color_palette(palettes[i], tracks_nunique)
                cdict['TrackID'].extend(tracks_unique)
                cdict['color'].extend(cp)
                
            color_df = pd.DataFrame.from_dict(cdict)
            df = pd.merge(df, color_df, left_on='TrackID', right_on='TrackID')

            # For plotting trajectories
            time_idx_list = []
            maxt = max(df['Time'])
            s = 50

            grp = df.groupby(['Time'])

            for time_idx, time_group in grp:
                print("Plotting time step: ", time_idx)

                # To plot trajectories, get all timepoints up to the current one
                time_idx_list.append(time_idx)
                traj = df.loc[df['Time'].isin(time_idx_list)]

                for _, track in traj.groupby(['TrackID']):
                    color = track['color'].unique()[0]
                    a = 0.25
                    lw = 2
                    ax1.plot3D(track[xyz[0]],    track[xyz[1]],    track[xyz[2]],    color = color, alpha = a, linewidth = lw )
                    ax2.plot3D(track[xyzrot[0]], track[xyzrot[1]], track[xyzrot[2]], color = color, alpha = a, linewidth = lw )

                # Plot the points
                ax1.scatter( time_group['Position X'], time_group['Position Y'], time_group['Position Z'], s = s, c = time_group['color'], depthshade=False )
                ax2.scatter( time_group['x_rot'],      time_group['y_rot'],      time_group['z_rot'],      s = s, c = time_group['color'], depthshade=False )

                ax1.set_title("raw data")
                ax2.set_title("axes stabilized")

                # Plot displacement vectors
                colnamelist = [ xyz, xyzrot ]

                # Plot the axes using the centCC
                centCC = time_group.groupby(['Cat']).mean().reset_index()
                color_a = 'dodgerblue'
                color_v = 'saddlebrown'

                for col_idx, collist in enumerate(colnamelist):

                    a_xyz = centCC.loc[centCC['Cat']=='A'][collist].to_numpy()
                    c_xyz = centCC.loc[centCC['Cat']=='AVC'][collist].to_numpy()
                    v_xyz = centCC.loc[centCC['Cat']=='V'][collist].to_numpy()

                    lw = 7
                    axs[col_idx].plot( [ a_xyz[0,0], c_xyz[0,0] ] , 
                                        [ a_xyz[0,1], c_xyz[0,1] ] , 
                                        [ a_xyz[0,2], c_xyz[0,2] ] , 
                                        '-', c=color_a, label="atrium axis", linewidth = lw )
                    axs[col_idx].plot( [ v_xyz[0,0], c_xyz[0,0] ] , 
                                        [ v_xyz[0,1], c_xyz[0,1] ] , 
                                        [ v_xyz[0,2], c_xyz[0,2] ] , 
                                        '-', c=color_v, label="ventricle axis", linewidth = lw )

                # Plot aesthetics
                for ax in axs:
                    ax.quiver(0,0,0,1,0,0, color = 'gray', length=25)
                    ax.quiver(0,0,0,0,1,0, color = 'gray', length=25)
                    ax.quiver(0,0,0,0,0,1, color = 'gray', length=25)

                    ax.set_xlim([-100,100])
                    ax.set_ylim([-100,100])
                    ax.set_zlim([-100,100])
                    ax.legend()
                    
                    # Grid and pane aesthetics
                    # Transparent spines
                    ax.w_xaxis.line.set_color((0.5, 0.5, 0.5, 0.5))
                    ax.w_yaxis.line.set_color((0.5, 0.5, 0.5, 0.5))
                    ax.w_zaxis.line.set_color((0.5, 0.5, 0.5, 0.5))

                    # Transparent panes
                    ax.w_xaxis.set_pane_color((0.5, 0.5, 0.5, 0))
                    ax.w_yaxis.set_pane_color((0.5, 0.5, 0.5, 0))
                    ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 0))

                    ax.view_init(elev=75., azim=300)  # "front" view with ventricle on top, atrium on bottom

                    # Font sizes
                    fontItems = [ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]
                    fontItems += ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
                    try:
                        fontItems += ax.get_legend().get_texts()
                    except Exception: # no legend
                        pass
                    for item in fontItems:
                        item.set_fontsize(20)

                ax1.set_xlim(xlim1)
                ax1.set_ylim(ylim1)
                ax1.set_zlim(zlim1)

                # Create an output directory
                outputpath = Path("Output")
                outputpath.mkdir(parents = True, exist_ok=True)
                outputpath_child = outputpath / Path(rep)
                outputpath_child.mkdir(parents = True, exist_ok=True)
                filestring = rep+"_t"+f'{time_idx:03}'+".png"
                fig.savefig(outputpath_child / filestring, bbox_inches='tight')

                for ax in axs:
                    if time_idx != maxt:
                        ax.cla()

###############################################################
# Plot vectors from start to end using corrected coordinates. #
###############################################################

# Matplotlib tools for drawing arrows in 3D (e.g. quiver) don't allow good control of arrowhead aesthetics.
# As an alternative, we inherit the 2D FancyArrowPatch method and apply a 3D transform on it.
class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

def plotStartEndVectors(df_all_data):
    for expcond, df_data in df_all_data.items():
        replicates = df_data['replicate'].unique()
        for rep in replicates:
            df = df_data.loc[df_data['replicate'] == rep]

            # Set up figure
            plt.style.use('dark_background')
            plt.rcParams['grid.color'] = "dimgray"
            fig = plt.figure()
            fig.set_size_inches(32, 18)
            ax = fig.add_subplot(1, 1, 1, projection='3d')

            # Construct a color palette with unique colors for each TrackID
            chambers = ['V', 'A', 'AVC']
            palettes = ['deeppink', 'cyan', 'gainsboro'] # matplotlib colors
            cdict = {'TrackID' : [], 'color' : []}
            for i in range(len(chambers)):
                dfs = df.sort_values(by=['z_rot'], ascending=True) # sort to color by depth (regardless of WHEN the depth was reached)
                tracks_nunique = dfs.loc[dfs['Cat'] == chambers[i]]['TrackID'].nunique()
                tracks_unique = dfs.loc[dfs['Cat'] == chambers[i]]['TrackID'].unique()
                cp = sns.dark_palette(palettes[i], tracks_nunique)
                cdict['TrackID'].extend(tracks_unique)
                cdict['color'].extend(cp)

            color_df = pd.DataFrame.from_dict(cdict)
            df = pd.merge(df, color_df, left_on='TrackID', right_on='TrackID')

            # Subtract the first timepoint on a per-track basis to get displacement vector
            # Note that each track may have its own min/max time in the time-lapse
            xyzrot = ['x_rot', 'y_rot', 'z_rot'] # rotated coordinates
            dxyz0_rot = ['dx0_rot', 'dy0_rot', 'dz0_rot']
            df[dxyz0_rot] = df.groupby(['TrackID']).apply(df_sub_init_xyz_df, xyzrot, subtract=True)

            for _, track in df.groupby(['TrackID']):
                
                mint = min(track['Time'].unique()) # first timepoint this track exists
                maxt = max(track['Time'].unique()) # final timepoint this track exists
                traj = track.loc[track['Time'].isin([mint, maxt])] 

                magcol = traj.groupby(['TrackID'])[dxyz0_rot].apply(row_mag_groupby)
                traj = traj.join(magcol)

                color = traj['color'].unique()[0]

                first_xyz_loc = traj.loc[traj['Time'] == mint][xyzrot]
                final_xyz_dir = traj.loc[traj['Time'] == maxt][dxyz0_rot]

                startpoint = ( first_xyz_loc[xyzrot[0]].tolist()[0], first_xyz_loc[xyzrot[1]].tolist()[0], first_xyz_loc[xyzrot[2]].tolist()[0] )
                vec = (final_xyz_dir[dxyz0_rot[0]].tolist()[0], final_xyz_dir[dxyz0_rot[1]].tolist()[0], final_xyz_dir[dxyz0_rot[2]].tolist()[0])
                
                a = Arrow3D(startpoint[0], # x
                            startpoint[1], # y
                            startpoint[2], # z
                            vec[0], # dx
                            vec[1], # dy
                            vec[2], # dz
                            mutation_scale=20, # Value with which attributes of arrowstyle (e.g., head_length) will be scaled.
                            lw=3, 
                            arrowstyle="-|>", 
                            color=color)
                ax.add_artist(a)

            # Plot aesthetics
            ax.set_xlim([-100,100])
            ax.set_ylim([-100,100])
            ax.set_zlim([-100,100])

            # Transparent spines
            ax.w_xaxis.line.set_color((0.5, 0.5, 0.5, 0.5))
            ax.w_yaxis.line.set_color((0.5, 0.5, 0.5, 0.5))
            ax.w_zaxis.line.set_color((0.5, 0.5, 0.5, 0.5))

            # Transparent panes
            ax.w_xaxis.set_pane_color((0.5, 0.5, 0.5, 0))
            ax.w_yaxis.set_pane_color((0.5, 0.5, 0.5, 0))
            ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 0))

            ax.view_init(elev=75., azim=300)  # "front" view with ventricle on top, atrium on bottom

            # Font sizes
            fontItems = [ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]
            fontItems += ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
            try:
                fontItems += ax.get_legend().get_texts()
            except Exception: # no legend
                pass
            for item in fontItems:
                item.set_fontsize(20)

            # Create an output directory if it doesn't exist and save figures there
            outputpath = Path("Output")
            outputpath.mkdir(parents = True, exist_ok=True)
            filestring = "vectors_start-end_" + rep + ".png"
            fig.savefig(outputpath / filestring, bbox_inches='tight')

###############
# Other plots #
###############

def plot_lines(df_data, ax, y, pal = 'inferno'):

    style = 'replicate'

    g = sns.lineplot(x = 'time_hr',
                        y = y,
                        hue = 'replicate',
                        style = style,
                        data = df_data,
                        ax = ax,
                        palette = pal,
                        linewidth = 2
                    )

def plot_swarm(df_data, ax, y = 'angvel', pal = 'inferno'):

    style = 'replicate'
    dodge = False

    grouping = 'replicate'
    g = sns.swarmplot(x = grouping,
                      y = y,
                      data = df_data,
                      ax = ax,
                      palette = pal,
                      hue = style,
                      dodge = dodge
                      )
    
    # Plot the mean as a bar
    means = df_data.groupby([grouping], sort=False)[y].mean()
    for xtick in g.get_xticks():
        ax.plot([xtick-0.1, xtick+0.1], 
            [means[xtick], means[xtick]], 
            color='gray', linewidth='5')

def adjustPlotAesthetics(ax, ylab = False):
    ymin = 1
    ymax = -1

    for axis in ax:
        # Y axis
        y1, y2 = axis.get_ylim()
        ymin = min(ymin, y1)
        ymax = max(ymax, y2)
    
    for axis in ax:
        sns.despine(top=True, right=True, ax=axis)

        # Y axis
        axis.set_ylim(ymin, ymax)
        if ylab:
            axis.set_ylabel(ylab)

        # X axis
        x1, x2 = axis.get_xlim()

        xl = x1*0.95
        xr = x2*1
        axis.set_xlim(xl, xr)

        # 0 line
        axis.hlines(0, xl, xr, colors = 'black', linestyles='dashdot', linewidth=0.5)

        # Font sizes
        fontItems = [axis.title, axis.xaxis.label, axis.yaxis.label]
        fontItems += axis.get_xticklabels() + axis.get_yticklabels()
        try:
            fontItems += axis.get_legend().get_texts()
        except Exception: # no legend
            pass
        for item in fontItems:
            item.set_fontsize(12)

def makePlots(datalists):

    # Set up the figures
    plt.style.use('default')

    ################
    # Do the plots #
    ################

    # datalists = [result_twist, 
    #              result_totrot, 
    #              fit_twist, 
    #              fit_totrot, 
    #              fit_twist_nr,
    #              df_after9,
    #              df_test_twist,
    #              df_test_totrot]

    result_totrot = datalists[1]
    fit_twist_nr = datalists[4]
    df_after9 = datalists[5]
    df_test_twist = datalists[6]
    df_test_totrot = datalists[7]

    plot_source_data = {}


    ############
    # Figure 2 #
    ############

    # Figure 2: total rotation + average angular velocity WT
    fig_angvel_wt = plt.figure(figsize=(16,8))
    ax_wt_1 = fig_angvel_wt.add_subplot(121) # total rotation atrium vs ventricle
    ax_wt_2 = fig_angvel_wt.add_subplot(122) # point swarm average angular velocity per interval

    try:
        # Total rotation
        df_totrot = result_totrot.loc[result_totrot['ec']=='wt']
        plot_lines(df_totrot.loc[df_totrot['Cat']=='V'], ax_wt_1 , y = 'total_rotation', pal = 'PuRd')
        plot_lines(df_totrot.loc[df_totrot['Cat']=='A'], ax_wt_1 , y = 'total_rotation', pal = 'PuBu')

        # Add custom legends to plot
        first_legend = ax_wt_1.get_legend()
        ax_wt_1.add_artist(first_legend)

        names = ['ventricle', 'atrium']
        pal  = sns.color_palette('PuRd', 1)
        pal += sns.color_palette('PuBu', 1)
        cdict = dict(zip(names, pal))

        custom_lines = []
        custom_legs = []
        for k, v in cdict.items():
            custom_lines.append(Line2D([0],[0], color=v, lw=4))
            custom_legs.append(k)

        # Create another legend
        ax_wt_1.legend(custom_lines, custom_legs, title="chamber", fontsize=4, loc='lower center')

        df_totrot_VA = df_totrot.drop(df_totrot[df_totrot['Cat'] == 'AVC'].index)
        plot_source_data['figure 2 total rotation'] = df_totrot_VA[['ec', 'replicate', 'Cat', 'time_hr', 'total_rotation']]
    except Exception:
        print()
        print("An error occurred. Dataset may be empty.")
        print("Skipping 'figure 2 total rotation wt'")
        print()

    # Swarmplot angular velocity
    try:
        wt = df_test_totrot.loc[df_test_totrot['ec'] == 'wt']
        wt_VA = wt.loc[wt['Cat'].isin(['V', 'A'])]

        order = ["wtV", "wtA"] 
        names = ['V', 'A']
        pal  = sns.color_palette('PuRd', 1)
        pal += sns.color_palette('PuBu', 1)
        cdict = dict(zip(names, pal))

        g = sns.swarmplot(x = 'group', y = 'angvel', data = wt_VA, ax = ax_wt_2, palette = cdict, hue = 'Cat', order=order)
        means = df_test_totrot.groupby(['group'])['angvel'].mean()
        for elem, xtick in zip(order, g.get_xticks()):
            ax_wt_2.plot([xtick-0.1, xtick+0.1], [means[elem], means[elem]], color='gray', linewidth='5')

        plot_source_data['figure 2 angular velocity'] = wt_VA
    except Exception:
        print()
        print("An error occurred. Dataset may be empty.")
        print("Skipping 'figure 2 angular velocity wt'")
        print()

    ############
    # Figure 4 #
    ############

    # Figure 4: total rotation + average angular velocity MT
    fig_angvel_mt = plt.figure(figsize=(16,8))
    ax_mt_1 = fig_angvel_mt.add_subplot(121) # total rotation atrium vs ventricle
    ax_mt_2 = fig_angvel_mt.add_subplot(122) # point swarm average angular velocity per interval

    try:
        # Total rotation
        df_totrot = result_totrot.loc[result_totrot['ec']=='oug']
        plot_lines(df_totrot.loc[df_totrot['Cat']=='V'], ax_mt_1 , y = 'total_rotation', pal = 'PuRd')
        plot_lines(df_totrot.loc[df_totrot['Cat']=='A'], ax_mt_1 , y = 'total_rotation', pal = 'PuBu')

        # Add custom legends to plot
        first_legend = ax_mt_1.get_legend()
        ax_mt_1.add_artist(first_legend)

        names = ['ventricle', 'atrium']
        pal  = sns.color_palette('PuRd', 1)
        pal += sns.color_palette('PuBu', 1)
        cdict = dict(zip(names, pal))

        custom_lines = []
        custom_legs = []
        for k, v in cdict.items():
            custom_lines.append(Line2D([0],[0], color=v, lw=4))
            custom_legs.append(k)

        # Create another legend
        ax_mt_1.legend(custom_lines, custom_legs, title="chamber", fontsize=4, loc='lower center')

        df_totrot_VA = df_totrot.drop(df_totrot[df_totrot['Cat'] == 'AVC'].index)
        plot_source_data['figure 4 total rotation'] = df_totrot_VA[['ec', 'replicate', 'Cat', 'time_hr', 'total_rotation']]
    except Exception:
        print()
        print("An error occurred. Dataset may be empty.")
        print("Skipping 'figure 4 total rotation oug'")
        print()

    # Swarmplot angular velocity
    try:
        mt = df_test_totrot.loc[df_test_totrot['ec'] == 'oug']
        mt_VA = mt.loc[mt['Cat'].isin(['V', 'A'])]

        order = ["ougV", "ougA"] 
        names = ['V', 'A']
        pal  = sns.color_palette('PuRd', 1)
        pal += sns.color_palette('PuBu', 1)
        cdict = dict(zip(names, pal))

        g = sns.swarmplot(x = 'group', y = 'angvel', data = mt_VA, ax = ax_mt_2, palette = cdict, hue = 'Cat', order=order)
        means = df_test_totrot.groupby(['group'])['angvel'].mean()
        for elem, xtick in zip(order, g.get_xticks()):
            ax_mt_2.plot([xtick-0.1, xtick+0.1], [means[elem], means[elem]], color='gray', linewidth='5')

        plot_source_data['figure 4 angular velocity'] = mt_VA
    except Exception:
        print()
        print("An error occurred. Dataset may be empty.")
        print("Skipping 'figure 4 angular velocity oug'")
        print()

    ####################
    # Figure 4 - Twist #
    ####################
    
    # Figure 4: total twist
    fig_twist = plt.figure(figsize=(16,8)) 
    tw1 = fig_twist.add_subplot(131) # total twist fit over time
    tw2 = fig_twist.add_subplot(132) # point swarm twist velocity used for statistical test
    tw3 = fig_twist.add_subplot(133) # point swarm total twist angle after 9h used for statistical test

    tw1.set_title('Average twist')
    tw1.set_xlabel('Time [hr]')
    tw1.set_ylabel('Total twist [degrees]')
    tw2.set_title("Average twist after 9 hours")
    tw3.set_title("Average twist velocity")

    try:
        # Fit twist narrow intervals
        names = ["wt", "oug"]
        pal  = sns.color_palette('Blues', 1)
        pal += sns.color_palette('Oranges', 1)
        cdict = dict(zip(names, pal))
        g = sns.lineplot(x = 'time_intervals', y = 'twisting angle', hue = 'ec', ci = 'sd',
                            data = fit_twist_nr, palette=cdict,
                            ax = tw1, linewidth = 4, alpha = 0.5)

        plot_source_data['figure 4 twist'] = fit_twist_nr[['ec','replicate', 'time_intervals', 'twisting angle']]

        # Total twisting angle after 9 hours
        order = ["wt", "oug"]

        names = order
        pal  = sns.color_palette('Blues', 1)
        pal += sns.color_palette('Oranges', 1)
        cdict = dict(zip(names, pal))

        g = sns.swarmplot(x = 'ec', y = 'mean total twist', data = df_after9, ax = tw2, palette = cdict, hue = 'ec', order = order)

        means = df_after9.groupby(['ec'], sort=False)['mean total twist'].mean()
        for elem, xtick in zip(order, g.get_xticks()):
            tw2.plot([xtick-0.1, xtick+0.1], [means[elem], means[elem]], color='gray', linewidth='5')
        
        plot_source_data['figure 4 mean total twist after 9 hours'] = df_after9

        # Twist velocity wt vs oug
        order = ["wt", "oug"]

        names = order
        pal  = sns.color_palette('Blues', 1)
        pal += sns.color_palette('Oranges', 1)
        cdict = dict(zip(names, pal))

        g = sns.swarmplot(x = 'ec', y = 'angvel', data = df_test_twist, ax = tw3, palette = cdict, hue = 'ec', order = order)

        means = df_test_twist.groupby(['ec'], sort=False)['angvel'].mean()
        for elem, xtick in zip(order, g.get_xticks()):
            tw3.plot([xtick-0.1, xtick+0.1], [means[elem], means[elem]], color='gray', linewidth='5')

        plot_source_data['figure 4 twist velocity'] = df_test_twist
    except Exception:
        print()
        print("An error occurred. One of the datasets may be empty.")
        print("Skipping (part of) 'figure 4 twist plots'")
        print()
    
    ##############
    # Supplement #
    ##############

    # WILDTYPE
    # Supplementary figure: AVC total rotation + average angular velocity WT
    fig_avcw = plt.figure(figsize=(16,8))
    ax_avcw_1 = fig_avcw.add_subplot(121) # total rotation AVC
    ax_avcw_2 = fig_avcw.add_subplot(122) # point swarm average angular velocity per interval

    try:
        # Total rotation
        df_totrot = result_totrot.loc[result_totrot['ec']=='wt']
        wt_AVC = wt.loc[wt['Cat'].isin(['AVC'])]
        AVCdf = df_totrot[df_totrot['Cat']=='AVC']
        plot_lines(AVCdf, ax_avcw_1 , y = 'total_rotation', pal = 'YlGn')

        plot_source_data['supplement wildtype total rotation'] = AVCdf[['ec', 'replicate', 'Cat', 'total_rotation']]
        
        # Swarmplot angular velocity
        order = ["wtAVC"]
        names = ['AVC']
        pal = sns.color_palette('YlGn', 1)
        cdict = dict(zip(names, pal))
            
        means = wt_AVC.groupby(['group'])['angvel'].mean()
        g = sns.swarmplot(x = 'group', y = 'angvel', data = wt_AVC, ax = ax_avcw_2, palette = cdict, hue = 'Cat')
        for elem, xtick in zip(order, g.get_xticks()):
            ax_avcw_2.plot([xtick-0.1, xtick+0.1], [means[elem], means[elem]], color='gray', linewidth='5')

        plot_source_data['supplement wildtype angular velocity'] = wt_AVC
    except Exception:
        print()
        print("An error occurred. Datasets may be empty.")
        print("Skipping 'supplementary plots wildtype'")
        print()
    
    # MUTANT
    # Supplementary figure: AVC total rotation + average angular velocity MT
    fig_avcm = plt.figure(figsize=(16,8))
    ax_avcm_1 = fig_avcm.add_subplot(121) # total rotation AVC
    ax_avcm_2 = fig_avcm.add_subplot(122) # point swarm average angular velocity per interval

    try:
        # Total rotation
        df_totrot = result_totrot.loc[result_totrot['ec']=='oug']
        mt_AVC = mt.loc[mt['Cat'].isin(['AVC'])]
        AVCdf = df_totrot[df_totrot['Cat']=='AVC']
        plot_lines(AVCdf, ax_avcm_1 , y = 'total_rotation', pal = 'YlGn')

        plot_source_data['supplement mutant total rotation'] = AVCdf[['ec', 'replicate', 'Cat', 'total_rotation']]

        # Swarmplot angular velocity
        order = ["ougAVC"]
        names = ['AVC']
        pal = sns.color_palette('YlGn', 1)
        cdict = dict(zip(names, pal))
            
        means = mt_AVC.groupby(['group'])['angvel'].mean()
        g = sns.swarmplot(x = 'group', y = 'angvel', data = mt_AVC, ax = ax_avcm_2, palette = cdict, hue = 'Cat')
        for elem, xtick in zip(order, g.get_xticks()):
            ax_avcm_2.plot([xtick-0.1, xtick+0.1], [means[elem], means[elem]], color='gray', linewidth='5')

        plot_source_data['supplement mutant angular velocity'] = mt_AVC    
    except Exception:
        print()
        print("An error occurred. Dataset may be empty.")
        print("Skipping 'supplementary plots oug'")
        print()

    #############################
    # Make the plots consistent #
    #############################
    adjustPlotAesthetics([ax_wt_1, ax_mt_1, ax_avcw_1, ax_avcm_1], ylab = 'Total rotation [degrees]')
    adjustPlotAesthetics([ax_avcw_2, ax_avcm_2], ylab = 'Average angular velocity [degree/hr]')
    adjustPlotAesthetics([ax_wt_2, ax_mt_2], ylab = 'Average angular velocity [degree/hr]')
    adjustPlotAesthetics([tw1], ylab = 'Twist [degrees]')
    adjustPlotAesthetics([tw2], ylab = 'Twist [degrees]')
    adjustPlotAesthetics([tw1, tw2])
    adjustPlotAesthetics([tw3], ylab = 'Twist angular velocity [degree/hr]')

    axt = [ax_wt_1, ax_mt_1, tw1]
    for ax in axt:
        ax.set_xlabel('Time [hr]')

    ################################
    # Save figures and source data #
    ################################

    figs = [fig_angvel_wt,
            fig_angvel_mt,
            fig_twist,
            fig_avcw,
            fig_avcm
        ]

    strings = [ 'total_rotation_angvel_wt', 
                'total_rotation_angvel_oug',
                'total_twist',
                'total_rotation_angvel_avc_wt',
                'total_rotation_angvel_avc_mt' 
            ]

    filetype = ['.png', '.pdf']

    # Create an output directory     
    outputpath = Path("Output")
    outputpath.mkdir(parents = True, exist_ok=True)

    for end in filetype:
        for figure, figname in zip(figs, strings):
            string_file = figname + end
            figure.savefig(outputpath / string_file, bbox_inches='tight')
    

    for dataname, dataset in plot_source_data.items():
        string_file = dataname + '.csv'
        dataset.to_csv(outputpath / string_file)

    plt.show()

"""
###########################
###########################
######    OPTIONS    ######
###########################
###########################
"""
 
window = 1.5            # length of binning interval in hours
plot_smoothed = True    # plot the smoothed absolute difference of cumulative angle
exclude_threshold = 0   # only include tracks with >exclude_threshold steps; steps not necessarily consecutive
printmovies = False     # print full movies or only analysis plots 
trimstart = 0           # remove the first trimstart timesteps from the analysis
trimend_h = 10.5        # remove all timesteps that exceed this threshold
avc_cent_threshold = 5  # number of cells that need to be present in avc to do reference centroid calculation
all_cent_threshold = 5  # number of cells that need to be present per category to do centroid calculation

print()
print("options:")
print("track exclusion threshold:", exclude_threshold)
print("--> Excluding all tracks with less than ", exclude_threshold+1, " spots!")
print("length of binning interval in hours:", window)
print("Printing movies:", printmovies)
print("Trimming start:", trimstart)
print("Trimming steps over:", trimend_h, "hours")
print("Threshold number of cells for doing centroid calculation in avc:", avc_cent_threshold)
print("Threshold number of cells for doing centroid calculation for all categories:", all_cent_threshold)
print()

#######################################
#######################################
#######################################
###           CHECK ARGS            ### 
#######################################
#######################################
#######################################

# Command line input example:
# python3 heartbending.py ./excel_set_1/ ./excel_set_2/ "*wt*" "*oug*"

# Check if command line arguments are given
if len(sys.argv) < 1:
    sys.stderr.write("Error. I need at least one path/to/file and a string to match the filename\n")
    sys.exit(1)

filepath_tl = sys.argv[1] # positions of the tracks throughout the timelapse
filepath_se = sys.argv[2] # start and end positions of each cell track as well as heart segment categories
exp_cond    = sys.argv[3:] # Experimental conditions, e.g. "wt" and "oug"


###############################
###         BEGIN           ###
###############################
# Initialize variables for storing input and output for each category
inp_dict = {}
cat_dict = {}
stabilized_data = {}
mapping = {}

# Check if files exist
for ec in exp_cond:
    inp_dict[ec] = []
    cat_dict[ec] = []
    stabilized_data[ec] = []

    rel_p_input_tl = Path.cwd() / filepath_tl # relative path to working directory
    abs_p_input_tl = rel_p_input_tl.resolve() # absolute path

    rel_p_input_se = Path.cwd() / filepath_se # relative path to working directory
    abs_p_input_se = rel_p_input_se.resolve() # absolute path

    print(ec)
    print("Files found:")
    for path_tl in Path(abs_p_input_tl).glob(ec):
        print(path_tl)
        if path_tl.is_file():
            inp_dict[ec].append(path_tl)
        else:
            print()
            print("Warning: File ",path_tl," NOT found\n")
            print()
    print()

    for path_se in Path(abs_p_input_se).glob(ec):
        print(path_se)
        if path_se.is_file():
            cat_dict[ec].append(path_se)
        else:
            print()
            print("Warning: File ",path_se," NOT found\n")
            print()
    print()

# The time-lapse files must be mapped to the file with the categories
for ec in inp_dict.keys():
    mapping[ec] = {}
    
    for file_path in inp_dict[ec]:
        file_id = ec[1:-1] #get rid of asterisks
        string_idx = file_path.name.find(file_id)

        if string_idx != -1:
            end = string_idx + len(file_id) + 1
            final_id = file_path.name[string_idx:end]
            if not final_id[-1].isdigit():
                final_id = final_id[:-1]
            if file_path.name[end].isdigit(): # to catch oug33 and oug44
                final_id += file_path.name[end]
                if file_path.name[end+1].isdigit(): # to catch oug444
                    final_id += file_path.name[end+1]

        for file_se in cat_dict[ec]:
            if final_id in file_se.name:
                string_idx = file_se.name.find(final_id)
                end = string_idx + len(final_id)

                if not file_se.name[end].isdigit():
                    mapping[ec][final_id] = {
                                            "tl": file_path, 
                                            "se": file_se
                                            }

for ec_idx, ec in enumerate(exp_cond):

    if len(inp_dict[ec])==0:
        sys.stderr.write("hearbending.py: Error. No files found for condition " + ec + "\n" + "\n")
        exp_cond.remove(ec) # remove empty condition
        continue
        #sys.exit(1)

    plot_string = ''
    for dataset_number, dataset in enumerate(mapping[ec]):

        timelapse_filepath = mapping[ec][dataset]["tl"]
        startend_filepath  = mapping[ec][dataset]["se"]

        print("Processing files: ", timelapse_filepath)
        print("                + ", startend_filepath)

        plot_string += str(dataset_number) + ' = ' + str(timelapse_filepath.name)
        if dataset_number < len(inp_dict[ec]):
            plot_string += '\n'

        df_tl = pd.read_excel(timelapse_filepath, header=1, sheet_name='Position')
        df_se = pd.read_excel(startend_filepath, header=1, sheet_name='Calculations')

        # Get subset of relevant IDs with tracks longer than t steps
        df_se_df_twist = df_se.copy()
        df_se_df_twist = df_se_df_twist[df_se_df_twist['Track Number of Spots'] > exclude_threshold][['Cat','ID']]

        df_se_df_twist.rename(columns = {'ID':'TrackID'}, inplace = True)
        df_se_df_twist = new_col_DV(df_se_df_twist) # put dorsal/ventral category in new column where applicable
        df_se_df_twist = rm_rename_cats(df_se_df_twist) # drop all unwanted tracks

        # Filter the other dataframe based on subset of relevant IDs
        df = pd.merge(df_tl, df_se_df_twist, how='outer', indicator='ind')

        df = df.loc[df.ind == 'both']
        df = df.drop(['ind'], axis = 1)

        # Timestep length
        df_time = pd.read_excel(startend_filepath, header=1, sheet_name='Time Since Track Start')
        unique_times = sorted(df_time['Time Since Track Start'].unique())
        seconds_per_step = unique_times[1] # 0th element t0 = 0.0, first element t1 = [time elapsed] - t0
        seconds_per_hour = 3600

        df['time_hr'] = (df['Time']-1)*seconds_per_step/seconds_per_hour # subtract 1 to start plots at t = 0

        # Drop the first "trimstart" timesteps from the analysis
        if trimstart > 0:
            print("Skipping the first", trimstart, "timepoints.")
            timelist = df['Time'].sort_values().unique()
            to_keep = timelist[trimstart:]
            df = df.loc[df['Time'].isin(to_keep)]

        # Drop timesteps that exceed threshold second count
        if trimend_h > 0:
            print("Skipping all timesteps that exceed", trimend_h, "hours.")
            timelist = df['time_hr'].sort_values().unique()
            to_keep = [i for i in timelist if i <= trimend_h]
            df = df.loc[df['time_hr'].isin(to_keep)]

        # Drop all time steps with less than all_cent_threshold cells in any category
        if all_cent_threshold > 0:
            droptimes = []
            for grp_name, grp in df.groupby(['Time']):
                vn = grp.loc[grp['Cat']=='V']['TrackID'].nunique()
                an = grp.loc[grp['Cat']=='A']['TrackID'].nunique()
                cn = grp.loc[grp['Cat']=='AVC']['TrackID'].nunique()
                ns = [vn, an, cn]

                if any(x < all_cent_threshold for x in ns):
                    droptimes.append(grp_name)
            
            if droptimes:
                print("Skipping all steps with <=", all_cent_threshold, "cells in ANY category.")
                print("Timesteps dropped:", droptimes)
                df.drop(df[(df['Time'].isin(droptimes))].index, inplace=True)

        ################################
        # Column names used throughout #
        ################################

        tracks = 'TrackID'
        xyz = ['Position X','Position Y','Position Z'] # original coordinates

        # coordinates after first translation: subtract AVC_0
        xyzC = ['xC', 'yC', 'zC']
        avcC = ['avc_x', 'avc_y', 'avc_z']

        # coordinates after second translation: subtract (AVC_t + AVC_0)
        xyzCC = ['xCC', 'yCC', 'zCC'] 
        centCC = ['centx', 'centy', 'centz']

        # rotated coordinates
        xyzrot = ['x_rot', 'y_rot', 'z_rot'] 
        col_ax = ['ax_x', 'ax_y', 'ax_z'] # axis coordinates
        normax = ['ax_x_norm', 'ax_y_norm', 'ax_z_norm'] # unit axis coordinates
        V_i = ['Vix','Viy','Viz'] # displacement vector coordinates

        v1 = ['V_t0_x', 'V_t0_y', 'V_t0_z'] # vector between Ai and the point at time t
        v2 = ['V_t1_x', 'V_t1_y', 'V_t1_z'] # vector between Ai and the point at time t+1

        ###################################################
        ###################################################
        # Find the axis of the data, translate and rotate #
        ###################################################
        ###################################################

        # Sort dataframe by time
        df = df.sort_values(by=['Time'])

        # Get the reference AVC centroid
        # Loop through time and find the first timepoint with at least 5 cells
        avc = df.loc[df['Cat'] == 'AVC']

        mint = min(df['Time'].unique()) # by default the first timepoint
        for time_idx, time_grp in avc.groupby(['Time']):
            if time_grp[tracks].nunique() >= avc_cent_threshold:
                mint = time_idx
                print()
                print(ec)
                print("minimum time with >=",avc_cent_threshold,"cells in AVC:", mint)
                print()
                break
        avc_cents = avc.groupby(['Time'])[xyz].mean().reset_index()
        reference = avc_cents.loc[avc_cents['Time'] == mint][xyz].to_numpy()

        # Subtract the reference from ALL points at all times
        idx = df[xyz].index
        A = df[xyz].to_numpy()
        B = np.tile( reference, (len(df[xyz]),1) )
        C = A - B
        temp = pd.DataFrame(C, columns = xyzC, index = idx)
        df[xyzC] = temp
       
        # Redo the AVC centCC on the corrected coordinates
        
        avc = df.loc[df['Cat'] == 'AVC']
        avc_cents = avc.groupby(['Time'])[xyzC].mean().reset_index()
        avc_cents.rename(columns = {xyzC[0]:avcC[0], xyzC[1]:avcC[1], xyzC[2]:avcC[2]}, inplace = True)

        df = pd.merge(df, avc_cents, left_on='Time', right_on='Time')
        
        # Subtract the AVC centroid at time t from all points in time t
        idx = df[xyzC].index
        A = df[xyzC].to_numpy()
        B = df[avcC].to_numpy()
        C = A - B
        temp = pd.DataFrame(C, columns = xyzCC, index = idx)
        df[xyzCC] = temp
        
        # Redo all the centCC on corrected coordinates and add them to the dataframe
        cents = df.groupby(['Time', 'Cat'])[xyzCC].mean().reset_index()
        cents.rename(columns = {xyzCC[0]:centCC[0], xyzCC[1]:centCC[1], xyzCC[2]:centCC[2]}, inplace = True)

        # Calculate the axes for every timepoint
        df_v = cents.loc[cents['Cat'] == 'V']  
        df_a = cents.loc[cents['Cat'] == 'A']  
        df_c = cents.loc[cents['Cat'] == 'AVC']

        ax_vc = df_v[centCC].to_numpy() - df_c[centCC].to_numpy() # V-C(t) = centroid of V(t)   - centroid of AVC(t)
        ax_ca = df_c[centCC].to_numpy() - df_a[centCC].to_numpy() # A-C(t) = centroid of AVC(t) - centroid of A(t)

        ################################
        # VENTRICLE AXIS ROTATION CALC #
        ################################
        ax_vc_norm = np.linalg.norm(ax_vc, axis=1)
        
        # df_vc is ventricle axes grouped by time
        df_vc = pd.DataFrame(ax_vc, columns = col_ax, index = df_v.index)
        df_norm = pd.DataFrame(ax_vc_norm, columns = ['norm'], index = df_v.index)
        
        df_vc['norm'] = df_norm
        df_vc[normax[0]] = df_vc[col_ax[0]]/df_vc['norm']
        df_vc[normax[1]] = df_vc[col_ax[1]]/df_vc['norm']
        df_vc[normax[2]] = df_vc[col_ax[2]]/df_vc['norm']

        # apply the rotation matrix to each row of the normed axis vector wrt to initial axis
        ar = df_vc[normax].to_numpy()

        v_init = ar[0] # initial axis
        
        ar_rotmat = [ RotationMatrix(v_init, v) for v in ar ] # list of rotation matrices

        index_V = df.loc[df['Cat'] == 'V'].index
        df_twist_df = df.loc[df['Cat'] == 'V']
        tot_rot_trnsl_points = []
        
        # Get all points of a given timepoint, apply the corresponding rotation matrix
        act_idx = 0
        for time_idx, time_group in df_twist_df.groupby(['Time']):
            translated_points = time_group[xyzCC].to_numpy()
            rot_mat = ar_rotmat[act_idx]
            rot_trnsl_points = np.array( [ rot_mat.dot(pnt) for pnt in translated_points ])
            
            # build a big array with the x,y,z coordinates of the points after rotation
            try:
                tot_rot_trnsl_points = np.append(tot_rot_trnsl_points, rot_trnsl_points, axis=0)
            except ValueError:
                tot_rot_trnsl_points = np.copy(rot_trnsl_points)

            act_idx += 1

        # Transform result back into a dataframe with the same index as before
        df_rot_v = pd.DataFrame(tot_rot_trnsl_points, 
                                columns = xyzrot, 
                                index = index_V)

        ###################################
        # AVC around V AXIS ROTATION CALC #
        ###################################

        # Take the ventricle axis
        ar = df_vc[normax].to_numpy()
        v_init = ar[0] # initial axis
        ar_rotmat = [ RotationMatrix(v_init, v) for v in ar ] # list of rotation matrices

        index_AVC = df.loc[df['Cat'] == 'AVC'].index
        df_twist_df = df.loc[df['Cat'] == 'AVC']
        tot_rot_trnsl_points = []
        
        # Get all points of a given timepoint, apply the corresponding rotation matrix
        act_idx = 0
        for time_idx, time_group in df_twist_df.groupby(['Time']):
            translated_points = time_group[xyzCC].to_numpy()
            rot_mat = ar_rotmat[act_idx]
            rot_trnsl_points = np.array( [ rot_mat.dot(pnt) for pnt in translated_points ])
            
            # build a big array with the x,y,z coordinates of the points after rotation
            try:
                tot_rot_trnsl_points = np.append(tot_rot_trnsl_points, rot_trnsl_points, axis=0)
            except ValueError:
                tot_rot_trnsl_points = np.copy(rot_trnsl_points)

            act_idx += 1

        # Transform result back into a dataframe with the same index as before
        df_avc = pd.DataFrame(tot_rot_trnsl_points, 
                              columns = xyzrot, 
                              index = index_AVC)

        #############################
        # ATRIUM AXIS ROTATION CALC #
        #############################

        ax_ca_norm = np.linalg.norm(ax_ca, axis=1)
        
        # df_ca is atrium axes grouped by time
        df_ca = pd.DataFrame(ax_ca, columns = col_ax, index = df_a.index)
        df_norm = pd.DataFrame(ax_ca_norm, columns = ['norm'], index = df_a.index)
        
        df_ca['norm'] = df_norm
        df_ca[normax[0]] = df_ca[col_ax[0]]/df_ca['norm']
        df_ca[normax[1]] = df_ca[col_ax[1]]/df_ca['norm']
        df_ca[normax[2]] = df_ca[col_ax[2]]/df_ca['norm']

        ar = df_ca[normax].to_numpy()

        a_init = ar[0]
        
        ar_rotmat = [ RotationMatrix(a_init, v) for v in ar ]

        index_A = df.loc[df['Cat'] == 'A'].index
        df_twist_df = df.loc[df['Cat'] == 'A']
        tot_rot_trnsl_points = []
        
        act_idx = 0
        for time_idx, time_group in df_twist_df.groupby(['Time']):
            translated_points = time_group[xyzCC].to_numpy()
            rot_mat = ar_rotmat[act_idx] 
            rot_trnsl_points = np.array( [ rot_mat.dot(pnt) for pnt in translated_points ])
            
            try:
                tot_rot_trnsl_points = np.append(tot_rot_trnsl_points, rot_trnsl_points, axis=0)
            except ValueError:
                tot_rot_trnsl_points = np.copy(rot_trnsl_points)
            
            act_idx += 1
            
        df_rot_a = pd.DataFrame(tot_rot_trnsl_points, 
                                columns = xyzrot, 
                                index = index_A)

        df_ax = pd.concat([df_rot_v, df_rot_a, df_avc])

        df = df.join(df_ax)

        ###########################################################
        ###########################################################
        # Find the rotation around the (fixed) axis for each cell #
        ###########################################################
        ###########################################################

        cats = ['V', 'A', 'AVC'] # categories
        axes = [v_init, a_init, v_init]
        df_catlist = [] # list of dataframes with analysis results

        t0 = df.loc[df['Time'] == mint]
        A0 = [t0[avcC[0]].unique(), t0[avcC[1]].unique(), t0[avcC[2]].unique()]
        A0 = np.array([ x[0] for x in A0 ])

        # Loop through each category
        for cidx, cat in enumerate(cats):
            df_cat = df.loc[df['Cat'] == cat]
            V_a = axes[cidx]  # axis vector
            V_a_norm = np.linalg.norm(V_a) # calculate norm and check it's > 0
            if V_a_norm<0.000001:
                print("Error: axis length = 0")
                sys.exit(1)
            V_a_normed = [v/V_a_norm for v in V_a] # unit vector along axis

            # Group by track ID (individual cells) and get the difference vector between timepoints
            df_cat = df_cat.sort_values(by=['Time']) # Sort dataframe by time   
            delta_T = df_cat.groupby(tracks)[xyzrot].diff()
            delta_T.rename(columns = {xyzrot[0]:V_i[0], 
                                      xyzrot[1]:V_i[1], 
                                      xyzrot[2]:V_i[2]}, 
                           inplace = True) 
            df_cat = df_cat.join(delta_T) # Join the dataframes

            # Get the dot product of V_i and V_a_normed
            # Multiply corresponding elements
            temp_x = df_cat[V_i[0]] * V_a_normed[0]
            temp_y = df_cat[V_i[1]] * V_a_normed[1]
            temp_z = df_cat[V_i[2]] * V_a_normed[2]
            # Sum up and add to dataframe
            magnitude_V_parallel = temp_x.add(temp_y, fill_value=0).add(temp_z, fill_value=0)

            # Get the vector parallel to the axis
            V_parallel = [ magnitude_V_parallel*v for v in V_a_normed ]
            df_cat['V_parallel_x'] = V_parallel[0]
            df_cat['V_parallel_y'] = V_parallel[1]
            df_cat['V_parallel_z'] = V_parallel[2]

            # Get the vector perpendicular to the axis
            df_cat['V_perpend_x'] = df_cat[V_i[0]] - df_cat['V_parallel_x']
            df_cat['V_perpend_y'] = df_cat[V_i[1]] - df_cat['V_parallel_y']
            df_cat['V_perpend_z'] = df_cat[V_i[2]] - df_cat['V_parallel_z']
            
            # Find Ai = the point on the axis that lies in plane with the point of reference at time t
            # First get the vector from Pti to A0
            df_cat['V_Pti_A0_x'] = df_cat[xyzrot[0]] - A0[0]
            df_cat['V_Pti_A0_y'] = df_cat[xyzrot[1]] - A0[1]
            df_cat['V_Pti_A0_z'] = df_cat[xyzrot[2]] - A0[2]

            # Then calculate the multiplier for the unit vector along the axis with dot products
            dot_V_a = np.dot(V_a, V_a)
            dot_V_Pti_A0_Va  = df_cat['V_Pti_A0_x']*V_a[0] + df_cat['V_Pti_A0_y']*V_a[1] + df_cat['V_Pti_A0_z']*V_a[2]
            df_cat['multiplier'] = dot_V_Pti_A0_Va/dot_V_a

            # Get Ai from the parametric line along the axis unit vector
            df_cat['Aix'] = A0[0] + V_a[0] * df_cat.groupby(tracks)['multiplier'].shift(1)
            df_cat['Aiy'] = A0[1] + V_a[1] * df_cat.groupby(tracks)['multiplier'].shift(1)
            df_cat['Aiz'] = A0[2] + V_a[2] * df_cat.groupby(tracks)['multiplier'].shift(1)

            # Find P_tp1p = the point at time t+1 projected on the plane
            df_cat['P_tp1p_x'] = df_cat[xyzrot[0]] + df_cat['V_parallel_x']
            df_cat['P_tp1p_y'] = df_cat[xyzrot[1]] + df_cat['V_parallel_y']
            df_cat['P_tp1p_z'] = df_cat[xyzrot[2]] + df_cat['V_parallel_z']

            # Get the vector between Ai and the point at time t
            df_cat['V_t0_x'] = df_cat.groupby(tracks)[xyzrot[0]].shift(1) - df_cat['Aix']
            df_cat['V_t0_y'] = df_cat.groupby(tracks)[xyzrot[1]].shift(1) - df_cat['Aiy']
            df_cat['V_t0_z'] = df_cat.groupby(tracks)[xyzrot[2]].shift(1) - df_cat['Aiz']

            # Get the vector between Ai and the point at time t+1
            df_cat['V_t1_x'] = df_cat['P_tp1p_x'] - df_cat['Aix']
            df_cat['V_t1_y'] = df_cat['P_tp1p_y'] - df_cat['Aiy']
            df_cat['V_t1_z'] = df_cat['P_tp1p_z'] - df_cat['Aiz']

            # Get the norm and unit vector
            v1_norm = df_cat[v1].apply(lambda values: sum([v**2 for v in values]), axis=1).pow(1./2)
            v2_norm = df_cat[v2].apply(lambda values: sum([v**2 for v in values]), axis=1).pow(1./2)

            v1_unit = []
            v2_unit = []
            for i in range(3):
                v1_unit.append(df_cat[v1[i]]/v1_norm)
                v2_unit.append(df_cat[v2[i]]/v2_norm)

            # Get the angle between v1 and v2 with respect to plane defined by norm
            x1=v1_unit[0];    y1=v1_unit[1];    z1=v1_unit[2]
            x2=v2_unit[0];    y2=v2_unit[1];    z2=v2_unit[2]
            xn=V_a_normed[0]; yn=V_a_normed[1]; zn=V_a_normed[2]
            
            dot = x1*x2 + y1*y2 + z1*z2
            det = x1*y2*zn + x2*yn*z1 + xn*y1*z2 - z1*y2*xn - z2*yn*x1 - zn*y1*x2
            df_cat['angvel'] = np.arctan2(det, dot) * 180/np.pi

            # Replicate ID
            df_cat['replicate'] = dataset
            df_catlist.append(df_cat)

        df_new = pd.concat(df_catlist)
        stabilized_data[ec].append(df_new)

# Concatenate results to one dataframe per experimental condition
for enum, ec in enumerate(exp_cond):
    stabilized_data[ec] = pd.concat(stabilized_data[ec])

###################
###################
# Quantification  #
###################
###################

################################################################
# Calculate cumulative rotation, twist, and angular velocities #
################################################################

result_totrot = [] # total rotation
fit_totrot = [] # fit on total rotation

result_twist = [] # total twist
fit_twist = [] # fit on twist
fit_twist_nr = [] # fit on twist data in narrow interval

gs1 = ['replicate', 'Cat', 'time_hr']
gs2 = ['replicate', 'Cat']

for enum, ec in enumerate(exp_cond):
    print("experimental condition:", ec)

    # Get the average angular velocity at all timepoints by replicate and category
    df_totrot = stabilized_data[ec].groupby(gs1).mean().reset_index().copy()
    df_totrot['angvel'] = df_totrot['angvel'].fillna(0)
    
    # Get the number of tracks at each timepoint by replicate and category
    df_totrot['track_counts'] = stabilized_data[ec].groupby(gs1).count().reset_index()[tracks]
        
    # Get the cumulative sum of the angular velocity (rotation) by replicate and category
    df_totrot['total_rotation'] = df_totrot.groupby(gs2).transform(lambda g: g.cumsum())['angvel']
    df_totrot['total_rotation'] = df_totrot['total_rotation'].fillna(0)

    # Get the average angular velocity over a time interval using a linear fit on the cumulative sum (rotation)
    fit_totrot_single = df_totrot.groupby(['replicate', 'Cat']).apply(lin_fit_function, interval_hours = window).reset_index()
    
    result_totrot.append(df_totrot)
    fit_totrot.append(fit_totrot_single)
    
    # Calculate the twist: difference between total rotation of ventricle and atrium by time and replicate
    df_twist = df_totrot[['Cat', 'replicate', 'total_rotation', 'time_hr']].copy()
    df_twist.drop(df_twist[df_twist['Cat'] == 'AVC'].index, inplace=True) # Remove AVC
    
    df_twist = df_twist.sort_values(by=['replicate', 'time_hr']) # Sort dataframe by time
    twist = df_twist.groupby(['time_hr', 'replicate'])['total_rotation'].diff().abs()
    namestring = 'twist'
    df_twist[namestring] = twist
    df_twist = df_twist.dropna()[['time_hr', 'replicate', namestring]].reset_index()
    try:
        # Fitting slope over large interval
        fit_twist_single = df_twist.groupby(['replicate']).apply(lin_fit_function, ydata = namestring, interval_hours = window).reset_index()

        # Fitting slope over narrow interval to get the mean + sd later
        fit_twist_single_nr = df_twist.groupby(['replicate']).apply(lin_fit_function, ydata = namestring, interval_hours = 0.5).reset_index()
        
        fit_twist_nr.append(fit_twist_single_nr)
        result_twist.append(df_twist)
        fit_twist.append(fit_twist_single)
    except ValueError:
        print("Error: Check if the dataframe is empty!")
    
# Reshaping data to do statistical tests
result_twist = pd.concat(result_twist) # total twist
result_totrot = pd.concat(result_totrot) # total rotation
fit_twist = pd.concat(fit_twist) # fit on twist
fit_totrot = pd.concat(fit_totrot) # fit on total rotation
fit_twist_nr = pd.concat(fit_twist_nr) # fit on twist data in narrow interval
fit_twist_nr.rename(columns = {'points':'twisting angle'}, inplace = True)

datalists = [result_twist, result_totrot, fit_twist, fit_totrot, fit_twist_nr]

# Add back the experimental condition as a column
for dataset in datalists:
    new_col = []
    for rep in dataset['replicate']:
        for ec in exp_cond:
            ec_sub = ec[1:-1]
            if ec_sub in rep:
                new_col.append(ec_sub)

    dataset['ec'] = new_col

datalists = createDataSubsets(datalists)

#################################
# Get statistics and make plots #
#################################

plotStartEndVectors(stabilized_data)

doStats(datalists)
makePlots(datalists)

# Plot the result of translation + rotation side-by-side with original data
if printmovies:
   plot_frames(stabilized_data)

#### END ####
