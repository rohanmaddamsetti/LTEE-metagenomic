import sys
import numpy
from math import log10,ceil
import pylab
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import parse_file
import timecourse_utils
from scipy.stats import fisher_exact
import statsmodels.api as sm
import mutation_spectrum_utils
from scipy.stats import beta
from numpy.random import shuffle
import figure_utils
import stats_utils

theory_times = numpy.arange(0,125)*500

#metapopulations = ['nonmutator','mutator']
metapopulations = ['mutator','nonmutator']
metapopulation_labels = {'nonmutator': 'Nonmutators', 'mutator':'Mutators'}
metapopulation_populations = {'nonmutator': parse_file.complete_nonmutator_lines, 'mutator': parse_file.mutator_lines}

colors = {'nonmutator': figure_utils.nonmutator_group_color, 'mutator': figure_utils.mutator_group_color}


##############################################################################
#
# Set up figures
#
##############################################################################

mpl.rcParams['font.size'] = 5.0
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'
mpl.rcParams['axes.labelpad'] = 2
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['mathtext.it'] = 'serif:italic'
mpl.rcParams['mathtext.bf'] = 'serif:bold'
mpl.rcParams['mathtext.fontset'] = 'custom'



pfix_axes = {}
parallelism_axes = {}

##############################################################################
#
# First set up Fig. 5 (rates of fitness and mutation accumulation with time)
#
##############################################################################

#fig5 = plt.figure(figsize=(6.8, 3.8))
#fig5 = plt.figure(figsize=(4.72,2.6))
fig5 = plt.figure(figsize=(5.35,3.1))

outer_grid = gridspec.GridSpec(2,1, height_ratios=[2.1,0.90], hspace=0.3) 

inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,width_ratios=[2.9,1.1], height_ratios=[1,1],wspace=0.20,hspace=0.15,subplot_spec=outer_grid[0])

lower_grid = gridspec.GridSpecFromSubplotSpec(1,3,width_ratios=[2.3,2.3,2.1],wspace=0.35,subplot_spec=outer_grid[1])


###########
#
# Nonmutator variant types through time
#
###########
nonmutator_axis = plt.Subplot(fig5, inner_grid[0,0])
fig5.add_subplot(nonmutator_axis)

nonmutator_axis.spines['top'].set_visible(False)
nonmutator_axis.spines['right'].set_visible(False)
nonmutator_axis.get_xaxis().tick_bottom()
nonmutator_axis.get_yaxis().tick_left()
#nonmutator_axis.get_xaxis().set_tick_params(direction='out')
#nonmutator_axis.set_title('       Nonmutators', y=0.9,fontsize=5,loc='left')
#nonmutator_axis.set_xlabel('Generation, $t$')

nonmutator_axis.set_ylim([0,2.0])

nonmutator_axis.set_yticks([0,0.5,1.0,1.5,2.0])
###########
#
# Mutator variant types through time
#
###########
mutator_axis = plt.Subplot(fig5, inner_grid[1,0])
fig5.add_subplot(mutator_axis)
mutator_axis.set_ylabel('Total mutations detected ($\\times 10^3$)',y=1.1,labelpad=3)
mutator_axis.set_xlabel('Generation, $t$',labelpad=2)

mutator_axis.spines['top'].set_visible(False)
mutator_axis.spines['right'].set_visible(False)
mutator_axis.get_xaxis().tick_bottom()
mutator_axis.get_yaxis().tick_left()
#mutator_axis.set_title('    Mutators',y=0.82,fontsize=5,loc='left')

#mutator_axis.get_yaxis().set_tick_params(direction='out')
#mutator_axis.get_xaxis().set_tick_params(direction='out')

metapopulation_axes = {'nonmutator':nonmutator_axis, 'mutator':mutator_axis}

###########
#
# Nonmutator multiplicity distribution
#
###########

nonmutator_parallelism_axis = plt.Subplot(fig5, inner_grid[0,1])

fig5.add_subplot(nonmutator_parallelism_axis)

nonmutator_parallelism_axis.spines['top'].set_visible(False)
nonmutator_parallelism_axis.spines['right'].set_visible(False)
nonmutator_parallelism_axis.get_xaxis().tick_bottom()
nonmutator_parallelism_axis.get_yaxis().tick_left()

###########
#
# Mutator multiplicity distribution
#
###########

mutator_parallelism_axis = plt.Subplot(fig5, inner_grid[1,1])
fig5.add_subplot(mutator_parallelism_axis)

mutator_parallelism_axis.set_ylabel(r'Fraction mutations $\geq m$',y=1.1,labelpad=2)
mutator_parallelism_axis.set_xlabel('Gene multiplicity, $m$',labelpad=2)
#nonmutator_parallelism_axis.set_xlabel('Gene multiplicity, $m$')

mutator_parallelism_axis.spines['top'].set_visible(False)
mutator_parallelism_axis.spines['right'].set_visible(False)
mutator_parallelism_axis.get_xaxis().tick_bottom()
mutator_parallelism_axis.get_yaxis().tick_left()

parallelism_axes[True] = {'nonmutator': nonmutator_parallelism_axis, 'mutator': mutator_parallelism_axis}

######################
# Variant type vs time
######################

time_axis = plt.Subplot(fig5, lower_grid[0])
fig5.add_subplot(time_axis)

time_axis.spines['top'].set_visible(False)
time_axis.spines['right'].set_visible(False)
time_axis.get_xaxis().tick_bottom()
time_axis.get_yaxis().tick_left()

time_axis.set_ylabel(r'Fraction mutations $\geq t$',labelpad=3)
time_axis.set_xlabel('Appearance time, $t$')
time_axis.set_xticks(figure_utils.time_xticks)
time_axis.set_xticklabels(figure_utils.time_xticklabels)
time_axis.set_xlim([0,60000])

###########
#
# Fixation probability vs variant type
#
###########

vartype_pfix_axis = plt.Subplot(fig5, lower_grid[1])
fig5.add_subplot(vartype_pfix_axis)

vartype_pfix_axis.spines['top'].set_visible(False)
vartype_pfix_axis.spines['right'].set_visible(False)
vartype_pfix_axis.get_xaxis().tick_bottom()
vartype_pfix_axis.get_yaxis().tick_left()

vartype_pfix_axis.set_ylabel('Pr[fixed|detected]')
vartype_pfix_axis.set_ylim([0,1.05])
vartype_pfix_axis.set_xlim([-0.5,len(parse_file.var_types)-0.5])
vartype_pfix_axis.set_xticks(numpy.arange(0,len(parse_file.var_types)))

###########
#
# Nonmutator Fixation probability vs multiplicity 
#
###########

nonmutator_pfix_axis = plt.Subplot(fig5, lower_grid[2])
fig5.add_subplot(nonmutator_pfix_axis)

nonmutator_pfix_axis.spines['top'].set_visible(False)
nonmutator_pfix_axis.spines['right'].set_visible(False)
nonmutator_pfix_axis.get_xaxis().tick_bottom()
nonmutator_pfix_axis.get_yaxis().tick_left()
nonmutator_pfix_axis.set_xlabel('Gene multiplicity, $m$')
nonmutator_pfix_axis.set_ylabel('Pr[fixed|detected]')
nonmutator_pfix_axis.set_ylim([0,1.05])

pfix_axes[True] = {'nonmutator': nonmutator_pfix_axis, 'mutator': nonmutator_pfix_axis}

##############################################################################
#
# Set up Fig. S10 (multiplicity plots without svs)
#
##############################################################################

nosv_fig = plt.figure(figsize=(7, 1.4))

nosv_outer_grid  = gridspec.GridSpec(1, 3, wspace=0.3)

###########
#
# Nonmutator multiplicity distribution (nosv)
#
###########

nonmutator_parallelism_axis_nosv = plt.Subplot(nosv_fig, nosv_outer_grid[0])

nosv_fig.add_subplot(nonmutator_parallelism_axis_nosv)

nonmutator_parallelism_axis_nosv.spines['top'].set_visible(False)
nonmutator_parallelism_axis_nosv.spines['right'].set_visible(False)
nonmutator_parallelism_axis_nosv.get_xaxis().tick_bottom()
nonmutator_parallelism_axis_nosv.get_yaxis().tick_left()

nonmutator_parallelism_axis_nosv.set_ylabel(r'Total mutations $\geq m$')
nonmutator_parallelism_axis_nosv.set_xlim([1,1e02])
nonmutator_parallelism_axis_nosv.set_xlabel('Gene multiplicity, $m$')

###########
#
# Mutator multiplicity distribution (nosv)
#
###########

mutator_parallelism_axis_nosv = plt.Subplot(nosv_fig, nosv_outer_grid[1])
nosv_fig.add_subplot(mutator_parallelism_axis_nosv)

mutator_parallelism_axis_nosv.spines['top'].set_visible(False)
mutator_parallelism_axis_nosv.spines['right'].set_visible(False)
mutator_parallelism_axis_nosv.get_xaxis().tick_bottom()
mutator_parallelism_axis_nosv.get_yaxis().tick_left()
mutator_parallelism_axis_nosv.set_xlim([1,1e02])
mutator_parallelism_axis_nosv.set_ylabel(r'Total mutations $\geq m$')
mutator_parallelism_axis_nosv.set_xlabel('Gene multiplicity, $m$')

parallelism_axes[False] = {'nonmutator': nonmutator_parallelism_axis_nosv, 'mutator': mutator_parallelism_axis_nosv}

###########
#
# Nonmutator Fixation probability vs multiplicity (nosv)
#
###########

nonmutator_pfix_axis_nosv = plt.Subplot(nosv_fig, nosv_outer_grid[2])
nosv_fig.add_subplot(nonmutator_pfix_axis_nosv)

nonmutator_pfix_axis_nosv.spines['top'].set_visible(False)
nonmutator_pfix_axis_nosv.spines['right'].set_visible(False)
nonmutator_pfix_axis_nosv.get_xaxis().tick_bottom()
nonmutator_pfix_axis_nosv.get_yaxis().tick_left()
nonmutator_pfix_axis_nosv.set_ylabel('Pr[fixed|detected]')
nonmutator_pfix_axis_nosv.set_xlabel('Gene multiplicity, $m$')
nonmutator_pfix_axis_nosv.set_ylim([0,1.05])
nonmutator_pfix_axis_nosv.set_xlim([1,1e02])

pfix_axes[False] = {'nonmutator': nonmutator_pfix_axis_nosv, 'mutator': nonmutator_pfix_axis_nosv}


#############################
#
# Now do actual calculations and plot figures
#
#############################
   

Lsyn, Lnon, substitution_specific_synonymous_fraction = parse_file.calculate_synonymous_nonsynonymous_target_sizes()

for metapopulation in metapopulations:

    axis = metapopulation_axes[metapopulation]
    color = colors[metapopulation]
    label = metapopulation_labels[metapopulation]
    populations = metapopulation_populations[metapopulation]
    
    if metapopulation == 'nonmutator':
        metapopulation_offset = -0.35
    else:
        metapopulation_offset = 0

    #############################
    #
    # Variant types through time
    #
    #############################

    var_type_trajectories = {var_type: numpy.zeros_like(theory_times)*1.0 for var_type in parse_file.var_types}
    var_type_totals = {var_type: 0 for var_type in parse_file.var_types}
    var_type_fixeds = {var_type: 0 for var_type in parse_file.var_types}

    for population in populations:
    
        sys.stderr.write("Processing %s...\t" % parse_file.get_pretty_name(population))

        # calculate mutation trajectories
        # load mutations
        mutations, depth_tuple = parse_file.parse_annotated_timecourse(population)
    
        population_avg_depth_times, population_avg_depths, clone_avg_depth_times, clone_avg_depths = depth_tuple
    
        dummy_times,fmajors,fminors,haplotype_trajectories = parse_file.parse_haplotype_timecourse(population)
        state_times, state_trajectories = parse_file.parse_well_mixed_state_timecourse(population)
    
        num_processed_mutations = 0
        for mutation_idx in range(0,len(mutations)):
 
            num_processed_mutations+=1 
            
            position, gene_name, allele, var_type, test_statistic, pvalue, cutoff_idx, depth_fold_change, depth_change_pvalue, times, alts, depths, clone_times, clone_alts, clone_depths = mutations[mutation_idx] 
            Ls = haplotype_trajectories[mutation_idx]
            state_Ls = state_trajectories[mutation_idx]
        
            good_idxs, filtered_alts, filtered_depths = timecourse_utils.mask_timepoints(times, alts, depths, var_type, cutoff_idx, depth_fold_change, depth_change_pvalue)
    
            freqs = timecourse_utils.estimate_frequencies(filtered_alts, filtered_depths)
        
            masked_times = times[good_idxs]
            masked_freqs = freqs[good_idxs]
            masked_state_Ls = state_Ls[good_idxs]
            masked_Ls = Ls[good_idxs]
            
            t = timecourse_utils.calculate_appearance_time(masked_times, masked_freqs, masked_state_Ls, masked_Ls)
            #fixed_weight = timecourse_utils.calculate_fixed_weight(masked_state_Ls[-1], masked_freqs[-1])
            fixed_weight = timecourse_utils.calculate_clade_fixed_weight(masked_Ls[-1], masked_freqs[-1])
        
            var_type_trajectories[var_type][theory_times>=t] += 1.0
            var_type_totals[var_type] += 1.0
            var_type_fixeds[var_type] += fixed_weight
        
        sys.stderr.write("added %d mutations!\n" % num_processed_mutations)

    
    var_type_trajectories = [var_type_trajectories[var_type] for var_type in parse_file.var_types]
    var_type_totals = numpy.array([var_type_totals[var_type] for var_type in parse_file.var_types])*1.0
    var_type_fixeds = numpy.array([var_type_fixeds[var_type] for var_type in parse_file.var_types])*1.0
    
    survival_probabilities = var_type_fixeds/var_type_totals
    pfix_upper = numpy.array([beta.ppf(0.84, var_type_fixeds[i]+1, var_type_totals[i]-var_type_fixeds[i]+1, loc=0, scale=1) for i in range(0,len(var_type_totals))])
    pfix_lower = numpy.array([beta.ppf(0.16, var_type_fixeds[i]+1, var_type_totals[i]-var_type_fixeds[i]+1, loc=0, scale=1) for i in range(0,len(var_type_totals))])
        
    vartype_pfix_axis.bar(numpy.arange(0,len(parse_file.var_types))+metapopulation_offset, survival_probabilities+0.05,width=0.35, bottom=-0.05, linewidth=0, facecolor=color,alpha=0.5)

    for i in range(0,len(parse_file.var_types)):
        if i==0:
            vartype_pfix_axis.bar([-10],[1],label=label,linewidth=0, facecolor=color,alpha=0.5)            
        vartype_pfix_axis.plot([i+metapopulation_offset+0.15]*2, [pfix_lower[i],pfix_upper[i]],color=color,alpha=0.5,linewidth=0.5)
        
        #vartype_pfix_axis.text(i+metapopulation_offset+0.15, pfix_upper[i]+0.05, "%d" % var_type_totals[i],rotation='vertical',verticalalignment='bottom',horizontalalignment='center',fontsize=5)

    vartype_xticklabels = ["%s" % figure_utils.get_var_type_short_name(var_type) for var_type,n in zip(parse_file.var_types,var_type_totals)]    
    vartype_pfix_axis.set_xticklabels(vartype_xticklabels,rotation='vertical')

    
    syn_total = var_type_totals[0]
    syn_fixed = var_type_fixeds[0]
    syn_pfix = syn_fixed*1.0/syn_total
    
    non_total = var_type_totals[1]+var_type_totals[2]
    non_fixed = var_type_fixeds[1]+var_type_fixeds[2]
    non_pfix = non_fixed*1.0/non_total
    
    oddsratio, pvalue = fisher_exact([[non_total, non_fixed], [syn_total, syn_fixed]])
    
    dnds_total = (non_total/syn_total)/(Lnon/Lsyn)
    dnds_fixed = (non_fixed/syn_fixed)/(Lnon/Lsyn)
    
    sys.stdout.write("Synonymous: Target=%g, Total=%0.1f, Fixed=%0.1f, pfix=%0.2f\n" % (Lsyn, syn_total, syn_fixed, syn_pfix))
    sys.stdout.write("Nonsynonymous: Target=%g, Total=%0.1f, Fixed=%0.1f, pfix=%0.2f\n" % (Lnon, non_total, non_fixed, non_pfix))
    sys.stdout.write("Total dN/dS=%0.2f\n" % (dnds_total))
    sys.stdout.write("Fixed dN/dS=%0.2f\n" % (dnds_fixed))
    sys.stdout.write("Pfix pvalue=%g\n" % pvalue)
    
    top = numpy.zeros_like(theory_times)*1.0
    bottom = numpy.zeros_like(theory_times)*1.0

    bottom_fixed = 0
    top_fixed = 0
    fixed_scale_factor = sum(var_type_totals)*1.0/sum(var_type_fixeds)

    for i in range(0,len(parse_file.var_types)):

        bottom = top*1.0
        top += var_type_trajectories[i]
    
        bottom_fixed = top_fixed*1.0
        top_fixed+=var_type_fixeds[i]
    
        color = figure_utils.get_var_type_color(parse_file.var_types[i])
    
        axis.fill_between(theory_times[:121], bottom[:121]/1000.0, top[:121]/1000.0, color=color)
        axis.fill_between([theory_times[120],theory_times[120]+2000],[bottom[120]/1000.0,bottom[120]/1000.0],[top[120]/1000.0,top[120]/1000.0],color=color)
        axis.bar([-1],[1],facecolor=color,edgecolor='none',label=parse_file.var_types[i])
        axis.fill_between([66000,69000], [bottom_fixed*fixed_scale_factor/1000.0,bottom_fixed*fixed_scale_factor/1000.0],[top_fixed*fixed_scale_factor/1000.0,top_fixed*fixed_scale_factor/1000.0], color=color)

    axis.set_xlim([0,70000])


    ###################################################################
    #
    # Gene parallelism analysis
    #
    ###################################################################

    for include_svs in [True, False]:

        if include_svs:
            convergence_matrix_filename = parse_file.data_directory + "gene_convergence_matrix.txt"
        else:
            convergence_matrix_filename = parse_file.data_directory + "gene_convergence_matrix_nosvs.txt"
            
        convergence_matrix = parse_file.parse_convergence_matrix(convergence_matrix_filename)

        # Calculate median appearance time
        pooled_appearance_times = []
        for gene_name in list(convergence_matrix.keys()):
            for population in metapopulation_populations[metapopulation]:
                for t,L,Lclade,f in convergence_matrix[gene_name]['mutations'][population]:
                    pooled_appearance_times.append(t)
                    
        tstar = numpy.median(pooled_appearance_times)
                    
        sys.stdout.write("Median appearance time = %g\n" % tstar)

        gene_parallelism_statistics = mutation_spectrum_utils.calculate_parallelism_statistics(convergence_matrix,allowed_populations=metapopulation_populations[metapopulation],Lmin=100)
        
        G, pvalue = mutation_spectrum_utils.calculate_total_parallelism(gene_parallelism_statistics)

        sys.stdout.write("Total parallelism = %g (p=%g)\n" % (G,pvalue))
        
        predictors = []
        early_predictors = []
        late_predictors = []
        responses = []
    
        gene_hits = []
        gene_fixeds = []
        gene_predictors = []
        gene_pfixs = []
    
        Ls = []
    
    
        for gene_name in list(convergence_matrix.keys()):
        
            
            convergence_matrix[gene_name]['length'] < 50 and convergence_matrix[gene_name]['length']
        
            Ls.append(convergence_matrix[gene_name]['length'])
            m = gene_parallelism_statistics[gene_name]['multiplicity']
                  
            n = 0
            nfixed = 0
                    
            for population in metapopulation_populations[metapopulation]:
                for t,L,Lclade,f in convergence_matrix[gene_name]['mutations'][population]:
                    
                    #fixed_weight = timecourse_utils.calculate_fixed_weight(L, f)
                    fixed_weight = timecourse_utils.calculate_clade_fixed_weight(Lclade,f)
                    
                    predictors.append(m)
                    responses.append(fixed_weight)
                    
                    n+=1
                    nfixed+=fixed_weight
                    
                    if t<=tstar:
                        early_predictors.append(m)
                    else:
                        late_predictors.append(m)
        
            if n > 0.5:
                gene_hits.append(n)
                gene_fixeds.append(nfixed)
                gene_predictors.append(m)
                gene_pfixs.append(nfixed*1.0/n)
                
        Ls = numpy.array(Ls)  
        ntot = len(predictors)      
        mavg = ntot*1.0/len(Ls)
        
        predictors, responses = (numpy.array(x) for x in zip(*sorted(zip(predictors, responses), key=lambda pair: (pair[0])))) 
    
        gene_hits, gene_predictors, gene_pfixs, gene_fixeds = (numpy.array(x) for x in zip(*sorted(zip(gene_hits, gene_predictors, gene_pfixs, gene_fixeds), key=lambda pair: (pair[0]))))    
    
        early_predictors.sort()
        late_predictors.sort()
       
        rescaled_predictors = numpy.exp(numpy.fabs(numpy.log(predictors/mavg)))
        
        logit_mod = sm.Logit(responses, sm.add_constant(rescaled_predictors))
        logit_res = logit_mod.fit()

        sys.stdout.write("Logistic regression for %ss:\n" % metapopulation)
        sys.stdout.write("%s\n" % str(logit_res.summary()))
        sys.stdout.write("Params:\n")
        sys.stdout.write("%s\n" % str(logit_res.params))
        
        sys.stdout.write("Avg mutation multiplicity=%g, Avg fixed mutation multiplicity=%g\n" % (predictors.sum()/len(responses), (predictors*responses).sum()/responses.sum()))
        
        alpha=0.5
        sliding_window_mids = numpy.logspace(0,2,100)
        sliding_window_lowers = sliding_window_mids/(10**(0.1))
        sliding_window_uppers = sliding_window_mids*(10**(0.1))
        sliding_window_lowers[0] = 1e-02
        
        sliding_window_idxs = (gene_predictors[None,:]<sliding_window_uppers[:,None])*(gene_predictors[None,:]>=sliding_window_lowers[:,None])
        
        window_num_genes = sliding_window_idxs.sum(axis=1)
        window_num_hits = (sliding_window_idxs*gene_hits[None,:]).sum(axis=1)
        window_num_fixeds = (sliding_window_idxs*gene_fixeds[None,:]).sum(axis=1)

        window_pfixs = window_num_fixeds*1.0/(window_num_hits+(window_num_hits==0)) 
         
        window_uppers = numpy.array([beta.ppf(0.84, window_num_fixeds[i]+1, window_num_hits[i]-window_num_fixeds[i]+1, loc=0, scale=1) for i in range(0,len(window_num_hits))])
        
        window_lowers = numpy.array([beta.ppf(0.16, window_num_fixeds[i]+1, window_num_hits[i]-window_num_fixeds[i]+1, loc=0, scale=1) for i in range(0,len(window_num_hits))])
         
        theory_predictors = sliding_window_mids
        theory_rescaled_predictors = numpy.exp(numpy.fabs(numpy.log(theory_predictors/mavg)))
        
        theory_response = 1/(1+numpy.exp(-(logit_res.params[0]+logit_res.params[1]*theory_rescaled_predictors)))
    
        pfix_axes[include_svs]['nonmutator'].fill_between( sliding_window_mids[window_num_genes>1.5], window_lowers[window_num_genes>1.5], window_uppers[window_num_genes>1.5], color=colors[metapopulation], alpha=0.5,linewidth=0)
        
        pfix_axes[include_svs]['nonmutator'].plot( sliding_window_mids[window_num_genes>1.5], window_pfixs[window_num_genes>1.5], '-', color=colors[metapopulation], alpha=0.5)
               
        pfix_axes[include_svs]['nonmutator'].semilogx(gene_predictors[-20:], gene_pfixs[-20:],'.',color=colors[metapopulation],alpha=0.5,markersize=2)
        
        sys.stdout.write("Dots for num_hits>=%g\n" % gene_hits[-20:].min())
        
        #pfix_axes[include_svs][metapopulation].plot(theory_predictors, theory_response, '-', color=colors[metapopulation], alpha=alpha)
              
        #line, = pfix_axes[include_svs][metapopulation].semilogx([1,100], [syn_pfix, syn_pfix],  '-', color=colors[metapopulation], linewidth=0.5)       
        
        #line.set_dashes((3,2))
        pfix_axes[include_svs][metapopulation].set_ylim([0,1.05])
        pfix_axes[include_svs][metapopulation].set_xlim([1,100])
        
        if include_svs==False:
            pass
            #pfix_axes[include_svs][metapopulation].set_yticklabels([])
        
        # "Done digitizing!"
    
        sys.stderr.write("Calculating null distribution...\n")
        null_survival_function = mutation_spectrum_utils.NullMultiplicitySurvivalFunction.from_parallelism_statistics(gene_parallelism_statistics)
        
        theory_ms = numpy.logspace(0,2,100)
        theory_survivals = null_survival_function(theory_ms)
        theory_survivals /= theory_survivals[0]
        
        sys.stderr.write("Done!\n")
    
        parallelism_axes[include_svs][metapopulation].step(predictors, (len(predictors)-numpy.arange(0,len(predictors)))*1.0/len(predictors), where='post',color=colors[metapopulation],alpha=alpha,label='All')
    
        parallelism_axes[include_svs][metapopulation].step(late_predictors, (len(late_predictors)-numpy.arange(0,len(late_predictors)))*1.0/len(late_predictors), where='post',color=colors[metapopulation],alpha=0.25,label='Late')
    
        parallelism_axes[include_svs][metapopulation].step(early_predictors, (len(early_predictors)-numpy.arange(0,len(early_predictors)))*1.0/len(early_predictors), where='post',color=colors[metapopulation],alpha=0.75,label='Early')
        
        if metapopulation=='nonmutator' and include_svs==True:
            parallelism_axes[include_svs][metapopulation].loglog(theory_ms,  theory_survivals,'-',linewidth=0.5,color='0.7')
        else:
            parallelism_axes[include_svs][metapopulation].loglog(theory_ms,  theory_survivals,'-',linewidth=0.5,color='0.7',label='Null')
    
        sys.stdout.write("Total mutations = %g\n" % len(predictors))
        sys.stdout.write("# m >= 2: %g\n" % (predictors>=2).sum())
        sys.stdout.write("%g of total\n" % ((predictors>=2).sum()*1.0/len(predictors)))
        sys.stdout.write("Expected # m >=2: %g\n" %  (theory_survivals[numpy.nonzero(theory_ms>=2)[0][0]]*len(predictors)))
        sys.stdout.write("Expected mavg=%g\n" % mavg)
        parallelism_axes[include_svs][metapopulation].set_ylim([1.0/len(predictors), 1.3])
        parallelism_axes[include_svs][metapopulation].set_xlim([1,100])
        
        if metapopulation=='mutator':
            legend_loc='center left'
        else:
            legend_loc='upper right'
            legend_loc=(0.6,0.45)
            
        parallelism_axes[include_svs][metapopulation].legend(loc=legend_loc,frameon=False,fontsize=5,handlelength=1)

#####
#
# Tweak xlim,ylim and ticklabels
#
#####

###
#
# Fig 5
#
###

nonmutator_axis.set_xticks([i*10000 for i in range(0,7)]+[67500])
#nonmutator_axis.set_xticklabels(["" for i in xrange(0,9)])
mutator_axis.set_xticks([i*10000 for i in range(0,7)]+[67500])
mutator_axis.set_xticklabels(['0']+['%d0k' % i for i in range(1,7)]+['Fixed'])
#nonmutator_axis.set_xticklabels(['0']+['%d0k' % i for i in xrange(1,7)]+['Fixed'])
nonmutator_axis.set_xticklabels([])      
nonmutator_axis.legend(loc='center left',frameon=False,handlelength=1)
vartype_pfix_axis.legend(loc='upper right',frameon=False,handlelength=1)        

nonmutator_parallelism_axis.set_xlim([1,1e02])        
nonmutator_parallelism_axis.set_xticklabels([])
mutator_parallelism_axis.set_xlim([1,1e02])

nonmutator_pfix_axis.set_xlim([1,1e02])
nonmutator_pfix_axis.set_ylim([0,1.05])

#mutator_pfix_axis.set_xlim([1,1e02])
#mutator_pfix_axis.set_ylim([0,1.05])

nonmutator_axis.text(2000,1.95,figure_utils.get_panel_label('a'),fontsize=6,fontweight='bold')
nonmutator_axis.text(4500,1.95,'Nonmutators',fontsize=5)

mutator_axis.text(2000,30.5,figure_utils.get_panel_label('b'),fontsize=6,fontweight='bold')
mutator_axis.text(4500,30.5,'Mutators',fontsize=5)

time_axis.text(0.05,1.05,figure_utils.get_panel_label('c'),fontsize=6,fontweight='bold')
vartype_pfix_axis.text(0,0.97,figure_utils.get_panel_label('d'),fontsize=6,fontweight='bold')
nonmutator_parallelism_axis.text(1.3,1.3,figure_utils.get_panel_label('e'),fontsize=6,fontweight='bold')
mutator_parallelism_axis.text(1.3,1.3,figure_utils.get_panel_label('f'),fontsize=6,fontweight='bold')
nonmutator_pfix_axis.text(1.5,0.97,figure_utils.get_panel_label('g'),fontsize=6,fontweight='bold')
#mutator_pfix_axis.text(1.5,0.97,figure_utils.get_panel_label('g'),fontsize=6,fontweight='bold')

###
#
# No sv fig
#
###

nonmutator_parallelism_axis_nosv.set_xlim([1,1e02])        
#nonmutator_parallelism_axis_nosv.set_xticklabels([])
mutator_parallelism_axis_nosv.set_xlim([1,1e02])
#mutator_parallelism_axis_nosv.set_xticklabels([])

nonmutator_pfix_axis_nosv.set_xlim([1,1e02])
nonmutator_pfix_axis_nosv.set_ylim([0,1.05])

#mutator_pfix_axis_nosv.set_xlim([1,1e02])
#mutator_pfix_axis_nosv.set_ylim([0,1.05])
#mutator_pfix_axis_nosv.set_yticklabels([])


####
#
# Do calculation
#
####

excluded_types = set(['sv'])

# map of variant types 
appearance_times = {}

pooled_appearance_times = []
pooled_var_types = []
restricted_appearance_times = []
restricted_var_types = []

populations = parse_file.complete_nonmutator_lines
        
for population in populations:
    sys.stderr.write("Processing %s...\t" % parse_file.get_pretty_name(population))

    # calculate mutation trajectories
    # load mutations
    mutations, depth_tuple = parse_file.parse_annotated_timecourse(population)
    
    population_avg_depth_times, population_avg_depths, clone_avg_depth_times, clone_avg_depths = depth_tuple
    
    dummy_times,fmajors,fminors,haplotype_trajectories = parse_file.parse_haplotype_timecourse(population)
    state_times, state_trajectories = parse_file.parse_well_mixed_state_timecourse(population)
        
    num_processed_mutations = 0
        
    for mutation_idx in range(0,len(mutations)):
 
        num_processed_mutations+=1 
            
        position, gene_name, allele, var_type, test_statistic, pvalue, cutoff_idx, depth_fold_change, depth_change_pvalue, times, alts, depths, clone_times, clone_alts, clone_depths = mutations[mutation_idx] 
        Ls = haplotype_trajectories[mutation_idx]
        state_Ls = state_trajectories[mutation_idx]
        
        good_idxs, filtered_alts, filtered_depths = timecourse_utils.mask_timepoints(times, alts, depths, var_type, cutoff_idx, depth_fold_change, depth_change_pvalue)
    
        freqs = timecourse_utils.estimate_frequencies(filtered_alts, filtered_depths)
        
        masked_times = times[good_idxs]
        masked_freqs = freqs[good_idxs]
        masked_state_Ls = state_Ls[good_idxs]
        masked_Ls = Ls[good_idxs]
            
        t = timecourse_utils.calculate_appearance_time(masked_times, masked_freqs, masked_state_Ls, masked_Ls)
        
        pooled_appearance_times.append(t)
        pooled_var_types.append(var_type)
        
        if var_type not in excluded_types:
            restricted_appearance_times.append(t)
            restricted_var_types.append(var_type)
        
    sys.stderr.write("added %d mutations\n" % num_processed_mutations)
    

# sort the lists (but preserve association between time and variant type
pooled_appearance_times, pooled_var_types = (list(x) for x in zip(*sorted(zip(pooled_appearance_times, pooled_var_types), key=lambda pair: pair[0])))
restricted_appearance_times, restricted_var_types = (list(x) for x in zip(*sorted(zip(restricted_appearance_times, restricted_var_types), key=lambda pair: pair[0])))

pooled_appearance_times = numpy.array(pooled_appearance_times)
restricted_appearance_times = numpy.array(restricted_appearance_times)

# calculate appearance time distribution for each variant type
observed_appearance_times = {}
observed_restricted_appearance_times = {}

for t,var_type in zip(pooled_appearance_times, pooled_var_types):
    if var_type not in observed_appearance_times:
        observed_appearance_times[var_type] = []
    observed_appearance_times[var_type].append(t)

for var_type in list(observed_appearance_times.keys()):
    observed_appearance_times[var_type] = numpy.array(observed_appearance_times[var_type])
    observed_appearance_times[var_type].sort()  

for t,var_type in zip(restricted_appearance_times, restricted_var_types):
    if var_type not in observed_restricted_appearance_times:
        observed_restricted_appearance_times[var_type] = []
    observed_restricted_appearance_times[var_type].append(t)
    
for var_type in list(observed_restricted_appearance_times.keys()):
    observed_restricted_appearance_times[var_type] = numpy.array(observed_restricted_appearance_times[var_type])
    observed_restricted_appearance_times[var_type].sort()  

####
#
# Now do bootstrap resampling to gauge significance
#
####

num_bootstraps = 10000
bootstrapped_kss = {var_type:[] for var_type in list(observed_appearance_times.keys())}
bootstrapped_restricted_kss = {var_type:[] for var_type in list(observed_restricted_appearance_times.keys())}

sys.stderr.write('Resampling %d bootstraps...\n' % num_bootstraps)
for bootstrap_idx in range(0,num_bootstraps):

    # permute var type labels
    shuffle(pooled_var_types)
    shuffle(restricted_var_types) 
    
    # recalculate distributions for each type
    bootstrapped_appearance_times = {}
    bootstrapped_restricted_appearance_times = {}

    for t,var_type in zip(pooled_appearance_times, pooled_var_types):
        if var_type not in bootstrapped_appearance_times:
            bootstrapped_appearance_times[var_type] = []
        bootstrapped_appearance_times[var_type].append(t)

    for var_type in list(bootstrapped_appearance_times.keys()):
        bootstrapped_appearance_times[var_type].sort()  

    for t,var_type in zip(restricted_appearance_times, restricted_var_types):
        if var_type not in bootstrapped_restricted_appearance_times:
            bootstrapped_restricted_appearance_times[var_type] = []
        bootstrapped_restricted_appearance_times[var_type].append(t)
    
    for var_type in list(bootstrapped_restricted_appearance_times.keys()):
        bootstrapped_restricted_appearance_times[var_type].sort()  

    # recalculate ks distances
    for var_type in list(bootstrapped_appearance_times.keys()):
        D = stats_utils.calculate_ks_distance(bootstrapped_appearance_times[var_type], pooled_appearance_times)
        bootstrapped_kss[var_type].append(D)
        
    for var_type in list(bootstrapped_restricted_appearance_times.keys()):
        D = stats_utils.calculate_ks_distance(bootstrapped_restricted_appearance_times[var_type], restricted_appearance_times)
        bootstrapped_restricted_kss[var_type].append(D)

# calculate pvalues
sys.stdout.write('Pvalues for temporal nonuniformity (n=%d bootstraps):\n' % num_bootstraps)
for var_type in list(observed_appearance_times.keys()):

    bootstrapped_appearance_times[var_type] = numpy.array(bootstrapped_appearance_times[var_type])

    D = stats_utils.calculate_ks_distance(observed_appearance_times[var_type], pooled_appearance_times)
    
    pvalue = ((bootstrapped_kss[var_type]>=D).sum()+1.0)/(len(bootstrapped_kss[var_type])+1.0)
        
    sys.stdout.write('%s: %g\n' % (var_type, pvalue))

sys.stdout.write('Excluding svs:\n')
for var_type in list(observed_restricted_appearance_times.keys()):

    bootstrapped_restricted_appearance_times[var_type] = numpy.array(bootstrapped_restricted_appearance_times[var_type])

    D = stats_utils.calculate_ks_distance(observed_restricted_appearance_times[var_type], restricted_appearance_times)
    
    pvalue = ((bootstrapped_restricted_kss[var_type]>=D).sum()+1.0)/(len(bootstrapped_restricted_kss[var_type])+1.0)
        
    sys.stdout.write('%s: %g\n' % (var_type, pvalue))
    
######
#
# Now do plotting
#
######            

all_ts, all_survivals = stats_utils.calculate_unnormalized_survival_from_vector(pooled_appearance_times, min_x=-1000,max_x=100000)

time_axis.step(all_ts, all_survivals/all_survivals[0], color='k', label='All types')
#missense_time_axis.step(all_ts, all_survivals/all_survivals[0], color='k', label='All')

restricted_ts, restricted_survivals = stats_utils.calculate_unnormalized_survival_from_vector(restricted_appearance_times, min_x=-1000,max_x=100000)
#missense_time_axis.step(all_ts, restricted_survivals/restricted_survivals[0], color='k', label='All (excluding sv)',alpha=0.5)



for var_type in parse_file.var_types:
 
    color = figure_utils.get_var_type_color(var_type)
    vartype_ts, vartype_survivals = stats_utils.calculate_unnormalized_survival_from_vector(observed_appearance_times[var_type], min_x=-1000, max_x=100000)
    time_axis.step(vartype_ts, vartype_survivals/vartype_survivals[0], color=color, alpha=0.7) #, label=var_type)

    if var_type == 'missense':
        pass
        #missense_time_axis.step(vartype_ts, vartype_survivals/vartype_survivals[0], color=color, alpha=0.7, label=var_type)
    

time_axis.legend(loc='upper right', frameon=False,handlelength=1)

# save figure to disk
fig5.savefig(parse_file.figure_directory+'fig5.pdf',bbox_inches='tight')  
nosv_fig.savefig(parse_file.figure_directory+'supplemental_parallelism_nosv.pdf',bbox_inches='tight')
#pylab.show()