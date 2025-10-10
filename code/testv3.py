import matplotlib.pyplot as plt
import nest
import numpy as np
import os
import math
from help_funcs import *
import sys
from extrapolation import gamma_kernel, pars0, pars12
# choose between 'dist' (distance-dependent, ZT0/12) and 'fixed' (old near/far)
DEFAULT_CONN_MODE = 'dist'  # or 'fixed'
PV2E_LOCAL_R_MM = 0.05       # 50 µm
PV2E_LOCAL_P    = 0.79
PV2E_MAX_R_MM   = 0.25       # zero beyond 250 µm, for symmetry with E→PV

USE_RECT_STIM  = True    # True => rectangle; False => keep Gaussian (legacy)
RECT_CENTER_X  = 0.50    # center of the vertical strip on the 1-mm torus

# --- PV→E calibration toggles ---
PV2E_TARGET_RATIO_ZT12divZT0 = 1.8       # from Zong et al.
PV2E_SPLIT_RATIO = True                  # True: split 1.8 into spread×weight; False: probability-only
PV2E_AREA_RATIO  = 1.33 if PV2E_SPLIT_RATIO else 1.8   # factor via longer λ at ZT12
PV2E_WGAIN_ZT12  = 1.8 / PV2E_AREA_RATIO               # ≈1.35 if split; else 1.0
# Base tail length constants (in mm). Choose a sane λ_ZT0, we’ll calibrate λ_ZT12.
PV2E_LAMBDA_ZT0_MM  = 0.06     # starting guess (60 µm)
PV2E_LAMBDA_ZT12_MM = None     # will be filled by calibration
print(f"[PV→E setup] target ratio(ZT12/ZT0)={PV2E_TARGET_RATIO_ZT12divZT0}, "
      f"split={PV2E_SPLIT_RATIO}, area_ratio={PV2E_AREA_RATIO:.3f}, "
      f"weight_gain={PV2E_WGAIN_ZT12:.3f}, λ_ZT0={PV2E_LAMBDA_ZT0_MM*1000:.0f}µm")





def gamma_kernel_nest(x, A, alpha, beta):
    # x: distance in µm (can be a NEST Parameter e.g., nest.spatial.distance*1000)
    # use ** and nest.math.exp so it works on Parameters
    return A * (x ** (alpha - 1.0)) * nest.math.exp(-beta * x)

def get_conn_prob(dist_mm, circadian):
    """Return Pyr→PV connection probability as a NEST expression.
       dist_mm: nest.spatial.distance (in mm). Hard-zero beyond 250 µm.
    """
    d_um = dist_mm * 1000.0
    A, alpha, beta = (pars0 if circadian.upper() == 'ZT0' else pars12)
    p_raw = gamma_kernel_nest(d_um, float(A), float(alpha), float(beta))
    # Clamp to [0, 1] using nested conditionals (works with NEST Parameters)
    p_raw = nest.logic.conditional(p_raw > 1.0, 1.0, nest.logic.conditional(p_raw < 0.0, 0.0, p_raw))
    # Hard-zero beyond 250 µm
    return nest.logic.conditional(dist_mm <= 0.25, p_raw, 0.0)

def pv2e_tail_p(N_pe_target, n_pv, p_local=PV2E_LOCAL_P, r_local=PV2E_LOCAL_R_MM):
    # fraction of presynaptic PV within r_local on the torus
    frac_local = np.pi * (r_local**2)           # area fraction
    frac_rest  = max(1.0 - frac_local, 1e-9)
    # Choose a constant p_out so expected indegree stays ≈ N_pe_target
    p_out = (N_pe_target / n_pv - p_local * frac_local) / frac_rest
    return float(min(1.0, max(0.0, p_out)))

def run_simulation(sim_parameters, condition, rngseed = '', verbose=False, circadian='ZT0'):
    print("SIMULATING:", condition)

    def _pv2e_integrated_mass(lambda_mm, r_local=PV2E_LOCAL_R_MM, p_local=PV2E_LOCAL_P, r_max=PV2E_MAX_R_MM):
        """
        Continuous 2D radial integral of connection probability on the 1x1 mm torus
        within max radius r_max: ∫ p(r) 2πr dr
        - p(r)=p_local for r<=r_local
        - p(r)=p_local * exp(-(r-r_local)/lambda_mm) for r_local<r<=r_max
        """
        A_local = p_local * math.pi * (r_local**2)
        # tail integral: ∫_{r_local}^{r_max} [p_local * exp(-(r-r0)/λ)] * 2πr dr
        r0 = r_local
        lam = max(lambda_mm, 1e-6)
        # closed form of ∫ r e^{- (r-r0)/λ} dr with u=r-r0
        # ∫ (u+r0) e^{-u/λ} du = -(λ u + λ^2) e^{-u/λ} - r0 λ e^{-u/λ}
        def F(r):
            u = r - r0
            return -(lam*u + lam*lam + r0*lam) * math.exp(-u/lam)
        tail = 2*math.pi*p_local * (F(r_max) - F(r_local))
        return A_local + max(tail, 0.0)
    def _calibrate_lambda_zt12(lambda_zt0, area_ratio_target):
        # scan λ_ZT12 to achieve desired area ratio
        base = _pv2e_integrated_mass(lambda_zt0)
        # search between λ_ZT0 and ~4× larger
        grid = np.linspace(lambda_zt0, 4.0*lambda_zt0, 400)
        vals = np.array([_pv2e_integrated_mass(l) for l in grid])
        target = area_ratio_target * base
        idx = (np.abs(vals - target)).argmin()
        return float(grid[idx]), float(vals[idx]/base)

    # Calibrate λ_ZT12 and report effective ratios (area and total with weight)
    PV2E_LAMBDA_ZT12_MM, eff_area_ratio = _calibrate_lambda_zt12(PV2E_LAMBDA_ZT0_MM, PV2E_AREA_RATIO)
    eff_total_ratio = eff_area_ratio * PV2E_WGAIN_ZT12
    print(f"[PV→E calibration] λ_ZT12≈{PV2E_LAMBDA_ZT12_MM*1000:.0f}µm, "
          f"area_ratio≈{eff_area_ratio:.3f}, weight_gain={PV2E_WGAIN_ZT12:.3f} "
          f"→ total_ratio≈{eff_total_ratio:.3f}")
    
    nest.ResetKernel()
    nonlin_name = "iaf_cond_exp_dend_nestml"
    try:
        nest.Install("nestml_iaf_module")
    except :
        pass

    nest.SetKernelStatus({'local_num_threads': 8})
    if rngseed != '':
        nest.rng_seed = rngseed
        rng = np.random.default_rng(rngseed)

    def reset_sim():
        nest.SetKernelStatus({'biological_time': 0.})
        for popu in [exc_neurons, pv_neurons, sst_neurons]:
            dVms =  {"V_m": [Vr+(Vt-Vr)*np.random.rand() for x in range(len(popu))]}
            popu.set(dVms)
    
    ## Simulation Parameters
    gext_baseline = sim_parameters['gext_baseline']
    g_tha_e        = sim_parameters['g_tha_e'] * 1000.0
    g_tha_p        = sim_parameters['g_tha_p'] * 1000.0
    par_gext_rate0 = sim_parameters['par_gext_rate0']
    par_gext_rate1 = sim_parameters['par_gext_rate1']
    vis_sat        = sim_parameters['vis_sat']
    
    par_ext_syn_1 = sim_parameters['par_ext_syn_1'] * 1000.0
    par_ext_syn_2 = sim_parameters['par_ext_syn_2'] * 1000.0
    par_ext_syn_3 = sim_parameters['par_ext_syn_3'] * 1000.0
    par_ext_syn_4 = sim_parameters['par_ext_syn_4'] * 1000.0
    chr2_str_sst  = sim_parameters['chr2_str_sst'] * 1000.0
    chr2_str_pv   = sim_parameters['chr2_str_pv'] * 1000.0
    chr2_rampt    = sim_parameters['chr2_rampt'] 
    chr2_duration    = sim_parameters['chr2_duration'] 
    
    g_syn_ee      = sim_parameters['g_syn_ee'] * 1000.0
    g_syn_ep      = sim_parameters['g_syn_ep'] * 1000.0
    g_syn_es      = sim_parameters['g_syn_es'] * 1000.0
    g_syn_pe      = sim_parameters['g_syn_pe'] * 1000.0
    g_syn_pp      = sim_parameters['g_syn_pp'] * 1000.0
    g_syn_se      = sim_parameters['g_syn_se'] * 1000.0
    g_syn_sp      = sim_parameters['g_syn_sp'] * 1000.0
    #g_syn_ss      = sim_parameters['g_syn_ss']

    alpha_ei_inv  = 1./sim_parameters['alpha_ei']
    stim_range = sim_parameters['stim_range']
    
    epsilon       = sim_parameters['epsilon']  # 0.02
    epsilon_th    = sim_parameters['epsilon_th']   #0.02
    p_far_e       = sim_parameters['p_far_e']
    p_far_p       = sim_parameters['p_far_p']
    p_far_s       = sim_parameters['p_far_s']
    p_chr2_random = sim_parameters['p_chr2_random']

    nb_repeats    = int(sim_parameters['nb_repeats'])
    
    ### Cell parameters ###
    tau_m     =  25.  # ms Membrane time constant
    c_m       =  0.25*1e3 # pF Capacitance
    tau_exc   =  6.   # ms Synaptic time constant (excitatory)
    tau_inh   =  20.  # ms Synaptic time constant (inhibitory)
    tau_ref   =  5.   # ms Refractory period
    El        = -70.  # mV Leak potential
    Vt        = -40.  # mV Spike Threhold
    Vr        = -65.  # mV Spike Reset
    e_rev_E   =   0.  # mV Reversal Potential for excitatory synapses
    e_rev_I   = -75.  # mV Reversal Potential for inhibitory synapses
    
    
    exc_params = {
        'tau_syn_exc'  : tau_exc,  'tau_syn_inh'  : tau_inh,
        'E_L'     : El,       'V_reset'    : Vr,       'V_th'   : Vt,
        'C_m'         : c_m,      't_ref' : tau_ref,  'E_exc'    : e_rev_E,
        'E_inh'    : e_rev_I, 'alpha_ei_inv'  :  alpha_ei_inv}

    neuron_params = {
        'tau_syn_ex'  : tau_exc,  'tau_syn_in'  : tau_inh,
        'E_L'     : El,       'V_reset'    : Vr,       'V_th'   : Vt,
        'C_m'         : c_m,      't_ref' : tau_ref,  'E_ex'    : e_rev_E,
        'E_in'    : e_rev_I}
    
    
    ### Network Parameters ###
    n_cells       = 10000               # Total number of cells in the recurrent network
    n_thalamus    = 1000                # Total number of cells in the input layer
    n_exc         = 8000
    n_inh         = n_cells - n_exc
    n_sst         = int(n_inh/2)
    n_pv         = int(n_inh/2)
    size          = float(1.)           # Size of the network
    simtime       = float(800)          # ms Simulation time for each trial
    se_lat        = 0.1                 # Spread of the lateral excitatory connections
    si_lat        = 0.1                 # Spread of the lateral inhibitory connections
    st_lat        = 0.2                 # Spread of the thalamic excitatory connections
    velocity      = 0.05                # mm/ms  velocity
    
    ### External Input Parameters ###
    m_xy          = [size/2., size/2.]  # center of the gaussian [x, y]
    sigma_mm      = float(sim_parameters.get('stim_sigma_mm', size/5.0))
    s_xy          = [sigma_mm, sigma_mm]  # sigma of the gaussian [sx, sy]
    m_time        = 100.         # time to peak of the Gaussian
    s_time        = 30.                 # sigma in time of the Gaussian
    time_spacing  = 0.                  # time in between two stimulation
    sim_delay     = 200.
    #s_xyies       = [(np.array(s_xy)*i).tolist() for i in np.ones(nb_repeats)]
    
    ### Simulation parameters ###
    dt            = 0.1                     # ms Time step
    max_distance  = size/np.sqrt(2)      # Since this is a torus
    max_delay     = dt + max_distance/velocity # Needed for the connectors
    min_delay     = 0.1                     # ms
    #rngseed       = 92847459                # Random seed
    #parallel_safe = False                   # To have results independant on number of nodes
    #verbose       = True                    # To display the state of the connectors
    #space         = Space(periodic_boundaries=((0, 1), (0, 1), (0, 1))) # Specify the boundary conditions
    
    
    ### Initialize Kernel Parameters
    nest.resolution = dt
    nest.set(min_delay=min_delay, max_delay=max_delay+dt)
    nest.set(print_time=True)
    
    Nth          = int(epsilon_th*n_thalamus)
    
    exc_pos = nest.spatial.free(pos=nest.random.uniform(min=0., max=1.),
                               extent=[1.,1.], edge_wrap=True)
    exc_neurons = nest.Create(nonlin_name, n_exc, params=exc_params, positions=exc_pos)
    
    pv_pos = nest.spatial.free(pos=nest.random.uniform(min=0., max=1.),
                               extent=[1.,1.], edge_wrap=True)
    pv_neurons = nest.Create('iaf_cond_exp', n_pv, params=neuron_params, positions=pv_pos)
    
    sst_pos = nest.spatial.free(pos=nest.random.uniform(min=0., max=1.),
                               extent=[1.,1.], edge_wrap=True)
    sst_neurons = nest.Create('iaf_cond_exp', n_sst,params=neuron_params,  positions=sst_pos)
    
    # noise_pos = nest.spatial.free(pos=nest.random.uniform(min=0., max=1.),
    #                            num_dimensions=2, edge_wrap=True)
    addnoise = nest.Create('poisson_generator', n_thalamus, {'rate':gext_baseline}) #, positions=noise_pos)
    
    # Connect external Poisson noise
    n2e_conn = {'rule': 'fixed_indegree', 'indegree': Nth}
    n2e_syn = {'synapse_model':'static_synapse','receptor_type': 1, 'weight': par_ext_syn_2*1.5e-3, 'delay': 0.1}
    n2e_synapses = nest.Connect(addnoise, exc_neurons, n2e_conn, syn_spec=n2e_syn)
    
    n2p_conn = {'rule': 'fixed_indegree', 'indegree': Nth}
    n2p_syn = {'synapse_model':'static_synapse', 'weight': par_ext_syn_1*1.5e-3, 'delay': 0.1}
    n2p_synapses = nest.Connect(addnoise, pv_neurons, n2p_conn, syn_spec=n2p_syn)
    
    n2s_conne = {'rule': 'fixed_indegree', 'indegree': Nth}
    n2s_syne = {'synapse_model':'static_synapse', 'weight': par_ext_syn_3*1.5e-3, 'delay': 0.1}
    n2s_e_synapses = nest.Connect(addnoise, sst_neurons, n2s_conne, syn_spec=n2s_syne)
    
    n2s_conni = {'rule': 'fixed_indegree', 'indegree': Nth}
    n2s_syni = {'synapse_model':'static_synapse', 'weight': -par_ext_syn_4*1.5e-3, 'delay': 0.1}
    n2s_i_synapses = nest.Connect(addnoise, sst_neurons, n2s_conni, syn_spec=n2s_syni)
    
    
    ### Connectivity Parameters
    ## Connectivity from Campagnola, et al. 2022
    cprob_ee   = 0.06
    cprob_ep   = 0.42
    cprob_es   = 0.3
    cprob_pe   = 0.35
    cprob_pp   = 0.39
    cprob_ps   = 0  #0.09
    cprob_se   = 0.23
    cprob_sp   = 0.18
    cprob_ss   = 0  #0.05
    
    #p_far_e    = 0.3
    #p_far_p    = 0.3
    #p_far_s    = 0.6
    
    N_ee       = int(epsilon*n_exc*cprob_ee)
    N_ee_far   = int(N_ee*p_far_e)
    N_ee_close = N_ee - N_ee_far
    
    N_ep       = int(epsilon*n_exc*cprob_ep)
    N_ep_far   = int(N_ep*p_far_e)
    N_ep_close = N_ep - N_ep_far
    
    N_es       = int(epsilon*n_exc*cprob_es)
    
    N_pe       = int(epsilon*n_inh/2*cprob_pe)
    N_pe_far   = int(N_pe*p_far_p)
    N_pe_close = N_pe - N_pe_far
    
    N_pp       = int(epsilon*n_inh/2*cprob_pp)
    N_pp_far   = int(N_pp*p_far_p)
    N_pp_close = N_pp - N_pp_far
    
    N_se       = int(epsilon*n_inh/2*cprob_se)
    N_se_far   = int(N_se*p_far_s)
    N_se_close = N_se - N_se_far
    
    N_sp       = int(epsilon*n_inh/2*cprob_sp)
    N_sp_far   = int(N_sp*p_far_s)
    N_sp_close = N_sp - N_sp_far
    
    ratio_in = se_lat*se_lat*np.pi
    ratio_out = 1.0 - ratio_in
    p_in_ee = N_ee_close/(ratio_in * n_exc)
    p_out_ee =  N_ee_far/(ratio_out * n_exc)
    e2e_mask = nest.logic.conditional(nest.spatial.distance <= se_lat, p_in_ee,p_out_ee)
    e2e_conn = {'rule': 'pairwise_bernoulli', 'p': e2e_mask}
    e2e_syn = {'synapse_model':'static_synapse','receptor_type': 1, 'weight': g_syn_ee, 'delay': 0.1 + nest.spatial.distance/velocity}
    nest.Connect(exc_neurons, exc_neurons, e2e_conn, syn_spec = e2e_syn)
    
    p_in_ep = N_ep_close/(ratio_in * n_exc)
    p_out_ep =  N_ep_far/(ratio_out * n_exc)
    if conn_mode == 'dist':
        p_e2p = get_conn_prob(nest.spatial.distance, circadian)
        e2p_conn = {'rule': 'pairwise_bernoulli', 'p': p_e2p}
    else:
        e2p_mask = nest.logic.conditional(nest.spatial.distance <= se_lat, p_in_ep, p_out_ep)
        e2p_conn = {'rule': 'pairwise_bernoulli', 'p': e2p_mask}
    
    e2p_syn = {'synapse_model':'static_synapse', 'weight': g_syn_ep, 'delay': 0.1 + nest.spatial.distance/velocity}
    nest.Connect(exc_neurons, pv_neurons, e2p_conn, syn_spec = e2p_syn)
    
    e2s_conn = {'rule': 'pairwise_bernoulli', 'p': N_es/n_exc}
    e2s_syn = {'synapse_model':'static_synapse', 'weight': g_syn_es, 'delay': 0.1 + nest.spatial.distance/velocity}
    nest.Connect(exc_neurons, sst_neurons, e2s_conn, syn_spec = e2s_syn)

    d = nest.spatial.distance  # in mm
    lam_used = (PV2E_LAMBDA_ZT0_MM if circadian.upper()=='ZT0' else PV2E_LAMBDA_ZT12_MM)
    p_local = PV2E_LOCAL_P
    # exponential tail starting at r_local with length constant lam_used (in mm)
    p_tail_expr = p_local * nest.math.exp(-(d - PV2E_LOCAL_R_MM)/lam_used)
    p_tail_expr = nest.logic.conditional(d > PV2E_LOCAL_R_MM, p_tail_expr, p_local)
    # zero beyond PV2E_MAX_R_MM (e.g., 250 µm)
    p2e_mask = nest.logic.conditional(d <= PV2E_MAX_R_MM, p_tail_expr, 0.0)
    p2e_conn = {'rule': 'pairwise_bernoulli', 'p': p2e_mask}

    # apply optional ZT12 weight gain to encode residual amplitude difference
    weight_p2e = g_syn_pe * (1.0 if circadian.upper()=='ZT0' else PV2E_WGAIN_ZT12)
    p2e_syn = {'synapse_model':'static_synapse','receptor_type': 2, 'weight': weight_p2e, 'delay': 0.1 + d/velocity}

    print(f"[PV→E connect] circadian={circadian}, λ={lam_used*1000:.0f}µm, "
          f"weight_gain={'1.00' if circadian.upper()=='ZT0' else f'{PV2E_WGAIN_ZT12:.2f}'}")
    nest.Connect(pv_neurons, exc_neurons, p2e_conn, syn_spec = p2e_syn)

    
    p_in_pp = N_pp_close/(ratio_in * n_pv)
    p_out_pp =  N_pp_far/(ratio_out * n_pv)
    p2p_mask = nest.logic.conditional(nest.spatial.distance <= se_lat, p_in_pp,p_out_pp)
    p2p_conn = {'rule': 'pairwise_bernoulli', 'p': p2p_mask}
    p2p_syn = {'synapse_model':'static_synapse', 'weight': -g_syn_pp, 'delay': 0.1 + nest.spatial.distance/velocity}
    nest.Connect(pv_neurons, pv_neurons, p2p_conn, syn_spec = p2p_syn)

    p_in_se = N_se_close/(ratio_in * n_sst)
    p_out_se =  N_se_far/(ratio_out * n_sst)
    s2e_mask = nest.logic.conditional(nest.spatial.distance <= se_lat, p_in_se,p_out_se)
    s2e_conn = {'rule': 'pairwise_bernoulli', 'p': s2e_mask}
    s2e_syn = {'synapse_model':'static_synapse','receptor_type': 3, 'weight': g_syn_se, 'delay': 0.1 + nest.spatial.distance/velocity}
    nest.Connect(sst_neurons, exc_neurons, s2e_conn, syn_spec = s2e_syn)
    
    p_in_sp = N_sp_close/(ratio_in * n_sst)
    p_out_sp =  N_sp_far/(ratio_out * n_sst)
    s2p_mask = nest.logic.conditional(nest.spatial.distance <= se_lat, p_in_sp,p_out_sp)
    s2p_conn = {'rule': 'pairwise_bernoulli', 'p': s2p_mask}
    s2p_syn = {'synapse_model':'static_synapse', 'weight': -g_syn_sp, 'delay': 0.1 + nest.spatial.distance/velocity}
    nest.Connect(sst_neurons, pv_neurons, s2p_conn, syn_spec = s2p_syn)
    
    #### Stimulation 
    stim_type, contrast = condition
    if stim_type == 'Spont':
        print("Spontaneous Sim")
        sim_name = 'Spont'
        sim_spontaneous = True
    elif stim_type == 'Control':
        print("Control Sim; no Chr2")
        sim_name = f'NoOpto_{circadian}_{contrast}_{conn_mode}'
        sim_spontaneous = False
    elif stim_type in ['PV', 'SST']:
        print(stim_type, 'stimulation at contrast', contrast)
        sim_name = f'{stim_type}_{contrast}'
        sim_spontaneous = False
    else:
        raise Exception()
    # Tag the output with the stimulus sigma (mm)
    # Tag output with width (rect) or sigma (legacy Gaussian), in mm
    try:
        if USE_RECT_STIM:
            sim_name += f"_w{float(sim_parameters.get('stim_rect_width_mm', 0.0)):.3f}mm"
        else:
            sim_name += f"_sig{float(sim_parameters.get('stim_sigma_mm', 0.0)):.3f}mm"
    except Exception:
        pass


    
    ### Reset time and variables
    reset_sim()
    
    ### Visual and Chr2 stimulation ###
    if not sim_spontaneous:
        #gext_rate     = par_gext_rate0 + par_gext_rate1*contrast  #contrast varies between 0.02 and 1
        gext_rate     = par_gext_rate0 + par_gext_rate1*np.tanh(contrast/vis_sat)  #contrast varies between 0.02 and 1
        intensities   = gext_rate*np.ones(nb_repeats)
        Chr2_times = np.array([sim_delay + simtime*(1.+2.*i) for i in range(nb_repeats)]) # Induce an instantaneous synaptic conductance in the target population at time 0 every other trial
        Chr2_proba = 1.    # percentage of cells that will receive the conductance change
    
        th_pos = nest.spatial.free(pos=nest.random.uniform(min=0., max=1.),
                                   extent=[1.,1.], edge_wrap=True)
        thalamus = nest.Create('inhomogeneous_poisson_generator', n_thalamus, positions=th_pos)
        
        #times     = np.arange(0, 200, 1.)
        times     = np.arange(0, 400, 1.)
        times [0] = 0.1
        for ni in range(n_thalamus):
            x, y = nest.GetPosition(thalamus[ni])
            rate_times = []
            rate_values = []
    
            for repeat in range(nb_repeats):
                #padding = repeat * (simtime+time_spacing)
                padding = repeat * (simtime+time_spacing) + sim_delay
                if USE_RECT_STIM:
                # runtime width (default 0.20mm if not overridden)
                    rect_w = float(sim_parameters.get('stim_rect_width_mm', 0.20))
                    inside = inside_vertical_strip_periodic(x, RECT_CENTER_X, rect_w)
                    if inside:
                        temporal = np.exp(-((times - m_time)**2 / (2*s_time**2)))
                        rates = gext_rate * temporal
                    else:
                        rates = np.zeros_like(times)
                else:
                # legacy Gaussian/Gabor-like spatial envelope
                    rates = gext_rate * np.exp(-((times-m_time)**2/(2*s_time**2)
                                     +(x-m_xy[0])**2/(2*s_xy[0]**2)
                                     +(y-m_xy[1])**2/(2*s_xy[1]**2)))
                
                
                rate_times += list(times+ padding)
                rate_values += list(rates)
    
            #print('GAUSSIAN', ni)
            #print(rate_times)
            #print(rate_values)
    
            nest.SetStatus(thalamus[ni], {'rate_times':rate_times, 'rate_values':rate_values})
    
        t2e_conn = {'rule': 'fixed_indegree', 'indegree': Nth, 'mask': {'circular': {'radius': st_lat}}}
        t2e_syn = {'synapse_model':'static_synapse','receptor_type': 1, 'weight': g_tha_e, 'delay': 0.1}
        nest.Connect(thalamus, exc_neurons,t2e_conn, syn_spec=t2e_syn)
    
        t2p_conn = {'rule': 'fixed_indegree', 'indegree': Nth, 'mask': {'circular': {'radius': st_lat}}}
        t2p_syn = {'synapse_model':'static_synapse', 'weight': g_tha_p, 'delay': 0.1}
        nest.Connect(thalamus, pv_neurons,t2p_conn, syn_spec=t2p_syn)

        #print('THALAMUS EXC CONNECTIONS:', len(nest.GetConnections(thalamus, exc_neurons)))
        #print('THALAMUS PV CONNECTIONS:', len(nest.GetConnections(thalamus, pv_neurons)))
        ENABLE_CHR2 = False
        if ENABLE_CHR2 and stim_type in ['PV', 'SST']:
            source_chr2 = nest.Create('inhomogeneous_poisson_generator', 1)
            chr2_rates = []
            chr2_switch_times = []
            chr2_rate = 1000.
            for stime in Chr2_times:

                for ti in range(int(chr2_rampt)):
                    chr2_switch_times.append(stime-chr2_rampt+ti)
                    chr2_rates.append( chr2_rate * ti / chr2_rampt)

                chr2_rates.append(chr2_rate)
                chr2_switch_times.append(stime)

                chr2_rates.append(0)
                chr2_switch_times.append(stime+chr2_duration)

            nest.SetStatus(source_chr2, {'rate_times':chr2_switch_times, 'rate_values':chr2_rates})

            if stim_type == 'PV':
                targets = pv_neurons
                chr2_str = chr2_str_pv
            elif stim_type == 'SST':
                targets = sst_neurons
                chr2_str = chr2_str_sst
            else:
                raise Exception()
    
            chr_syn = {'synapse_model':'static_synapse', 'weight': chr2_str}
            for ni in range(n_pv):
                dist = nest.Distance(np.array([0.,0.]), targets[ni])
                if dist[0] < stim_range:
                    #print(dist[0], 'CONNECT!', nest.GetPosition(targets[ni]))
                    nest.Connect(source_chr2, targets[ni], syn_spec = chr_syn)
                elif stim_type == 'SST':
                    roll = rng.random()
                    if p_chr2_random > roll:
                        nest.Connect(source_chr2, targets[ni], syn_spec = chr_syn)
                        #print('random chr2 connect!')
    
            #print('STIM CONNECTIONS:', len(nest.GetConnections(source_chr2, targets)))
            #print(nest.GetStatus(source_chr2, 'spike_times'))
    
    ### Recording
    exc_sr = nest.Create('spike_recorder',)
    pv_sr = nest.Create('spike_recorder')
    sst_sr = nest.Create('spike_recorder')
    #noise_sr = nest.Create('spike_recorder')
    
    nest.Connect(exc_neurons,exc_sr)
    nest.Connect(pv_neurons, pv_sr)
    nest.Connect(sst_neurons, sst_sr)
    #nest.Connect(addnoise, noise_spikes)
    
    ### CHECK Connections ###
    if verbose:
        print('e2e:',len(nest.GetConnections(exc_neurons, exc_neurons)))
        print('e2p:',len(nest.GetConnections(exc_neurons, pv_neurons)))
        print('e2s:',len(nest.GetConnections(exc_neurons, sst_neurons)))
        print('p2e:',len(nest.GetConnections(pv_neurons, exc_neurons)))
        print('p2p:',len(nest.GetConnections(pv_neurons, pv_neurons)))
        print('s2e:',len(nest.GetConnections(sst_neurons, exc_neurons)))
        print('s2p:',len(nest.GetConnections(sst_neurons, pv_neurons)))
    
    simtime = nb_repeats * simtime + (nb_repeats - 1) * time_spacing + sim_delay
    nest.Simulate(simtime)
    
    ### Median FR ###
    exc_indices = np.arange(1,8001)
    pv_indices = np.arange(8001,9001)
    sst_indices = np.arange(9001,10001)
    
    
    exc_spikes = []
    exc_rates = []
    for ni in exc_indices:
        spike_ids = np.where(exc_sr.events['senders'] == ni)[0]
        spike_times = exc_sr.events['times'][spike_ids]
        exc_spikes.append(spike_times)
        nspikes = len(spike_times)
        exc_rates.append(nspikes / simtime * 1000.0)
    
    pv_spikes = []
    pv_rates = []
    for ni in pv_indices:
        spike_ids = np.where(pv_sr.events['senders'] == ni)[0]
        spike_times = pv_sr.events['times'][spike_ids]
        pv_spikes.append(spike_times)
        nspikes = len(spike_times)
        pv_rates.append(nspikes / simtime * 1000.0)
    
    sst_spikes = []
    sst_rates = []
    for ni in sst_indices:
        spike_ids = np.where(sst_sr.events['senders'] == ni)[0]
        spike_times = sst_sr.events['times'][spike_ids]
        sst_spikes.append(spike_times)
        nspikes = len(spike_times)
        sst_rates.append(nspikes / simtime * 1000.0)
    
    all_spikes = exc_spikes + pv_spikes + sst_spikes
    with open('%s/%s_spikes.pickle'%(result_dir,sim_name),'wb') as f:
        pickle.dump(all_spikes,f)
    
    all_positions = np.concatenate([np.array(nest.GetPosition(exc_neurons)), np.array(nest.GetPosition(pv_neurons)), np.array(nest.GetPosition(sst_neurons))])
    with open('%s/%s_positions.pickle'%(result_dir,sim_name), 'wb') as f:
        pickle.dump(all_positions, f)
    
    if verbose:
        print("EXC median and mean:", np.median(exc_rates), np.mean(exc_rates))
        print("PV median and mean:", np.median(pv_rates), np.mean(pv_rates))
        print("SST median and mean:", np.median(sst_rates), np.mean(sst_rates))

if __name__=="__main__":

    try:
        sim_parameters = read_sim_params('sim_parameters.txt')
    except FileNotFoundError:
        print('Provide sim_parameters.txt')
        sys.exit()
    
    contrast_values = [0.02, 0.05,0.1,0.18, 0.33]
    
    conditions = [['Control', c] for c in contrast_values] + [['Spont', 0.0]] 
    #condition = conditions[cond_i]
    
    try:
        rngseed = int(sys.argv[1])
        cond_i = int(sys.argv[2])
        circadian = sys.argv[3].upper()
        assert circadian in ('ZT0', 'ZT12')
        conn_mode = sys.argv[4].lower() if len(sys.argv) > 4 else DEFAULT_CONN_MODE
        # Optional 5th CLI arg: meaning depends on the chosen stim shape
        value_override = float(sys.argv[5]) if len(sys.argv) > 5 else None
        if value_override is not None:
            if USE_RECT_STIM:
                sim_parameters['stim_rect_width_mm'] = value_override  # runtime-only
            else:
                sim_parameters['stim_sigma_mm'] = value_override


    except IndexError:
        print("Provide two arguments: RNG seed and id for simulation conditions.")
        print("Second argument should be 0 for spontaneous conditions, 1-5 for PV stim, 6-10 for SST stim with contrasts 0.02, 0.05, 0.1, 0.18, and 0.33")
        sys.exit()

    #assert stim_type in ['SST', 'PV', 'Spont']
    #condition = [stim_type, contrast]
    condition = conditions[cond_i]
    
    
    result_dir = 'results_%s'%rngseed
    if not os.path.exists(result_dir):
        try:
            os.makedirs(result_dir)
        except:
            pass
    
    run_simulation(sim_parameters, condition, rngseed = rngseed, circadian = circadian)
    
