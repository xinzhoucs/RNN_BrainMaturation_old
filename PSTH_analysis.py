from __future__ import division
# basic packages #
import os
import numpy as np
import tensorflow as tf
import pickle
from collections import OrderedDict

# trial generation and network building #
from task import generate_trials, rule_name
from network import Model
import tools

# for ANOVA analysis and plot #
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
#from statsmodels.stats.multicomp import MultiComparison # for Tukey HSD Test
import matplotlib.pyplot as plt

# Colors used for PSTH_all
kelly_colors = \
[np.array([ 0.94901961,  0.95294118,  0.95686275]),
 np.array([ 0.13333333,  0.13333333,  0.13333333]),
 np.array([ 0.95294118,  0.76470588,  0.        ]),
 np.array([ 0.52941176,  0.3372549 ,  0.57254902]),
 np.array([ 0.95294118,  0.51764706,  0.        ]),
 np.array([ 0.63137255,  0.79215686,  0.94509804]),
 np.array([ 0.74509804,  0.        ,  0.19607843]),
 np.array([ 0.76078431,  0.69803922,  0.50196078]),
 np.array([ 0.51764706,  0.51764706,  0.50980392]),
 np.array([ 0.        ,  0.53333333,  0.3372549 ]),
 np.array([ 0.90196078,  0.56078431,  0.6745098 ]),
 np.array([ 0.        ,  0.40392157,  0.64705882]),
 np.array([ 0.97647059,  0.57647059,  0.4745098 ]),
 np.array([ 0.37647059,  0.30588235,  0.59215686]),
 np.array([ 0.96470588,  0.65098039,  0.        ]),
 np.array([ 0.70196078,  0.26666667,  0.42352941]),
 np.array([ 0.8627451 ,  0.82745098,  0.        ]),
 np.array([ 0.53333333,  0.17647059,  0.09019608]),
 np.array([ 0.55294118,  0.71372549,  0.        ]),
 np.array([ 0.39607843,  0.27058824,  0.13333333]),
 np.array([ 0.88627451,  0.34509804,  0.13333333]),
 np.array([ 0.16862745,  0.23921569,  0.14901961]),
 #add by yichen start
 np.array([ 0.70196078,  0.66666667,  0.42352941]),
 np.array([ 0.1627451 ,  0.82745098,  0.        ]),
 np.array([ 0.53333333,  0.17647059,  0.49019608]),
 np.array([ 0.55294118,  0.01372549,  0.        ]),
 np.array([ 0.39607843,  0.77058824,  0.13333333]),
 np.array([ 0.58627451,  0.94509804,  0.13333333]),
 np.array([ 0.16862745,  0.03921569,  0.74901961])
 # add by yichen end
 ]

class PSTH_Analysis(object):

    def __init__(self, model_dir):
        
        self.model_dir = model_dir
        self.log = tools.load_log(model_dir)
        self.hp = tools.load_hp(self.log['model_dir'])
        self.neurons = self.hp['n_rnn']
        self.print_basic_info()

    def print_basic_info(self,):

        print('rule trained: ', self.hp['rule_trains'])
        print('minimum trial number: 0')
        print('maximum trial number: ', self.log['trials'][-1])
        print('minimum trial step  : ', self.log['trials'][1])
        print('total number        : ', len(self.log['trials']))

        fig_pref = plt.figure()
        for rule in self.hp['rule_trains']:
            plt.plot(self.log['trials'], self.log['perf_'+rule], label = rule)
        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        plt.title('Growth of Performance')
        plt.show()
    
    def _compute_H(self, model, rule, trial, sess,):

        feed_dict = tools.gen_feed_dict(model, trial, self.hp)
        h = sess.run(model.h, feed_dict=feed_dict)

        fname = os.path.join(model.model_dir, 'H_'+rule+'.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(h, f)

    def compute_H(self,  
                rules=None, 
                trial_list=None, 
                recompute=False,):
        
        if rules is not None:
            self.rules = rules
        else:
            self.rules = self.hp['rule_trains']
        
        if trial_list is not None:
            self.trial_list = trial_list
        else:
            self.trial_list = self.log['trials']

        self.in_loc = dict()
        self.in_loc_set = dict()
        self.epoch_info = dict()

        trial_store = dict()
        #self.trial_store = dict()########################## do we really need self.?
        print("Epoch information:")
        for rule in self.rules:
            trial_store[rule] = generate_trials(rule, self.hp, 'test', noise_on=False)
            self.in_loc[rule] = np.array([np.argmax(i) for i in trial_store[rule].input_loc])
            self.in_loc_set[rule] = sorted(set(self.in_loc[rule]))
            self.epoch_info[rule] = trial_store[rule].epochs
            #self.trial_store[rule] = generate_trials(rule, self.hp, 'test', noise_on=False)
            #self.in_loc[rule] = np.array([np.argmax(i) for i in self.trial_store[rule].input_loc])
            print('\t'+rule+':')
            for e_name, e_time in self.epoch_info[rule].items():
                print('\t\t'+e_name+':',e_time)
        
        for trial_num in self.trial_list:
            sub_dir = self.model_dir+'/'+str(trial_num)+'/'
            for rule in self.rules:
                if recompute or not os.path.exists(sub_dir+'H_'+rule+'.pkl'):
                    model = Model(sub_dir, hp=self.hp)
                    with tf.Session() as sess:
                        model.restore()
                        self._compute_H(model, rule, trial_store[rule], sess,)

    def generate_neuron_info(self, 
                            epochs, 
                            trial_list = None, 
                            rules = None, 
                            norm = True, 
                            p_value = 0.05, 
                            abs_active_thresh = 1e-3,):
    
        self.neuron_info = OrderedDict()

        if trial_list is None:
            trial_list = self.trial_list
        if rules is None:
            rules = self.rules

        for rule in rules:
            for epoch in epochs:
                if epoch not in self.epoch_info[rule].keys():
                    raise KeyError('Rule ',rule,' dose not have epoch ',epoch,'!')
    
        for trial_num in trial_list:
            self.neuron_info[trial_num] = OrderedDict()

            for rule in rules:
                H = tools.load_pickle(self.model_dir+'/'+str(trial_num)+'/'+'H_'+rule+'.pkl')
                self.neuron_info[trial_num][rule] = OrderedDict()

                for epoch in epochs:
                    self.neuron_info[trial_num][rule][epoch] = OrderedDict()

                    for info_type in ['selective_neurons','active_neurons','exh_neurons','inh_neurons','mix_neurons',\
                        'firerate_loc_order','firerate_max_central']:
                        self.neuron_info[trial_num][rule][epoch][info_type] = list()

                    for neuron in range(self.neurons):
                        neuron_data_abs = OrderedDict()
                        neuron_data_norm = OrderedDict()
                        neuron_data = OrderedDict()
                        firerate_abs = list()
                        firerate_norm = list()
                        firerate = list()

                        for loc in self.in_loc_set[rule]:
                            fix_level = H[self.epoch_info[rule]['fix1'][0]:self.epoch_info[rule]['fix1'][1], \
                                self.in_loc[rule] == loc, neuron].mean(axis=1).mean(axis=0)
                            neuron_data_abs[loc] = H[self.epoch_info[rule][epoch][0]:self.epoch_info[rule][epoch][1],\
                                self.in_loc[rule] == loc, neuron].mean(axis=0) #axis = 1 for trial-wise mean, 0 for time-wise mean
                            neuron_data_norm[loc] = neuron_data_abs[loc]/fix_level-1
                            if norm:
                                neuron_data[loc] = neuron_data_norm[loc]
                            else:
                                neuron_data[loc] = neuron_data_abs[loc]
                            firerate_abs.append(neuron_data_abs[loc].mean())
                            firerate_norm.append(neuron_data_norm[loc].mean())
                            firerate.append(neuron_data[loc].mean())

                        data_frame = pd.DataFrame(neuron_data)
                        data_frame_melt = data_frame.melt()
                        data_frame_melt.columns = ['Location','Fire_rate']
                        model = ols('Fire_rate~C(Location)',data=data_frame_melt).fit()
                        anova_table = anova_lm(model, typ = 2)

                        if max(firerate_abs) > abs_active_thresh:
                            self.neuron_info[trial_num][rule][epoch]['active_neurons'].append(neuron)

                            if anova_table['PR(>F)'][0] <= p_value:
                                self.neuron_info[trial_num][rule][epoch]['selective_neurons'].append(neuron)

                                if max(firerate_norm) < 0:
                                    self.neuron_info[trial_num][rule][epoch]['inh_neurons'].append(neuron)
                                elif min(firerate_norm) >= 0:
                                #else:
                                    self.neuron_info[trial_num][rule][epoch]['exh_neurons'].append(neuron)
                                else:
                                    self.neuron_info[trial_num][rule][epoch]['mix_neurons'].append(neuron)

                        max_index = firerate.index(max(firerate))
                        temp_len = len(firerate)
                        if temp_len%2 == 0:
                            mc_len = temp_len + 1
                        else:
                            mc_len = temp_len

                        firerate_max_central = np.zeros(mc_len)
                        for i in range(temp_len):
                            new_index = (i-max_index+temp_len//2)%temp_len
                            firerate_max_central[new_index] = firerate[i]
                        if temp_len%2 == 0:
                            firerate_max_central[-1] = firerate_max_central[0]

                        self.neuron_info[trial_num][rule][epoch]['firerate_loc_order'].append(firerate)
                        self.neuron_info[trial_num][rule][epoch]['firerate_max_central'].append(firerate_max_central)

                    self.neuron_info[trial_num][rule][epoch]['firerate_loc_order'] = \
                        np.array(self.neuron_info[trial_num][rule][epoch]['firerate_loc_order'])
                    self.neuron_info[trial_num][rule][epoch]['firerate_max_central'] = \
                        np.array(self.neuron_info[trial_num][rule][epoch]['firerate_max_central'])

    def plot_tuning_feature(self, 
                            epochs,
                            rules = None, 
                            trial_list = None,
                            neuron_types = [('exh_neurons','mix_neurons'),('inh_neurons',)]):
        
        from scipy.optimize import curve_fit
        import math

        if trial_list is None:
            trial_list = self.trial_list
        if rules is None:
            rules = self.rules

        neuron_cate = OrderedDict()
        tuning_by_trial = OrderedDict() 

        for rule in rules:
            neuron_cate[rule] = OrderedDict()
            for epoch in epochs:
                neuron_cate[rule][epoch] = OrderedDict()
                tools.mkdir_p('figure/figure_'+self.model_dir+'/'+rule+'/'+epoch+'/')
                if epoch not in self.epoch_info[rule].keys():
                    raise KeyError('Rule ',rule,' dose not have epoch ',epoch,'!')

        for trial_num in trial_list:
            tuning_by_trial[trial_num] = OrderedDict()

            for rule in rules:
                tuning_by_trial[trial_num][rule] = OrderedDict()

                for epoch in epochs:
                    tuning_by_trial[trial_num][rule][epoch] = dict()

                    for key, value in self.neuron_info[trial_num][rule][epoch].items():
                        if 'neurons' in key:
                            print(key,"number:",len(value))
                            if key not in neuron_cate[rule][epoch].keys():
                                neuron_cate[rule][epoch][key] = [len(value)]
                            else:
                                neuron_cate[rule][epoch][key].append(len(value))

                    for type_pair in neuron_types:
                        type_pair_folder = 'figure/figure_'+self.model_dir+'/'+rule+'/'+epoch+'/'+'_'.join(type_pair)+'/'
                        tools.mkdir_p(type_pair_folder)
                        fig_by_epoch = plt.figure()
                        tuning_by_neuron = list()

                        for n_type in type_pair:
                            for neuron in self.neuron_info[trial_num][rule][epoch][n_type]:
                                tuning_by_neuron.append(self.neuron_info[trial_num][rule][epoch]['firerate_max_central'][neuron])

                        tuning_by_neuron = np.array(tuning_by_neuron)
                        plt.plot(np.arange(len(self.neuron_info[trial_num][rule][epoch]['firerate_max_central'][neuron])),
                        tuning_by_neuron.T)
            
                        plt.title("Rule:"+rule+" Epoch:"+epoch+" Trial:"+str(trial_num)+\
                            " Perf:"+str(self.log['perf_'+rule][trial_num//self.log['trials'][1]])[:4])

                        save_name_neuron = type_pair_folder+str(trial_num)+'.png'
                        plt.tight_layout()
                        plt.savefig(save_name_neuron, transparent=False, bbox_inches='tight')
                        plt.close(fig_by_epoch)

                        tuning_by_trial[trial_num][rule][epoch][type_pair] = tuning_by_neuron.mean(axis=0)
        
        for type_pair in neuron_types:
            for rule in rules:
                for epoch in epochs:
                    fig_tuning = plt.figure()
                    for trial_num in trial_list:
                        plt.plot(np.arange(len(tuning_by_trial[trial_num][rule][epoch][type_pair])),
                        tuning_by_trial[trial_num][rule][epoch][type_pair], 
                        label = str(trial_num)+' Perf:'+str(self.log['perf_'+rule][trial_num//self.log['trials'][1]])[:4])
                    
                    type_pair_folder = 'figure/figure_'+self.model_dir+'/'+rule+'/'+epoch+'/'+'_'.join(type_pair)+'/'
                    save_name_trial = type_pair_folder+'tuning_all_'+str(trial_list[0])+'to'+str(trial_list[-1])+'.png'
                    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                    plt.tight_layout()
                    plt.savefig(save_name_trial, transparent=False,bbox_inches='tight')
                    plt.close(fig_tuning)
        
        #tunning by growth and gaussian curve fit
        tuning_by_growth = OrderedDict()
        def gaussian(x, a,u, sig):
            return a*np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (sig * math.sqrt(2 * math.pi))
        
        for type_pair in neuron_types:
            tuning_by_growth[type_pair] = OrderedDict()

            for rule in rules:
                tuning_by_growth[type_pair][rule] = OrderedDict()

                for epoch in epochs:
                    tuning_by_growth[type_pair][rule][epoch] = OrderedDict()
                    for growth_key in ['less_than_I','I_to_Y','elder_than_Y']:
                        tuning_by_growth[type_pair][rule][epoch][growth_key]=dict()
                        for type_key in ['growth','tuning']:
                            tuning_by_growth[type_pair][rule][epoch][growth_key][type_key]=list()

                    for trial_num in trial_list:
                        growth = self.log['perf_'+rule][trial_num//self.log['trials'][1]]
                        if growth <= self.hp['infancy_target_perf']:
                            tuning_by_growth[type_pair][rule][epoch]['less_than_I']['growth'].append(growth)
                            tuning_by_growth[type_pair][rule][epoch]['less_than_I']['tuning'].\
                                append(tuning_by_trial[trial_num][rule][epoch][type_pair])
                        elif growth <= self.hp['young_target_perf']:
                            tuning_by_growth[type_pair][rule][epoch]['I_to_Y']['growth'].append(growth)
                            tuning_by_growth[type_pair][rule][epoch]['I_to_Y']['tuning'].\
                                append(tuning_by_trial[trial_num][rule][epoch][type_pair])
                        else:
                            tuning_by_growth[type_pair][rule][epoch]['elder_than_Y']['growth'].append(growth)
                            tuning_by_growth[type_pair][rule][epoch]['elder_than_Y']['tuning'].\
                                append(tuning_by_trial[trial_num][rule][epoch][type_pair])

                    for growth_key in ['less_than_I','I_to_Y','elder_than_Y']:
                        for type_key in ['growth','tuning']:
                            try:
                                tuning_by_growth[type_pair][rule][epoch][growth_key][type_key]=\
                                    np.array(tuning_by_growth[type_pair][rule][epoch][growth_key][type_key]).mean(axis=0)
                            except:
                                pass

        for type_pair in neuron_types:
            for rule in rules:
                for epoch in epochs:
                    fig_tuning_growth = plt.figure()
                    for growth_key in ['less_than_I','I_to_Y','elder_than_Y']:
                        
                        tuning_temp = tuning_by_growth[type_pair][rule][epoch][growth_key]['tuning']
                        temp_x = np.arange(len(tuning_temp))  ###############################################TODO:something wrong for anti saccade
                        gaussian_x = np.arange(-0.1,len(tuning_temp)-0.9,0.1)
                        #gaussian fit
                        paras , _ = curve_fit(gaussian,temp_x,tuning_temp+(-1)*np.min(tuning_temp),\
                            p0=[np.max(tuning_temp)+1,len(tuning_temp)//2,1])
                        gaussian_y = gaussian(gaussian_x,paras[0],paras[1],paras[2])-np.min(tuning_temp)*(-1)

                        tuning_by_growth[type_pair][rule][epoch][growth_key]['gaussian_fit']=gaussian_y

                        if growth_key == 'less_than_I':
                            color = 'green'
                        elif growth_key == 'I_to_Y':
                            color = 'blue'
                        else:
                            color = 'red'

                        plt.scatter(temp_x, tuning_temp, marker = '+',color = color, s = 70 ,\
                            label = growth_key+'_'+str(tuning_by_growth[type_pair][rule][epoch][growth_key]['growth'])[:4])
                        plt.plot(gaussian_x, gaussian_y, color=color,linestyle = '--',\
                            label = growth_key+'_'+str(tuning_by_growth[type_pair][rule][epoch][growth_key]['growth'])[:4])

                    plt.legend()
                    type_pair_folder = 'figure/figure_'+self.model_dir+'/'+rule+'/'+epoch+'/'+'_'.join(type_pair)+'/'
                    save_name_tun_growth = type_pair_folder+'tuning_growth_'+str(trial_list[0])+'to'+str(trial_list[-1])+'.png'
                    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                    plt.tight_layout()
                    plt.savefig(save_name_tun_growth, transparent=False,bbox_inches='tight')
                    plt.close(fig_tuning_growth)
                    #plt.show()



                    
        
        #plot neuron category changes
        for rule in rules:
            for epoch in epochs:
                fig_cate = plt.figure()
                for key,value in neuron_cate[rule][epoch].items():
                    plt.plot(trial_list, value, label = key)
                plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                plt.tight_layout()
                plt.savefig('figure/figure_'+self.model_dir+'/'+rule+'/'+epoch+'/'+'neuron_cate_'+str(trial_list[0])+'to'+str(trial_list[-1])+'.png',transparent=False,bbox_inches='tight')
                plt.close(fig_cate)


    def plot_PSTH(self, 
                epochs, 
                rules=None, 
                trial_list=None, 
                neuron_types=[('exh_neurons','mix_neurons')], 
                norm = True,
                separate_plot = False,
                fuse_rules = False):
        
        if trial_list is None:
            trial_list = self.trial_list
        if rules is None:
            rules = self.rules

        psth_to_plot = OrderedDict()

        for trial_num in trial_list:
            psth_to_plot[trial_num] = OrderedDict()
            for rule in rules:
                H = tools.load_pickle(self.model_dir+'/'+str(trial_num)+'/'+'H_'+rule+'.pkl')
                psth_to_plot[trial_num][rule] = OrderedDict()
                for epoch in epochs:
                    psth_to_plot[trial_num][rule][epoch] = OrderedDict()
                    for type_pair in neuron_types:
                        psth_to_plot[trial_num][rule][epoch][type_pair] = OrderedDict()
                        psth_neuron = list()
                        anti_dir_psth = list()

                        for n_type in type_pair:
                            
                            for neuron in self.neuron_info[trial_num][rule][epoch][n_type]:

                                sel_loc = np.argmax(self.neuron_info[trial_num][rule][epoch]['firerate_loc_order'][neuron])
                                anti_loc = (sel_loc+len(self.in_loc_set[rule])//2)%len(self.in_loc_set[rule])

                                psth_temp = H[:,self.in_loc[rule] == sel_loc, neuron].mean(axis=1)
                                fix_level = H[self.epoch_info[rule]['fix1'][0]:self.epoch_info[rule]['fix1'][1], \
                                    self.in_loc[rule] == sel_loc, neuron].mean(axis=1).mean(axis=0)
                                if len(self.in_loc_set[rule])%2:
                                    anti_dir_psth_temp = (H[:,self.in_loc[rule] == anti_loc, neuron].mean(axis=1)+\
                                        H[:,self.in_loc[rule] == (anti_loc+1), neuron].mean(axis=1))/2.0
                                else:
                                    anti_dir_psth_temp = H[:,self.in_loc[rule] == anti_loc, neuron].mean(axis=1)

                                anti_dir_psth_norm = anti_dir_psth_temp/fix_level-1
                                psth_norm = psth_temp/fix_level-1
                                if norm:
                                    psth_neuron.append(psth_norm)
                                    anti_dir_psth.append(anti_dir_psth_norm)
                                else:
                                    psth_neuron.append(psth_temp)
                                    anti_dir_psth.append(anti_dir_psth_temp)

                        try:
                            psth_to_plot[trial_num][rule][epoch][type_pair]['sel_dir'] = np.array(psth_neuron).mean(axis=0)
                        except:
                            pass
                        try:
                            psth_to_plot[trial_num][rule][epoch][type_pair]['anti_sel_dir'] = np.array(anti_dir_psth).mean(axis=0)
                        except:
                            pass

        for rule in rules:
            for epoch in epochs:
                for type_pair in neuron_types:
                    if not separate_plot:
                        fig_psth = plt.figure()
                    for trial_num in trial_list:

                        if separate_plot:
                            fig_psth = plt.figure()
                            color = None
                        else:
                            color = kelly_colors[(trial_list.index(trial_num)+1)%len(kelly_colors)]

                        try:
                            plt.plot(np.arange(len(psth_to_plot[trial_num][rule][epoch][type_pair]['sel_dir']))*self.hp['dt']/1000,
                            psth_to_plot[trial_num][rule][epoch][type_pair]['sel_dir'],label=str(trial_num)+'sel',color=color)
                        except:
                            pass

                        try:
                            plt.plot(np.arange(len(psth_to_plot[trial_num][rule][epoch][type_pair]['anti_sel_dir']))*self.hp['dt']/1000,
                            psth_to_plot[trial_num][rule][epoch][type_pair]['anti_sel_dir'],linestyle = '--',label=str(trial_num)+'anti',color=color)
                        except:
                            pass
                        if separate_plot:
                            plt.title("Rule:"+rule+" Epoch:"+epoch+" Neuron_type:"+"_".join(type_pair))
                            plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                            type_pair_folder = 'figure/figure_'+self.model_dir+'/'+rule+'/'+epoch+'/'+'_'.join(type_pair)+'/'
                            tools.mkdir_p(type_pair_folder)
                            plt.tight_layout()
                            plt.savefig(type_pair_folder+'PSTH-'+str(trial_num)+'.png',transparent=False,bbox_inches='tight')
                            plt.close(fig_psth)

                    if not separate_plot:
                        plt.title("Rule:"+rule+" Epoch:"+epoch+" Neuron_type:"+"_".join(type_pair))
                        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                        type_pair_folder = 'figure/figure_'+self.model_dir+'/'+rule+'/'+epoch+'/'+'_'.join(type_pair)+'/'
                        tools.mkdir_p(type_pair_folder)
                        plt.tight_layout()
                        plt.savefig(type_pair_folder+'PSTH_all_'+str(trial_list[0])+'to'+str(trial_list[-1])+'.png',
                        transparent=False,bbox_inches='tight')
                        plt.close(fig_psth)

        plot_by_growth = dict()
        for rule in rules:
            plot_by_growth[rule] = dict()

            for epoch in epochs:
                plot_by_growth[rule][epoch] = dict()

                for type_pair in neuron_types:
                    plot_by_growth[rule][epoch][type_pair] = dict()
                    plot_by_growth[rule][epoch][type_pair]['less_than_I'] = dict()
                    plot_by_growth[rule][epoch][type_pair]['I_to_Y'] = dict()
                    plot_by_growth[rule][epoch][type_pair]['elder_than_Y'] = dict()

                    for key in plot_by_growth[rule][epoch][type_pair].keys():
                        plot_by_growth[rule][epoch][type_pair][key]['sel'] = list()
                        plot_by_growth[rule][epoch][type_pair][key]['anti'] = list()
                        plot_by_growth[rule][epoch][type_pair][key]['growth'] = list()

                    for trial_num in trial_list:
                        growth = self.log['perf_'+rule][trial_num//self.log['trials'][1]]
                        if growth <= self.hp['infancy_target_perf']:
                            plot_by_growth[rule][epoch][type_pair]['less_than_I']['sel'].append(\
                                psth_to_plot[trial_num][rule][epoch][type_pair]['sel_dir'])
                            plot_by_growth[rule][epoch][type_pair]['less_than_I']['anti'].append(\
                                psth_to_plot[trial_num][rule][epoch][type_pair]['anti_sel_dir'])
                            plot_by_growth[rule][epoch][type_pair]['less_than_I']['growth'].append(growth)
                        elif growth <= self.hp['young_target_perf']:
                            plot_by_growth[rule][epoch][type_pair]['I_to_Y']['sel'].append(\
                                psth_to_plot[trial_num][rule][epoch][type_pair]['sel_dir'])
                            plot_by_growth[rule][epoch][type_pair]['I_to_Y']['anti'].append(\
                                psth_to_plot[trial_num][rule][epoch][type_pair]['anti_sel_dir'])
                            plot_by_growth[rule][epoch][type_pair]['I_to_Y']['growth'].append(growth)
                        else:
                            plot_by_growth[rule][epoch][type_pair]['elder_than_Y']['sel'].append(\
                                psth_to_plot[trial_num][rule][epoch][type_pair]['sel_dir'])
                            plot_by_growth[rule][epoch][type_pair]['elder_than_Y']['anti'].append(\
                                psth_to_plot[trial_num][rule][epoch][type_pair]['anti_sel_dir'])
                            plot_by_growth[rule][epoch][type_pair]['elder_than_Y']['growth'].append(growth)

                    for growth_key in plot_by_growth[rule][epoch][type_pair].keys():
                        for type_key, value in plot_by_growth[rule][epoch][type_pair][growth_key].items():
                            try:
                                plot_by_growth[rule][epoch][type_pair][growth_key][type_key] = np.array(value).mean(axis=0)
                            except:
                                pass

        for rule in rules:
            for epoch in epochs:
                for type_pair in neuron_types:
                    fig_psth = plt.figure()
                    for growth_key,value in plot_by_growth[rule][epoch][type_pair].items():
                        if growth_key == 'less_than_I':
                            color = 'green'
                        elif growth_key == 'I_to_Y':
                            color = 'blue'
                        else:
                            color = 'red'
                        
                        try:
                            plt.plot(np.arange(len(value['sel']))*self.hp['dt']/1000,value['sel'],
                            label = growth_key+'_'+str(value['growth'])[:4], color = color)
                        except:
                            pass

                        try:
                            plt.plot(np.arange(len(value['anti']))*self.hp['dt']/1000,value['anti'],
                            linestyle = '--',label = growth_key+'_'+str(value['growth'])[:4], color = color)
                        except:
                            pass
                    plt.title("Rule:"+rule+" Epoch:"+epoch+" Neuron_type:"+"_".join(type_pair))
                    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                    type_pair_folder = 'figure/figure_'+self.model_dir+'/'+rule+'/'+epoch+'/'+'_'.join(type_pair)+'/'
                    tools.mkdir_p(type_pair_folder)
                    plt.tight_layout()
                    plt.savefig(type_pair_folder+'PSTH_bygrowth_'+str(trial_list[0])+'to'+str(trial_list[-1])+'.png',
                    transparent=False,bbox_inches='tight')
                    plt.close(fig_psth)

        if fuse_rules:
            ls_list = ['-','--','-.',':']
            for type_pair in neuron_types:
                for epoch in epochs:
                    fig_psth_fr = plt.figure()
                    for rule, linestyle in zip(rules,ls_list[0:len(rules)]):
                        for key,value in plot_by_growth[rule][epoch][type_pair].items():
                            if key == 'less_than_I':
                                color = 'green'
                            elif key == 'I_to_Y':
                                color = 'blue'
                            else:
                                color = 'red' 

                            try:
                                plt.plot(np.arange(len(value['sel']))*self.hp['dt']/1000,value['sel'],
                                label = rule+'_'+key+'_'+str(value['growth'])[:4], color = color, linestyle=linestyle)
                            except:
                                pass

                    plt.legend()
                    plt.title("Rule:"+rule+" Epoch:"+epoch+" Neuron_type:"+"_".join(type_pair))
                    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                    plt.tight_layout()
                    plt.savefig('figure/figure_'+self.model_dir+'/'+'-'.join(rules)+'_'+epoch+'_'+'-'.join(type_pair)\
                        +str(trial_list[0])+'to'+str(trial_list[-1])+'.png',
                    transparent=False,bbox_inches='tight')
                    plt.close(fig_psth_fr)








if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--modeldir', type=str, default='data/6tasks')#add by yichen
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_dir = args.modeldir

    #start = 357120
    #end = 480000
    #step = 1280*12

    #for 6tasks folder for odr/odrd stage 
    #start = 520960
    #end = 628480
    #step = 1280*12

    #for odrd folder
    #start = 979200
    #end = 1048320
    #step = 1280*6

    #for 6tasks folder for anti saccade stage 
    start = 0
    end = 102400
    step = 1280*8

    #for 6tasks folder for middle stage
    #start = 101120
    #end = 497920
    #step = 1280*31

    PA = PSTH_Analysis(model_dir)
    #PA.compute_H(rules = ["odrd","odr"], trial_list=range(start,end+1,step), recompute=False)#####
    PA.compute_H(rules = ['overlap','zero_gap','gap'], trial_list=range(start,end+1,step), recompute=False)#####
    print('start G')
    #PA.generate_neuron_info(epochs = ['stim1','delay1'],)
    PA.generate_neuron_info(epochs = ['stim1',],)
    print('start P')
    #PA.plot_tuning_feature(epochs = ['stim1','delay1'],)
    #PA.plot_tuning_feature(epochs = ['stim1',],)
    #PA.plot_PSTH(epochs = ['stim1','delay1'], separate_plot=False,neuron_types=[('exh_neurons','mix_neurons'),('inh_neurons',)])
    PA.plot_PSTH(epochs = ['stim1',], separate_plot=False,fuse_rules=True)
    print('finish')
