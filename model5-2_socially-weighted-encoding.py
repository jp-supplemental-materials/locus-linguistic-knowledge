#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:55:31 2026

@author: author
"""

import random
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import copy
import scipy.optimize

# uncomment if you want random seeds to recall same values
#random.seed(89)
#np.random.seed(113)

start_time = time.perf_counter() #counter to track how long the script takes
####Model Input: Dictionary of lists with four word forms and their starting nasalance values.

def seed_clouds():
    #Nasalance values must be between 0 and 1. Prestige value (second value in the vector) initialized at 1.0
    #four categories correspond to oral (ORAL), nasal (NAS), oral with nasal suffix (ORAL-N), and nasal with nasal suffix (NAS-N)
    #each category has 2 lexical items. 
    exemplars = {'ORAL': {'tɑː': [0.2, 1.0], 'kiː': [0.2, 1.0]}, 
                       'NAS': {'tɑ̃ː': [0.6, 1.0], 'kĩː': [0.6, 1.0]}, 
                       'ORAL-N': {'tɑː-n': [0.2, 1.0], 'kiː-n': [0.2, 1.0]}, 
                       'NAS-N':{'tɑ̃ː-n': [0.6, 1.0], 'kĩː-n': [0.6, 1.0]}}
    # Make clouds for each form, each cloud has 1 initial sample
    # The values of each initial exemplar are decided by adding uniform noise
    sd = 0.025
    num_samples = 1
    exemplar_clouds = {}
    for cat, forms in exemplars.items():
        exemplar_clouds[cat] = {}
        for form, val in forms.items():
            samples = np.random.normal(loc=val, scale=sd, size = (num_samples, 2))
            samples[:,0] = np.clip(samples[:,0], 0, 1)
            exemplar_clouds[cat][form] = samples.tolist()
    return exemplar_clouds
exemplar_clouds = seed_clouds()

#define morphological relationships between forms in exemplar_clouds and add them to dict semantic_pairs for quick lookup of relations
semantic_relations = [('tɑː', 'tɑː-n'), ('kiː', 'kiː-n'), ('tɑ̃ː', 'tɑ̃ː-n'), ('kĩː', 'kĩː-n')]
semantic_pairs = {word: (word, related) for word, related in semantic_relations}
semantic_pairs.update({related: (related, word) for word, related in semantic_relations})

# Precompute a direct lookup dictionary to find a form's category efficiently
form_to_category = {word: cat for cat, forms in exemplar_clouds.items() for word in forms}

####Random noise function - ensures no two exemplars are the same
def random_noise(exemplar_val, exemplar_form, exemplar_cat, clouds):
    nas_mean = exemplar_val
    sd = 0.025     #standard deviation
    #generate a random noise value to add to the exemplar value
    exemplar_val_with_rand_noise = np.random.normal(loc=nas_mean, scale=sd)
    #set bounds of new_exemplar_val at (0, 1)
    return np.clip(exemplar_val_with_rand_noise, 0, 1)

####production bias to generate penalties for different production deviations
#phonetic nasalization bias: penalize pre-N vowels that aren't very nasalized
def preN_nasality_bias(exemplar_val, exemplar_cat, clouds):
    #determine mean value of nasal cloud
    nas_form_means = []

    for form in clouds['NAS']:
        cloud_array = np.asarray(clouds['NAS'][form], dtype=float)
        nas_form_means.append(np.mean(cloud_array[:, 0]))
        
    nas_mean = np.mean(nas_form_means) 
    
    if exemplar_cat.endswith('-N'):
        return (nas_mean - exemplar_val)**2
    else:
        return 0.0

#morphological bias:penalize morphologically related inflected forms that are far from their base in nasalance
def morphological_bias(exemplar_val, exemplar_form, exemplar_cat, clouds):
    if not exemplar_cat.endswith('-N'): #ensuring base forms are not drawn toward genitive forms by assigning a 0.0 penalty
        return 0.0
    related_form = semantic_pairs[exemplar_form][1] #assign morphologically related form to variable
    related_cat = form_to_category[related_form] 
    related_cloud = np.asarray(clouds[related_cat][related_form],dtype=float)
    sem_base_val = np.mean(related_cloud[:, 0])
    return (sem_base_val - exemplar_val)**2

#Define the frequency of each form. What approximate proportion of exemplars do you want from each form?
category_frequencies = [1, 1, 1, 1] #there are four categories in exemplar_clouds: [oral, nasal, oral-n, nasal-n]
form_frequencies = [1, 1] #there are two forms from each category in exemplar_clouds

#function to calculate constraint scalars based on current frequency of exemplar categories
def get_bias_scalars(form, category, clouds):
    #Step 1: Determine Frequency of morphologically related form (e.g., tɑː and tɑː-n)
    morph_base_form = semantic_pairs[form][1]
    morph_base_cat = form_to_category[morph_base_form]
    morph_base_form_freq = len(clouds[morph_base_cat][morph_base_form])

    
    #Step 2: Determine frequency of the current exemplar_form
    exemplar_form_freq = len(clouds[category][form])
    
    #Step 3: Determine the penalty scalars based on frequency (static for preN_nasality_constraint at this point)
    preN_bias_scalar = 5
    morph_bias_scalar = morph_base_form_freq / exemplar_form_freq
    
    return {
        'morph_base_form': morph_base_form,
        'morph_base_cat': morph_base_cat,
        'preN_bias_scalar': preN_bias_scalar,
        'morph_bias_scalar': morph_bias_scalar,
    }

####Objective function - calculates penalty of current exemplar_val in relation to biases
def objective(exemplar_val, exemplar_form, exemplar_cat, clouds):
    #Step 1: calculate bias scalars using get_bias_scalars function
    scalars = get_bias_scalars(exemplar_form, exemplar_cat, clouds)
    preN_bias_scalar = scalars['preN_bias_scalar']
    morph_bias_scalar = scalars['morph_bias_scalar']
    
    #Step 2: Calculate the pre-N bias penalty for the current exemplar
    preN_bias_pen = preN_nasality_bias(exemplar_val[0], exemplar_cat, clouds)
    #Step 3: Calculate the morphological bias penalty for the current exemplar
    morph_bias_pen = morphological_bias(exemplar_val[0], exemplar_form, exemplar_cat, clouds)
    
    #Step 3: Return the sum over all penalties
    return (preN_bias_scalar*preN_bias_pen + morph_bias_scalar*morph_bias_pen)

####Model Process
def exemplar_accumulation(clouds, num_exemplars):
    
    # Initialize an empty list to store (iteration, nasalance, prestige, word_form) tuples for plotting
    data_points = []
    for cat, words in clouds.items():
        for word, values in words.items():
            for value in values:
                #add initial values to data_points
                data_points.append((0, value[0], value[1], word))
    #store categories in clouds dictionary as a list for fast lookup
    categories = list(clouds.keys())
    #store individual words in dict of values associated with the category as the key
    semantic_forms = {}
    for cat in categories:
        semantic_forms[cat] = []
        for form in clouds[cat]:
            semantic_forms[cat].append(form) 
    
    #develop num_exemplars new exemplars
    for i in range(num_exemplars):
        
        ##Step 1 (SELECTION): create a new exemplar
        #choose one exemplar category for production from clouds dictionary based on the prespecified category frequencies
        new_exemplar_cat = random.choices(categories, weights = category_frequencies, k = 1)[0]
        #choose one form from that category for production based on the frequency of each form in that category
        new_exemplar_form = random.choices(semantic_forms[new_exemplar_cat], weights = form_frequencies, k = 1)[0]
       
        ##Step 2 (ENTRENCHMENT) make the mean value of the selected form the starting value for the new exemplar (entrenchment)
        selected_cloud = np.asarray(clouds[new_exemplar_cat][new_exemplar_form], dtype=float)
        #introducing high-nasality examplars after 400 iterations
        if (i > 400) and (i < 490) and (new_exemplar_cat == 'ORAL-N'):
            #determine mean value of nasal cloud
            nas_form_means = []

            for form in clouds['NAS']:
                cloud_array = np.asarray(clouds['NAS'][form], dtype=float)
                nas_form_means.append(np.mean(cloud_array[:, 0]))
                
            nas_mean = np.mean(nas_form_means) 
            new_exemplar_val = nas_mean + .15
            new_exemplar_prestige = np.random.normal(loc=100, scale=.025)
        else:
            new_exemplar_val = np.average(selected_cloud[:,0], weights = selected_cloud[:,1])
            new_exemplar_prestige = np.random.normal(loc=1, scale=.025)
        ##Step 3 (OPTIMIZATION): Optimize the new exemplar
        #restricting how much the exemplar val can change from biases to +/-.01 to minimize objective
        lower_bound = max(0, new_exemplar_val - .075) #max movement of +/- .075 is arbitrary at this point but meant to avoid one fell swoop changes
        upper_bound = min(1, new_exemplar_val + .075)
        bounds = [(lower_bound, upper_bound)]
        #define objective parameters
        optimized_exemplar_val = scipy.optimize.minimize(fun = lambda x: objective(x, new_exemplar_form, new_exemplar_cat, clouds),
                                                         x0 = [new_exemplar_val],
                                                         method = 'SLSQP',
                                                         bounds = bounds,
                                                         options={'maxiter':1000, #limit the maximum number of iterations to 1000
                                                                  'ftol':1e-4, #Sets the function tolerance to 1e-4, meaning the optimizer will stop when changes in the function value are smaller than this threshold.
                                                                  'eps':1e-5, #sets step size of the Jacobian (step_size = gradient * Learning_Rate)
                                                                  })
        
        new_exemplar_val = optimized_exemplar_val.x[0]
        
        ##Step 4 (NOISE): add random noise to the exemplar
        new_exemplar_val = random_noise(new_exemplar_val, new_exemplar_form, new_exemplar_cat, clouds)
        
        
        ##Step 5 (STORAGE): Categorize new exemplar into appropriate category
        new_exemplar = [new_exemplar_val, new_exemplar_prestige]
        clouds[new_exemplar_cat][new_exemplar_form].append(new_exemplar)
        
        ## Step 6 (DECAY)
        # Keep the cloud at no more than 50 exemplars.
        if len(clouds[new_exemplar_cat][new_exemplar_form]) > 50:

            current_cloud = clouds[new_exemplar_cat][new_exemplar_form]

            # Prestige is index 1 of each [nasality, prestige] exemplar.
            prestige_values = [exemplar[1] for exemplar in current_cloud]

            # Exemplars at or above this cutoff constitute the top 10%.
            prestige_cutoff = np.percentile(prestige_values, 80)

            # Find the oldest exemplar that is not in the top 10%.
            for index, exemplar in enumerate(current_cloud):
                if exemplar[1] < prestige_cutoff:
                    del current_cloud[index]
                    break
        
        #append new exemplar to data_points list for plotting
        data_points.append((i, new_exemplar_val, new_exemplar_prestige, new_exemplar_form))
        
    ##Final Step: calculate mean of each form's exemplar cloud
    final_exemplar_means = {
    cat: {
        form: np.round(np.mean(np.asarray(clouds[cat][form], dtype=float)[:, 0]), 3)  # Compute mean & round
        for form in clouds[cat]  # Iterate over words in each category
    }
    for cat in categories  # Iterate over categories
    }
    #output is a dictionary of nested dictionaries for each form as key and the mean of that exemplar cloud as the value. 
    return final_exemplar_means, data_points


#run the model n times using the multiple_trials function: num_trials specifies how many simulations of exemplar_accumulation() you want to run.
def multiple_trials(clouds, num_exemplars, num_trials):
    
    #Step 1: Create an empty list to store the means for each form after each iteration of exemplar_accumulation()
    trial_means = []
    
    #Step 2: Run n trials of exemplar_accumulation, plotting each trial and appending the results to trial_means
    trial_num = 0
    while trial_num < num_trials:
        trial_num += 1
        
        #deep copy original exemplar_clouds to avoid modification of external cloud values
        trial_clouds = copy.deepcopy(clouds)
        
        final_means, data_points = exemplar_accumulation(trial_clouds, num_exemplars)
        for cat, forms in final_means.items():
            for form, mean_nasalance in forms.items():
                trial_means.append({
                    'category': cat,
                    'form': form,
                    'mean_nasalance': mean_nasalance,
                    'iteration': trial_num})
                
        #plot the exemplar nasalance values over time
        #updating label colors and markers
        color_map = {
            'tɑː': '#fdc700', 'kiː': '#fdc700', 
            'tɑ̃ː': '#708090', 'kĩː': '#708090',
            'tɑː-n': '#006aad', 'kiː-n': '#006aad',
            'tɑ̃ː-n': '#961E25', 'kĩː-n': '#961E25'}
        marker_map = {
            'tɑː': 'o', 'kiː': 'o', 
            'tɑ̃ː': 'o', 'kĩː': 'o',
            'tɑː-n': '^', 'kiː-n': '^',
            'tɑ̃ː-n': '^', 'kĩː-n': '^'
            }

        plt.figure(figsize = (16, 10), facecolor = 'white')
        #preparing legend labels with mean values from final_means
        legend_labels = []

        for cat, words in final_means.items():
            for word, mean_val in words.items():
                #extract mean value and round to two decimals
                mean_rounded = round(mean_val, 2)
                legend_labels.append(f"{word} ({mean_rounded})")
                
                #get iteration, nasalance, and prestige values for this word
                iteration = [entry[0] for entry in data_points if entry[3] == word]
                nasalance = [entry[1] for entry in data_points if entry[3] == word]
                prestige = [entry[2] for entry in data_points if entry[3] == word]

                #Generating scatterplot
                plt.scatter(iteration, nasalance, label = word, alpha = 1, s=[75*np.sqrt(p) for p in prestige], color = color_map[word], marker = marker_map[word])

        plt.xlabel("Exemplar Production Iteration", fontweight = 'bold', fontsize = 32, color = "#003c6c", labelpad = 20)
        plt.ylabel("Nasality", fontweight = 'bold', fontsize = 32, color = "#003c6c", labelpad = 20)
        plt.tick_params(axis='both', labelsize=32, colors="#003c6c")
        plt.tight_layout()
        plt.gca().set_facecolor('white') #for changing background color of subplot
        legend = plt.legend(legend_labels, title = "words (μ nas.)", 
                   title_fontproperties=font_manager.FontProperties(weight="bold", size=32),
                   bbox_to_anchor = (1, 1), loc = 'upper left', edgecolor = "#003c6c", fontsize = 32, 
                   labelcolor = '#003c6c', facecolor = 'white')
        legend.get_title().set_color("#003c6c")
        plt.grid(False)
        ax = plt.gca()  # get current axes
        for spine in ax.spines.values():
            spine.set_edgecolor("#003c6c")
            spine.set_linewidth(2.5)
        plt.show()
        
    #Step 3: Convert trial_means to df and return the results
    exemplar_trials = pd.DataFrame(trial_means)
    return exemplar_trials


exemplar_trials = multiple_trials(exemplar_clouds, num_exemplars=1000, num_trials=1)

end_time = time.perf_counter()
print(f"Execution time: {round(end_time - start_time, 2)} seconds/{round((end_time - start_time)/60, 2)} minutes")

exemplar_trials.to_csv("model5-3-100SIMS.csv", index = False)