'''
Thought: 
    
    Go through all 3 second slices in the dataset, and print / listen and do the following: 
        * Listen to the sound
        * Plot the sound
        * Print the proposed label
        * Print the information contained in the gold standard 
            * What type of abnormality
            * How many
            * When they occur
        * press 'c' for crackle, 'n' for normal and 'w' for wheeze
        * Move on to next sample by pressing enter


Input = Lung sound file
    idx |   filename  |  start_idx  | stop_idx  | patient ID   | proposed label  


Output = Dataframe with the format: 
    idx |   filename  |  start_idx  | stop_idx  | patient ID   | label  
'''

multiple_present = {'insp_wheeze' : [f'sub_{i}_ob2_i_wh_number_insp_t72', f'sub_{i}_ob1_i_wh_number_insp_t72'],
                    'exp_wheeze': [f'sub_{i}_ob2_e_wh_number_exp_t72', f'sub_{i}_ob1_e_wh_number_exp_t72'],
                    'insp_crackle': [f'sub_{i}_ob2_i_cr_number_insp_t72', f'sub_{i}_ob1_i_cr_number_insp_t72'] , 
                    'exp_crackle' : [f'sub_{i}_ob2_e_cr_number_exp_t72', f'sub_{i}_ob1_e_cr_number_exp_t72']}