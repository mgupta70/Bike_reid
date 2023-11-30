import shutil
from fastai.vision.all import *
import pickle
import gc

def find_bike_nobike_folders(main_dir):
    ''' input = main_dir as path
    output = list of folders'''
    
    all_bike_folders = []
    all_no_bike_folders = []
    for f in main_dir.ls():
        if f.is_dir():
            sub_f = f.iterdir()
            for item in sub_f:
                if item.name == 'Bicyclist':
                    all_bike_folders.append(item)
                elif item.name == 'No_Bicyclist':
                    all_no_bike_folders.append(item)
    
                        
    return all_bike_folders, all_no_bike_folders


def get_bike_nobike_df(main_dir):
    '''input = main_dir as path
    output = pandas df of bike and nobike counts'''
    
    df = pd.DataFrame( columns=['loc_id', 'Bicyclist', 'No_Bicyclist', 'Set'] )

    for f in main_dir.ls():
        if f.is_dir() and f.name[:3]=='Loc':
            num_bike = len(get_image_files(f/'Bicyclist'))
            num_nobike = len(get_image_files(f/'No_Bicyclist'))
            row_to_append = pd.DataFrame([{'loc_id':f.name, 'Bicyclist':num_bike, 'No_Bicyclist': num_nobike}])
            df = pd.concat([df,row_to_append])
            
    df.reset_index(drop=True, inplace=True)
    return df


def get_df_from_loc_nums(main_dir, loc_nums):
    '''input = main_dir as path
    output = pandas df of bike and nobike counts'''
    
    df = pd.DataFrame( columns=['loc_id', 'Bicyclist', 'No_Bicyclist', 'Set'] )

    for f in main_dir.ls():
        if f.is_dir() and f.name[-2:] in loc_nums:
            num_bike = len(get_image_files(f/'Bicyclist'))
            num_nobike = len(get_image_files(f/'No_Bicyclist'))
            row_to_append = pd.DataFrame([{'loc_id':f.name, 'Bicyclist':num_bike, 'No_Bicyclist': num_nobike}])
            df = pd.concat([df,row_to_append])
            
    df.reset_index(drop=True, inplace=True)
    return df



def plot_bike_nobike_data(df):
    '''input = pandas df with defined column names
    output = bar plot of bike no bike images'''
    
    # Create the vertical bar plot using pandas' DataFrame plot function
    spacing = 1.5 # set spacing
    bar_width = 0.4  # Set the width of the bars
    plt.figure(figsize=(12, 6))

    index = df.index
    bar1 = plt.bar(spacing*index, df['Bicyclist'], bar_width, label='Bicyclist')
    bar2 = plt.bar(spacing*index + bar_width, df['No_Bicyclist'], bar_width, label='No Bicyclist')

    # Rest of the plot customization
    plt.ylabel('Count')
    plt.xlabel('loc_id')
    plt.title('Bicyclist and No Bicyclist Data Grouped by loc_id')
    plt.xticks(spacing*index + bar_width / 2, df['loc_id'], rotation=90)
    plt.legend()

    # Add the counts above each bar
    for i, v in enumerate(df['Bicyclist']):
        plt.text(spacing*i, v, str(v), ha='center', va='bottom', fontweight='bold')

    for i, v in enumerate(df['No_Bicyclist']):
        plt.text(spacing*i + bar_width, v, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()
    
    
def make_balanced_dataset(main_dir, loc_nums, out_dir, create_new = False):
    ''' Inputs:
    main_dir - path type
    loc_nums - list type
    out_dir - path type
    '''
    
    if create_new == True:
    
        # Stage-1: Create ouput directory with subfolder structure
        for num in loc_nums:
            loc = 'Loc_'+num
            dest = out_dir/loc
            dest.mkdir(exist_ok=True, parents=True)

            subfolder_names= ['Bicyclist', 'No_Bicyclist']

            for sub in subfolder_names:
                dest = out_dir/loc/sub
                dest.mkdir(exist_ok=True, parents=True)

        # Stage-2: Sample equal number of nobike images as bike images
        ## Stage 2-1: Copy Bicyclists as is

        for num in loc_nums:
            loc = 'Loc_'+num
            src = main_dir/loc/'Bicyclist'
            bike_ims = get_image_files(src)
            num_bike_ims = len(bike_ims)
            dest = out_dir/loc/'Bicyclist'

            for file in bike_ims:       
                src_file = file
                dest_file = dest/file.name
                shutil.copy(src_file, dest_file)

            print('Moved {} bike images of Loc_{}'.format(num_bike_ims, num))

            ## Stage 2-2: Sample a subset of nobike images

            src = main_dir/loc/'No_Bicyclist'
            nobike_ims = get_image_files(src)
            num_nobike_ims = len(nobike_ims)
            dest = out_dir/loc/'No_Bicyclist'

            set_seed(42)
            idxs = torch.multinomial(torch.arange(len(nobike_ims)).float(),int(1*num_bike_ims)) # sampling exactly same number of images as num_bikes 
            sampled_nobike_ims = nobike_ims[idxs]

            for file in sampled_nobike_ims:       
                src_file = file
                dest_file = dest/file.name
                shutil.copy(src_file, dest_file)

            print('Moved {} nobike images of Loc_{}'.format(len(sampled_nobike_ims), num))
            
            
    else:
        print('Folders with Balanced datasets were already created')
        
           
        
def plot_sorted_bike_data(df, sort_by = 'Bicyclist', train_val = False):
    '''input: pandas df
    sort_by = Bicyclist or No_Bicyclist
    train_val = if pandas df has Set assigned'''
    
    spacing = 1.5 # set spacing
    bar_width = 0.4  # Set the width of the bars
    plt.figure(figsize=(12, 6))

    df_sorted = df.sort_values(by=[sort_by])
    index = df.index
    
    if train_val==True:
        colors = ['red' if set_value=='Train' else 'blue' for set_value in df_sorted['Set']]
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Train'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Val')])
        
    else:
        colors = ['red' for loc in df_sorted['loc_id']]
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label = sort_by)])

    
    bar1 = plt.bar(spacing*index, df_sorted[sort_by], bar_width, label=sort_by, color=colors)

    # Rest of the plot customization
    plt.ylabel('Count')
    plt.xlabel('loc_id')
    plt.title('{} Data Grouped by loc_id'.format(sort_by))
    plt.xticks(spacing*index + bar_width / 2, df_sorted['loc_id'], rotation=90)
    
    # Add the counts above each bar
    for i, v in enumerate(df_sorted[sort_by]):
        plt.text(spacing*i, v, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()
    


def save_pickle(var, name):
    ''' Saves a pickle file for variable `var` with `name`  '''
    with open('{}.pickle'.format(name), 'wb') as file:
        pickle.dump(var, file)
        file.close()
    
    
def load_pickle(name):
    ''' Loads a name file '''
    with open(name, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


def clear_gpu_cache():
    #print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()