
import math
import numpy as np
import pandas as pd

# Create points 2 standard deviations apart around 0
def density_centers(df, num):
    std = np.nanstd(df.iloc[:,1:])
    return np.linspace(-std*num, std*num, num=num*2, endpoint=False)[1::2]

# Extracts a specific gene
def extract(df, name):
    return df[df.iloc[:, 0] == name].iloc[:, 1:].to_numpy()[0]

# Takes 2 arrays of the same length and drops all columns with NaN present
def drop_nan(x, y):
    combined = np.array(np.concatenate(([x],[y]), axis=0))
    combined = combined[:, ~pd.isna(combined).any(axis=0)]
    return combined[0], combined[1]

# Convert 2 vectors into a heatmap
def densitymap(x, y, xDensityCenters, yDensityCenters, xdiscrete=False, ydiscrete=False, sigma=1):
    if len(x) != len(y):
        return "inconsistent size of x and y vectors"

    # Initialize variables
    sigma_sq_inv = (1/sigma)**2
    mat = np.zeros((len(yDensityCenters), len(xDensityCenters)))

    # create density map depending on discreteness of x and y 
    if not xdiscrete and not ydiscrete:
        for pt in range(len(x)):
            temp = np.zeros((len(yDensityCenters), len(xDensityCenters)))
            for i, center_x in enumerate(xDensityCenters):
                for j, center_y in enumerate(yDensityCenters):
                    dist_sq = (x[pt] - center_x)**2+(y[pt] - center_y)**2
                    temp[j, i] = np.exp(-0.5*sigma_sq_inv*dist_sq)
                    
            temp /= np.sum(temp)
            mat += temp
            
    elif xdiscrete and ydiscrete:
        for i, center_x in enumerate(xDensityCenters):
            for j, center_y in enumerate(yDensityCenters):
                mat[j, i] += np.sum((x[y==center_y]==center_x))
                
    elif xdiscrete:
        for pt in range(len(x)):
            temp = np.zeros(len(yDensityCenters))
            for i, center_y in enumerate(yDensityCenters):
                dist_sq = (y[pt] - center_y)**2
                temp[i] = np.exp(-0.5*sigma_sq_inv*dist_sq)
                    
            temp /= np.sum(temp)
            mat[:, xDensityCenters.index(x[pt])] += temp
    else:
        for pt in range(len(y)):
            temp = np.zeros(len(xDensityCenters))
            for i, center_x in enumerate(xDensityCenters):
                dist_sq = (x[pt] - center_x)**2
                temp[i] = np.exp(-0.5*sigma_sq_inv*dist_sq)
                    
            temp /= np.sum(temp)
            mat[yDensityCenters.index(y[pt])] += temp
    
    # Normalize the kernel
    mat /= len(x) 
    return mat

# Main function that creates features based on datasets and pairs
def build_density_map(datasets, pairs, continuous, density_points):

    out = pd.DataFrame({'pair':[f'{p1}.{p2}' for p1, p2 in pairs]})

    for i in range(len(datasets)):
        for j in range(len(datasets)):
            
            # Initialize dataframes and variables
            df1 = datasets[i]
            df2 = datasets[j]
            mask = df1.columns.str.strip().isin(df2.columns.str.strip())
            mask[0] = True
            df1 = df1.loc[:, mask]
            mask = df2.columns.str.strip().isin(df1.columns.str.strip())
            mask[0] = True
            df2 = df2.loc[:, mask]
    
            df1_pts = density_points[i]
            df2_pts = density_points[j]

            df1_cont = continuous[i]
            df2_cont = continuous[j]

            temp = pd.DataFrame(index=range(len(out)), 
                                columns=[f'{datasets[i].name}.{datasets[j].name}.{value}' for value in range(len(df1_pts) * len(df2_pts))])

            #Calculate bandwidth
            if df1_cont:
                if df2_cont:
                    std = math.sqrt((np.nanstd(df1.iloc[:,1:].to_numpy())**2
                                     +np.nanstd(df2.iloc[:,1:].to_numpy())**2)/2)
                else:
                    std = np.nanstd(df1.iloc[:,1:].to_numpy())
            else:
                std = np.nanstd(df2.iloc[:,1:].to_numpy())

            # Insert density maps onto final matrix
            for index in range(len(pairs)):
                p1, p2 = pairs[index]
                x = extract(df1, p1)
                y = extract(df2, p2)
                x, y = drop_nan(x, y)
                mat = densitymap(x, y, df1_pts, df2_pts, xdiscrete=not df1_cont, ydiscrete=not df2_cont, sigma=std)
                temp.iloc[index] = mat.flatten()

            # Logarithmic transformation
            temp += 1/len(df1.columns)
            temp = temp.map(np.log)
            
            out = pd.concat([out, temp], axis=1)
    
    return out

# ---------------------------------------------------------------------

import argparse
from pathlib import Path
import configparser


parser = argparse.ArgumentParser()
parser.add_argument("config", help="configuation file input", type=Path)
parser.add_argument("pairs", help="csv file containing the pairs to make features of", type=Path)
args = parser.parse_args()

pairs = pd.read_csv(args.pairs).to_numpy()
config = configparser.ConfigParser()
config.read(args.config)


# datasets = gene_exp, copy_num, shRNA, gene_mut, CRISPR
# cont = True, False, True, False, True
# points = density_centers(gene_exp, 7), [0,1,2,3,4,6,8], density_centers(shRNA, 7), [0, 1], density_centers(CRISPR, 7)

# pos1_feat = build_density_map(datasets, pos1_t, cont, points)
# pos1_feat
