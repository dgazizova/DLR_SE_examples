import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

def get_SE(path_to_kernel, path_to_constants):
    df1 = pd.read_csv(path_to_kernel, delimiter="\t")
    df2 = pd.read_csv(path_to_constants, delimiter="\t")
    print(f"Kernel: {df1.head()}")
    print(f"Constants: {df2.head()}")

    nu = df1['nu'].unique()
    SE = np.zeros(len(nu), dtype=complex)

    for i, iw_ in enumerate(nu):
        I = df1[df1['nu'] == iw_]['Re_K'].to_numpy() + 1j * df1[df1['nu'] == iw_]['Im_K'].to_numpy()
        A = df2['Re_C'].to_numpy() + 1j * df2['Im_C'].to_numpy()
        SE[i] = np.sum(I * A)
    return SE


path_to_kernel = os.path.join(os.getenv('PATH_TO_CSV', ''), 'Kernel.csv')

# 2d Hubbard model
L = 11
k_ext_dict = {"[pi, pi]": "pi_pi", "[0, pi]": "0_pi"}
k_ext = k_ext_dict["[pi, pi]"]
path_to_constants = os.path.join(os.getenv('PATH_TO_CSV', ''), f'2d_Hubbard/Constant_L{L}_{k_ext}.csv')

SE = get_SE(path_to_kernel=path_to_kernel, path_to_constants=path_to_constants)
print(f"Self energy for 2d Hubbard k_ext = [pi, pi]: {SE}")

# H2 molecule
path_to_constants = os.path.join(os.getenv('PATH_TO_CSV', ''), f'H2/Constant_Sigma_11.csv')
SE = get_SE(path_to_kernel=path_to_kernel, path_to_constants=path_to_constants)
print(f"Self energy for H2 for Sigma_11: {SE}")




