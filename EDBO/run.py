from edboplus.edbo.plus.optimizer_botorch import EDBOplus


# Define features to be used in the algorithm
columns = ['reductant_C', 'temperature', 'catalyst_C', 
       'cat2lig', 'solvent_dielectric constant', 'concentration', 'reductant_n_carbons',
       'reductant_n_beta_H', 'reductant_metal', 'reductant_isTurbo',
       'ligand_vmin_vmin_boltz', 'ligand_vbur_vbur_min', 'ligand_sterimol_burL_boltz']


EDBOplus().run(
    filename='scope_Z_round_6_solvents.csv',  # Previously generated scope.
    objectives=['yield', 'z%'],  # Objectives to be optimized.
    objective_mode=['max', 'max'],  # Maximize yield and ee but minimize side_product. CHANGE FOR E VS Z
    batch=5,  # Number of experiments in parallel that we want to perform in this round.
    columns_features= columns, # features to be included in the model.
    init_sampling_method='cvtsampling'  # initialization method.
)
