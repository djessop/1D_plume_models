# 1D_plume_models

python plumeequations.py [plot_soln (0/1)] [souce_and_environmental_conditions_file (source_environmental_conditions.xlsx/source_environmental_conditions.json)]

Predicts the rise height, width etc. of experimental plumes.

To run, fill in fields in source_environmental_conditions.xlsx (or the json equivalent), or your own equivalent file.  These will be read and parsed by plumeequations.py, and a prediction made for the rise height.  If plotsolution is True, a plot of plume width, axial velocity, particle concentration and salinity will be shown.