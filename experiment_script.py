from LensClassifierExperiment import LensClassifierExperiment


sw_exp_debug = LensClassifierExperiment(debug_mode = True)
for i, experiment_type in enumerate(sw_exp_debug.valid_exp_names):
    _, to_plot = sw_exp_debug.run(experiment_type, parvals = sw_exp_debug.suggested_parvals[i])
    sw_exp_debug.plot_roc(experiment_type, to_plot)


sw_exp = LensClassifierExperiment(debug_mode = False)
for experiment_type in sw_exp.valid_exp_names:
    _, to_plot = sw_exp.run(experiment_type, parvals = sw_exp.suggested_parvals[i])
    sw_exp.plot_roc(experiment_type, to_plot)