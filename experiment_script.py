from LensClassifierExperiment import LensClassifierExperiment


sw_exp = LensClassifierExperiment(mode = "full_run")
for i, experiment_type in enumerate(sw_exp.valid_exp_names):
    _, to_plot, preds = sw_exp.run(experiment_type, parvals = sw_exp.suggested_parvals[i])
    sw_exp.plot_roc(experiment_type, to_plot)

