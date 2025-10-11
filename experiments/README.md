## Experiments with the OS2D methods (retail and INSTRE datasets)

### Preparations
```bash
# activate the env
conda activate os2d
# move to the root folder
# set OS2D_ROOT, e.g., by OS2D_ROOT=`pwd`
cd $OS2D_ROOT
export PYTHONPATH=$OS2D_ROOT:$PYTHONPATH
```

### Train models
```bash
# to use one local GPU run
python experiments/launcher_exp1.py
python experiments/launcher_exp2.py
python experiments/launcher_exp3_instre.py
# note that the first call will process the INSTRE dataset and create the cache file, this might cause crashes if done by deveral proceses in parallel, use --job-indices flag to run only some jobs first
```

### View logged information
```bash
# View all the saved logs in Visdom
python os2d/utils/plot_visdom.py --log_paths output/exp1
python os2d/utils/plot_visdom.py --log_paths output/exp2
python os2d/utils/plot_visdom.py --log_paths output/exp3
```

### Collect data for ablation tables
```bash
# Table 1:
python experiments/launcher_exp1_collect.py
# Table 2:
python experiments/launcher_exp2_collect.py
```

### Evaluation on test sets
```bash
# Retail product datasets: run eval
python experiments/launcher_grozi_eval.py
# Collect results (create a part of Table 3)
python launcher_grozi_eval_collect.py

# INSTRE datasets: run eval
python experiments/launcher_instre_eval.py
# Collect results (create a part of Table 4)
python launcher_instre_eval_collect.py
```


2025-05-19 13:24:34,326 OS2D.evaluate INFO: loss 0.0870, class_loss_per_element_detached_cpu 0.0000, loc_smoothL1 6.1797, cls_RLL 0.0870, cls_RLL_pos 0.0756, cls_RLL_neg 0.0114, mAP@0.50 0.0924, mAPw@0.50 0.0840, recall@0.50 0.1068, AP_joint_classes@0.50 0.0000, eval_time 2171.9054, 
2025-05-19 13:24:34,327 OS2D.train INFO: Init model is the current best on grozi-val-new-cl by mAP@0.50, value 0.0924