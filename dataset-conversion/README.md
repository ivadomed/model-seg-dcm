# Pre-training on multiple datasets

### Download datasets

Download T2w images and spinal cord segmentations for the following datasets.

```commandline
cd ~/duke/temp/janvalosek/ssl_pretraining_multiple_datasets
```

`spine-generic multi-subject` (n=267)

```commandline
git clone https://github.com/spine-generic/data-multi-subject
cd data-multi-subject
git checkout sb/156-add-preprocessed-images
git annex get $(find . -name "*space-other_T2w.nii.gz")
git annex get $(find . -name "*space-other_T2w_label-SC_seg.nii.gz")
```

`whole-spine` (n=45)

```commandline
git clone git@data.neuro.polymtl.ca:datasets/whole-spine
cd whole-spine
git annex dead here
git annex get $(find . -name "*T2w.nii.gz")
git annex get $(find . -name "*T2w_seg.nii.gz")
```

`canproco` (n=413)

```commandline
git clone git@data.neuro.polymtl.ca:datasets/canproco
cd canproco
git annex dead here
git annex get $(find . -name "*ses-M0_T2w.nii.gz")
git annex get $(find . -name "*ses-M0_T2w_seg-manual.nii.gz")
```

`sci-colorado` (n=80)

```commandline
git clone git@data.neuro.polymtl.ca:datasets/sci-colorado
cd sci-colorado
git annex dead here
git annex get $(find . -name "*T2w.nii.gz")
git annex get $(find . -name "*T2w_seg-manual.nii.gz")
```

`dcm-zurich` (n=135)

```commandline
git clone git@data.neuro.polymtl.ca:datasets/dcm-zurich
cd dcm-zurich
git annex dead here
git annex get $(find . -name "*acq-axial_T2w.nii.gz")
git annex get $(find . -name "*acq-axial_T2w_label-SC_mask-manual.nii.gz")
```

`sci-paris` (n=14)

```commandline
git clone git@data.neuro.polymtl.ca:datasets/sci-paris
cd sci-paris
git annex dead here
git annex get $(find . -name "*T2w.nii.gz")
git annex get $(find . -name "*T2w_seg.nii.gz")
```

### Create MSD-style JSON datalists

```commandline
python /Users/user/code/model-seg-dcm/vit_unetr_ssl/create_msd_data.py --path-data data-multi-subject --dataset-name spine-generic --path-out . --split 0.8 0.2 --seed 42
python /Users/user/code/model-seg-dcm/vit_unetr_ssl/create_msd_data.py --path-data whole-spine --dataset-name whole-spine --path-out . --split 0.8 0.2 --seed 42
python /Users/user/code/model-seg-dcm/vit_unetr_ssl/create_msd_data.py --path-data canproco --dataset-name canproco --path-out . --split 0.8 0.2 --seed 42
python /Users/user/code/model-seg-dcm/vit_unetr_ssl/create_msd_data.py --path-data sci-colorado --dataset-name sci-colorado --path-out . --split 0.8 0.2 --seed 42
python /Users/user/code/model-seg-dcm/vit_unetr_ssl/create_msd_data.py --path-data dcm-zurich --dataset-name dcm-zurich --path-out . --split 0.8 0.2 --seed 42
python /Users/user/code/model-seg-dcm/vit_unetr_ssl/create_msd_data.py --path-data sci-paris --dataset-name sci-paris --path-out . --split 0.8 0.2 --seed 42
```