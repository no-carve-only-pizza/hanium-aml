# hanium-aml

## First attack experiment

LFW is available through `data/raw/lfw` and contains 13,233 images from 5,749 people.

Create a small 100-image sample:

```bash
python src/make_sample.py
```

Run the first FGSM attack smoke test with pretrained ResNet-50:

```bash
python src/fgsm_resnet50.py
```

Outputs are written to:

```text
outputs/attacks/fgsm/images/
outputs/attacks/fgsm/perturbations/
outputs/attacks/fgsm/metadata.csv
```

Beginner note: this first experiment uses ImageNet pretrained ResNet-50 only to verify the attack pipeline. It is not an identity recognition model yet.

## Colab Pro workflow for targeted face attacks

Training should be run on Colab Pro when possible. The scripts automatically select CUDA when a Colab GPU is available, otherwise MPS on Mac, otherwise CPU.

Check the runtime first:

```bash
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Step 1. Prepare a 10-identity LFW dataset:

```bash
python src/prepare_lfw_identity_dataset.py
```

This creates:

```text
data/processed/lfw_identity_10/
  train/
  val/
  test/
  labels.json
```

Step 2. Fine-tune ResNet-50 as the face identity classifier:

```bash
python src/train_face_resnet50.py --epochs 8 --batch-size 32 --num-workers 2
```

For Colab Pro, if GPU memory is comfortable, try:

```bash
python src/train_face_resnet50.py --epochs 12 --batch-size 64 --num-workers 2
```

Checkpoint and metrics are saved to:

```text
checkpoints/face_resnet50_lfw10/best.pt
checkpoints/face_resnet50_lfw10/metrics.json
checkpoints/face_resnet50_lfw10/history.csv
```

Step 3. Run targeted FGSM against the trained face model:

```bash
python src/targeted_fgsm_face.py --epsilon 0.03 --limit 100
```

By default, each image targets the next identity label cyclically. To force every image toward one target identity, pass a class id:

```bash
python src/targeted_fgsm_face.py --epsilon 0.03 --limit 100 --target-class 0
```

Outputs are saved to:

```text
outputs/attacks/fgsm_face/images/
outputs/attacks/fgsm_face/perturbations/
outputs/attacks/fgsm_face/metadata_targeted_eps0.030.csv
```

Recommended first sweep after the model is trained:

```bash
for eps in 0.005 0.010 0.030 0.050; do
  python src/targeted_fgsm_face.py --epsilon "$eps" --limit 100
done
```

## Attack output index for defense integration

After running attacks in Colab, build one unified CSV for defense/web modules:

```bash
python src/build_attack_index.py
```

Output:

```text
outputs/attacks/attack_index.csv
```

Important columns:

```text
sample_id, attack, attack_family, file, adv_file, perturbation_file,
success, clean_correct, success_on_clean,
true_label, true_name, target_label, target_name,
pred_before_name, pred_after_name,
epsilon, theta, alpha, steps, max_queries, queries_used,
l0, l2, linf, time_sec, target_conf_gain
```

Defense modules should use `adv_file` as input and keep `sample_id` when writing defense results so attack/defense metrics can be joined later.

