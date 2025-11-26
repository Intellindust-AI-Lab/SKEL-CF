## MOYO & EMDB & 3DPW & SPEC-SYN (3D metrics)

In the project root directory, run:

```bash
bash scripts/eval.sh
```

## COCO & H36M (2D & 3D metrics)
```bash
bash scripts/eval_hsmr.sh
```

By modifying the following fields in the script:
```
trainer.ckpt_path=''
exp_name=""
```
you can specify the checkpoint you want to evaluate and the directory where the evaluation logs will be stored.
