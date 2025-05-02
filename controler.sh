

wrkdir="/g/schwab/Marco/repos/dinomitos"
container="/g/schwab/rheinnec/container_legacy/python_latest.sif"
logdir="/scratch/rheinnec/logs"

sbatch \
    -J "dinomito" \
    -t 72:00:00 \
    --mem 100000 \
    -e "$logdir/log_dinomitos.txt" \
    -o "$logdir/out_dinomitos.txt" \
    --wrap="singularity exec --bind /g/schwab $container python3 $wrkdir/main.py"


#singularity exec --bind /g/schwab $container python3 $wrkdir/main.py
squeue -u rheinnec