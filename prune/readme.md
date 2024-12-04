To run the code with flash attention:
In bash first run:
pip install flash-attn --no-cache-dir --target=$TMPDIR
This downloads flash-attention under scratch
Then export it:
export PYTHONPATH=$TMPDIR:$PYTHONPATH

First u can run trained_model_struct.py to see the structure
Then run the phi_prune.py to prune.
phi3_before_prune.py does finetuning without prune.
