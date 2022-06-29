#cd ~/Documents/lxy/TimeSeriesSSL/ || exit
#conda activate new
#train(){
#  local loop=true
#  while $loop
#  do
#    loop=false
#    {
#      # python Train_crwu.py --gpuid=0 --r=0.5 --batch_size=256 --lambda_u=25 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
#      python Train_crwu.py --gpuid=0 --r="$1" --batch_size=256 --lambda_u="$2" --p_threshold="$3" --lr=0.01 \
#      --noise_mode="$4" --num_epochs=300
#    } || {
#      loop=true
#    }
#  done
#}

# baseline
#python baseline_jn.py --gpuid=0 --r=0.2 --batch_size=256 --lr=0.002 --noise_mode="sym" --num_epochs=300
#python baseline_jn.py --gpuid=0 --r=0.5 --batch_size=256 --lr=0.002 --noise_mode="sym" --num_epochs=300
#python baseline_jn.py --gpuid=0 --r=0.8 --batch_size=256 --lr=0.002 --noise_mode="sym" --num_epochs=300
#python baseline_jn.py --gpuid=0 --r=0.9 --batch_size=256 --lr=0.002 --noise_mode="sym" --num_epochs=300
#python baseline_jn.py --gpuid=0 --r=0.4 --batch_size=256 --lr=0.002 --noise_mode="asym" --num_epochs=300
#
## divide-mix
## r=0.2
#python Train_jn.py --gpuid=0 --r=0.2 --batch_size=256 --lambda_u=0 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
#python Train_jn.py --gpuid=0 --r=0.2 --batch_size=256 --lambda_u=10 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
#python Train_jn.py --gpuid=0 --r=0.2 --batch_size=256 --lambda_u=25 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
## r=0.5
#python Train_jn.py --gpuid=0 --r=0.5 --batch_size=256 --lambda_u=0 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
#python Train_jn.py --gpuid=0 --r=0.5 --batch_size=256 --lambda_u=10 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
#python Train_jn.py --gpuid=0 --r=0.5 --batch_size=256 --lambda_u=25 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
## r=0.8
#python Train_jn.py --gpuid=0 --r=0.8 --batch_size=256 --lambda_u=10 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
#python Train_jn.py --gpuid=0 --r=0.8 --batch_size=256 --lambda_u=25 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
#python Train_jn.py --gpuid=0 --r=0.8 --batch_size=256 --lambda_u=50 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
#python Train_jn.py --gpuid=0 --r=0.8 --batch_size=256 --lambda_u=100 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
## r=0.9
#python Train_jn.py --gpuid=0 --r=0.9 --batch_size=256 --lambda_u=10 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
#python Train_jn.py --gpuid=0 --r=0.9 --batch_size=256 --lambda_u=25 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
#python Train_jn.py --gpuid=0 --r=0.9 --batch_size=256 --lambda_u=50 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
#python Train_jn.py --gpuid=0 --r=0.9 --batch_size=256 --lambda_u=100 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300
## r=0.4 noise_mode="asym"
#python Train_jn.py --gpuid=0 --r=0.4 --batch_size=256 --lambda_u=0 --p_threshold=0.5 --lr=0.01 --noise_mode="asym" --num_epochs=300
#python Train_jn.py --gpuid=0 --r=0.4 --batch_size=256 --lambda_u=10 --p_threshold=0.5 --lr=0.01 --noise_mode="asym" --num_epochs=300
#python Train_jn.py --gpuid=0 --r=0.4 --batch_size=256 --lambda_u=25 --p_threshold=0.5 --lr=0.01 --noise_mode="asym" --num_epochs=300
#python Train_jn.py --gpuid=0 --r=0.4 --batch_size=256 --lambda_u=50 --p_threshold=0.5 --lr=0.01 --noise_mode="asym" --num_epochs=300

# co-teaching+
#python Train_jn_cotea.py --gpuid=0 --r=0.2 --noise_mode='sym'
#python Train_jn_cotea.py --gpuid=0 --r=0.5 --noise_mode='sym'
#python Train_jn_cotea.py --gpuid=0 --r=0.8 --noise_mode='sym'
#python Train_jn_cotea.py --gpuid=0 --r=0.9 --noise_mode='sym'
#python Train_jn_cotea.py --gpuid=0 --r=0.4 --noise_mode='asym'

# pencil
#python Train_jn_pencil.py --gpuid=0 --r=0.2 --noise_mode="sym"
#python Train_jn_pencil.py --gpuid=0 --r=0.5 --noise_mode="sym"
#python Train_jn_pencil.py --gpuid=0 --r=0.8 --noise_mode="sym"
#python Train_jn_pencil.py --gpuid=0 --r=0.9 --noise_mode="sym"
#python Train_jn_pencil.py --gpuid=0 --r=0.4 --noise_mode="asym"

# mlnt
#python Train_jn_mlnt.py --gpuid=0 --r=0.2 --noise_mode="sym"
#python Train_jn_mlnt.py --gpuid=0 --r=0.5 --noise_mode="sym"
#python Train_jn_mlnt.py --gpuid=0 --r=0.8 --noise_mode="sym"
#python Train_jn_mlnt.py --gpuid=0 --r=0.9 --noise_mode="sym"
#python Train_jn_mlnt.py --gpuid=0 --r=0.4 --noise_mode="asym"

#python Train_jn.py --gpuid=0 --r=0.2 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.3
#python Train_jn.py --gpuid=0 --r=0.2 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.5
#python Train_jn.py --gpuid=0 --r=0.2 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.7
#python Train_jn.py --gpuid=0 --r=0.2 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.3
#python Train_jn.py --gpuid=0 --r=0.2 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.5
#python Train_jn.py --gpuid=0 --r=0.2 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.7
#
#python Train_jn.py --gpuid=0 --r=0.5 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.3
#python Train_jn.py --gpuid=0 --r=0.5 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.5
#python Train_jn.py --gpuid=0 --r=0.5 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.7
#python Train_jn.py --gpuid=0 --r=0.5 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.3
#python Train_jn.py --gpuid=0 --r=0.5 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.5
#python Train_jn.py --gpuid=0 --r=0.5 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.7
#
#python Train_jn.py --gpuid=0 --r=0.8 --batch_size=64 --lambda_u=50  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.3
#python Train_jn.py --gpuid=0 --r=0.8 --batch_size=64 --lambda_u=50  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.5
#python Train_jn.py --gpuid=0 --r=0.8 --batch_size=64 --lambda_u=50  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.7
#python Train_jn.py --gpuid=0 --r=0.8 --batch_size=64 --lambda_u=50  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.3
#python Train_jn.py --gpuid=0 --r=0.8 --batch_size=64 --lambda_u=50  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.5
#python Train_jn.py --gpuid=0 --r=0.8 --batch_size=64 --lambda_u=50  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.7
#
#python Train_jn.py --gpuid=0 --r=0.9 --batch_size=64 --lambda_u=50  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.3
#python Train_jn.py --gpuid=0 --r=0.9 --batch_size=64 --lambda_u=50  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.5
#python Train_jn.py --gpuid=0 --r=0.9 --batch_size=64 --lambda_u=50  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.7
#python Train_jn.py --gpuid=0 --r=0.9 --batch_size=64 --lambda_u=50  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.3
#python Train_jn.py --gpuid=0 --r=0.9 --batch_size=64 --lambda_u=50  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.5
#python Train_jn.py --gpuid=0 --r=0.9 --batch_size=64 --lambda_u=50  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.7
#
#python Train_jn.py --gpuid=0 --r=0.4 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.3 --noise_mode="asym"
#python Train_jn.py --gpuid=0 --r=0.4 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.5 --noise_mode="asym"
#python Train_jn.py --gpuid=0 --r=0.4 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=0.5 --T=0.7 --noise_mode="asym"
#python Train_jn.py --gpuid=0 --r=0.4 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.3 --noise_mode="asym"
#python Train_jn.py --gpuid=0 --r=0.4 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.5 --noise_mode="asym"
#python Train_jn.py --gpuid=0 --r=0.4 --batch_size=64 --lambda_u=0  --p_threshold=0.5 --lr=0.01 --alpha=4 --T=0.7 --noise_mode="asym"