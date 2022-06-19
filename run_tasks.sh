cd ~/Documents/lxy/TimeSeriesSSL/ || exit
conda activate new
train(){
  local loop=true
  while $loop
  do
    loop=false
    {
      # python Train_crwu.py --gpuid=1 --r=0.5 --batch_size=256 --lambda_u=25 --p_threshold=0.5 --lr=0.01 --num_epochs=300
      python Train_crwu.py --gpuid=1 --r="$1" --batch_size=256 --lambda_u="$2" --p_threshold="$3" --lr=0.01 \
      --noise_mode="$4" --num_epochs=300
    } || {
      loop=true
    }
  done
}
# r=0.2
train 0.2 10 0.5 "sym"
train 0.2 20 0.5 "sym"
train 0.2 30 0.5 "sym"
train 0.2 20 0.2 "sym"
train 0.2 20 0.4 "sym"
train 0.2 20 0.6 "sym"
train 0.2 10 0.5 "asym"
train 0.2 20 0.5 "asym"
train 0.2 30 0.5 "asym"
train 0.2 20 0.2 "asym"
train 0.2 20 0.4 "asym"
train 0.2 20 0.6 "asym"
# r=0.5
train 0.5 10 0.5 "sym"
train 0.5 20 0.5 "sym"
train 0.5 30 0.5 "sym"
train 0.5 20 0.2 "sym"
train 0.5 20 0.4 "sym"
train 0.5 20 0.6 "sym"
train 0.5 10 0.5 "asym"
train 0.5 20 0.5 "asym"
train 0.5 30 0.5 "asym"
train 0.5 20 0.2 "asym"
train 0.5 20 0.4 "asym"
train 0.5 20 0.6 "asym"
# r=0.8
train 0.8 10 0.5 "sym"
train 0.8 20 0.5 "sym"
train 0.8 30 0.5 "sym"
train 0.8 20 0.2 "sym"
train 0.8 20 0.4 "sym"
train 0.8 20 0.6 "sym"
train 0.8 10 0.5 "asym"
train 0.8 20 0.5 "asym"
train 0.8 30 0.5 "asym"
train 0.8 20 0.2 "asym"
train 0.8 20 0.4 "asym"
train 0.8 20 0.6 "asym"
# r=0.9
train 0.9 10 0.5 "sym"
train 0.9 20 0.5 "sym"
train 0.9 30 0.5 "sym"
train 0.9 20 0.2 "sym"
train 0.9 20 0.4 "sym"
train 0.9 20 0.6 "sym"
train 0.9 10 0.5 "asym"
train 0.9 20 0.5 "asym"
train 0.9 30 0.5 "asym"
train 0.9 20 0.2 "asym"
train 0.9 20 0.4 "asym"
train 0.9 20 0.6 "asym"