#python Train_jn_fa.py --gpuid=0 --r=0.2 --batch_size=64 --lambda_u=0 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
#python Train_jn_fa.py --gpuid=0 --r=0.5 --batch_size=64 --lambda_u=0 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
#python Train_jn_fa.py --gpuid=0 --r=0.8 --batch_size=64 --lambda_u=50 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
#python Train_jn_fa.py --gpuid=0 --r=0.9 --batch_size=64 --lambda_u=50 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3

#python Train_crwu_fb.py --gpuid=0 --r=0.2 --batch_size=64 --lambda_u=0 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
#python Train_crwu_fb.py --gpuid=0 --r=0.5 --batch_size=64 --lambda_u=0 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
#python Train_crwu_fb.py --gpuid=0 --r=0.8 --batch_size=64 --lambda_u=50 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
#python Train_crwu_fb.py --gpuid=0 --r=0.9 --batch_size=64 --lambda_u=50 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3

python Train_jn_wo_aug.py --gpuid=0 --r=0.2 --batch_size=64 --lambda_u=0 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
python Train_jn_wo_aug.py --gpuid=0 --r=0.5 --batch_size=64 --lambda_u=0 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
python Train_jn_wo_aug.py --gpuid=0 --r=0.8 --batch_size=64 --lambda_u=50 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
python Train_jn_wo_aug.py --gpuid=0 --r=0.9 --batch_size=64 --lambda_u=50 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3

#python Train_jn_wo_cotrain.py --gpuid=0 --r=0.2 --batch_size=64 --lambda_u=0 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
#python Train_jn_wo_cotrain.py --gpuid=0 --r=0.5 --batch_size=64 --lambda_u=0 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
#python Train_jn_wo_cotrain.py --gpuid=0 --r=0.8 --batch_size=64 --lambda_u=50 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
#python Train_jn_wo_cotrain.py --gpuid=0 --r=0.9 --batch_size=64 --lambda_u=50 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3

#python Train_jn_wo_refine.py --gpuid=0 --r=0.2 --batch_size=64 --lambda_u=0 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
#python Train_jn_wo_refine.py --gpuid=0 --r=0.5 --batch_size=64 --lambda_u=0 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
python Train_jn_wo_refine.py --gpuid=0 --r=0.8 --batch_size=64 --lambda_u=50 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3
python Train_jn_wo_refine.py --gpuid=0 --r=0.9 --batch_size=64 --lambda_u=50 --p_threshold=0.5 --lr=0.01 --noise_mode="sym" --num_epochs=300 --alpha=0.5 --T=0.3