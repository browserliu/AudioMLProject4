#!/bin/sh
#@uthor: Ankit Bahuguna
# nohup python scoring_GMM_UBM_final.py <number_of_gaussians> <fold> <en_ubm or de_ubm> >

# EN - UBM

nohup python scoring_GMM_UBM_final.py 64 5 en_ubm > logs/64_5_en.log &
nohup python scoring_GMM_UBM_final.py 64 15 en_ubm > logs/64_15_en.log &
nohup python scoring_GMM_UBM_final.py 64 30 en_ubm > logs/64_30_en.log &

nohup python scoring_GMM_UBM_final.py 128 5 en_ubm > logs/128_5_en.log &
nohup python scoring_GMM_UBM_final.py 128 15 en_ubm > logs/128_15_en.log &
nohup python scoring_GMM_UBM_final.py 128 30 en_ubm > logs/128_30_en.log &

nohup python scoring_GMM_UBM_final.py 256 5 en_ubm > logs/256_5_en.log &
nohup python scoring_GMM_UBM_final.py 256 15 en_ubm > logs/256_15_en.log &
nohup python scoring_GMM_UBM_final.py 256 30 en_ubm > logs/256_30_en.log &


wait

# DE - UBM

nohup python scoring_GMM_UBM_final.py 64 5 de_ubm > logs/64_5_de.log &
nohup python scoring_GMM_UBM_final.py 64 15 de_ubm > logs/64_15_de.log &
nohup python scoring_GMM_UBM_final.py 64 30 de_ubm > logs/64_30_de.log &

nohup python scoring_GMM_UBM_final.py 128 5 de_ubm > logs/128_5_de.log &
nohup python scoring_GMM_UBM_final.py 128 15 de_ubm > logs/128_15_de.log &
nohup python scoring_GMM_UBM_final.py 128 30 de_ubm > logs/128_30_de.log &

nohup python scoring_GMM_UBM_final.py 256 5 de_ubm > logs/256_5_de.log &
nohup python scoring_GMM_UBM_final.py 256 15 de_ubm > logs/256_15_de.log &
nohup python scoring_GMM_UBM_final.py 256 30 de_ubm > logs/256_30_de.log &

wait

nohup python scoring_GMM_UBM_final.py 512 5 en_ubm > logs/512_5_en.log &
nohup python scoring_GMM_UBM_final.py 512 15 en_ubm > logs/512_15_en.log &
nohup python scoring_GMM_UBM_final.py 512 30 en_ubm > logs/512_30_en.log &


nohup python scoring_GMM_UBM_final.py 512 5 de_ubm > logs/512_5_de.log &
nohup python scoring_GMM_UBM_final.py 512 15 de_ubm > logs/512_15_de.log &
nohup python scoring_GMM_UBM_final.py 512 30 de_ubm > logs/512_30_de.log &

echo "Done"