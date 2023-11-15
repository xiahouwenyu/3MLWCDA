name=/data/home/cwy/Science/3MLWCDA0.91/data/20210305_20230731_ihep_goodlist.root

# pp=(0.3364 0.3608 1 0.6841 0.4544 0.3304 0.2555 0.218 0.1967 0.1952 0.1968)
# pp=(0.84 0.64 0.5 0.44 0.36 0.3 0.42 )

# pp=(0.42 0.32 0.25 0.22 0.18 0.15 0.30 0.27 0.22 0.20 0.17 0.15)
pp=(0.4123 0.3111 0.2476 0.2064 0.1633 0.1379 0.2896 0.2594 0.2152 0.189 0.1561 0.14)

# for nn in {6}
# do
nn=7
# python3.9 convert_root2fits.py -i ${name}  -p ${pp[${nn}]}  -n ${nn}
python3.9 convert_root2fits.py -i ${name}  -p 0.2594  -n ${nn}
# done