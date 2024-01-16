num=1024
rm -rf ./skytxt/KM2A*txt
rm -rf *pixrun_noi.sh.err*
rm -rf *pixrun_noi.sh.out*
rm -rf ./output/*pixrun_noi.sh.err*
rm -rf ./output/*pixrun_noi.sh.out*
asub_eos "./pixrun_noi.sh -argu %{ProcId} -n ${num} -prio 999  -o ./output -e ./output"

num2=1024
rm -rf ./skytxt/WCDA*txt
rm -rf *pixrun_noi2.sh.err*
rm -rf *pixrun_noi2.sh.out*
rm -rf ./output/*pixrun_noi2.sh.err*
rm -rf ./output/*pixrun_noi2.sh.out*
asub_eos "./pixrun_noi2.sh -argu %{ProcId} -n ${num2} -prio 999 -o ./output -e ./output"