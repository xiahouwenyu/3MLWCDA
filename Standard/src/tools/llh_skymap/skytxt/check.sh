for i in {0..768}
do
if [ -f sig_no${i}.txt ]
then
echo ""
else
echo ${i}
fi
done