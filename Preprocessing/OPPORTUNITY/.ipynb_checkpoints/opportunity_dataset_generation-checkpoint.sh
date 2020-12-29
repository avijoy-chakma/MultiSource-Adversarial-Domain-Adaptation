
win_size=(64 32)
window_overlap=(0.70 0.80 0.90) 
source1=0
source2=1
target=2
activity=4

for var1 in "${win_size[@]}"
do
	for var2 in "${window_overlap[@]}"
  	do 
		python OPPORTUNITY-processing.py --win_size ${var1}  --window_overlap ${var2}  --source1 $source1 --source2 $source2 --target $target --activity $activity
	done
done