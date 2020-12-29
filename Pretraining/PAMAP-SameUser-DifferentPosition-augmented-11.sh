
win_size=(128)
window_overlap=(0.87)
activity=11

source1=(0 1 2)
source2=(0 1 2)
target=(0 1 2)
user=(1 2 5 8)

END=2

url=$PWD
file_url="/Pre-trained/PAMAP-SameUser-DifferentPosition-Augmented-11/PAMAP-SameUser-DifferentPosition-Augmented-11-verify.txt"
echo "S1_User S1_Pos S2_User S2_Pos Tar_User Tar_Pos A_Acc A_Precision A_F1 A_Recall B_Acc B_Precision B_F1 B_Recall" >> $url$file_url

for i in "${user[@]}"
do 
    for var1 in "${source1[@]}"
    do
        for var2 in "${source2[@]}"
        do
            for var3 in "${target[@]}"
            do
                if [ $var1 != $var2 -a $var2 != $var3 -a $var1 != $var3 -a $var2 -gt $var1 ]; then
                    echo "Executing User ${i} ${source1[$var1]} ${source2[$var2]} ${target[$var3]}"; 
                    for var4 in "${win_size[@]}"
                    do
                        for var5 in "${window_overlap[@]}"
                        do 
                            python PAMAP-SameUser-DifferentPosition-augmented-11.py --win_size ${var4}  --overlap ${var5}  --source1 ${source1[$var1]} --source2 ${source2[$var2]} --target ${target[$var3]} --activity $activity --user ${i}
                        done
                    done
                fi
            done
        done
    done
done